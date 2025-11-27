import os
from dataclasses import dataclass
import json
import torch
import evaluate
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# ======================================================
# Configuration
# ======================================================

@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT: str = "input/gnhk_dataset"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ======================================================
# Load CSV annotations
# ======================================================

def load_dataframes():
    train_csv = os.path.join(DatasetConfig.DATA_ROOT, "train_processed.csv")
    test_csv = os.path.join(DatasetConfig.DATA_ROOT, "test_processed.csv")

    train_df = pd.read_csv(train_csv, header=None, skiprows=1,
                           names=["image_filename", "text"])
    test_df = pd.read_csv(test_csv, header=None, skiprows=1,
                          names=["image_filename", "text"])
    return train_df, test_df


# ======================================================
# Dataset class
# ======================================================

class CustomOCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df.copy()
        self.processor = processor
        self.max_target_length = max_target_length
        self.df["text"] = self.df["text"].fillna("")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.loc[idx, "image_filename"]
        text = self.df.loc[idx, "text"]

        image_path = os.path.join(self.root_dir, file_name)
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length
        ).input_ids

        labels = [
            token if token != self.processor.tokenizer.pad_token_id else -100
            for token in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ======================================================
# CER Metric
# ======================================================
import numpy as np
cer_metric = evaluate.load("cer")

def compute_cer_builder(processor):
    def compute_cer(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Handle case where predictions come as a tuple (e.g., (logits, ...))
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.array(pred_ids)
        # If we got logits: [batch, seq_len, vocab_size] -> take argmax over vocab dim
        if pred_ids.ndim == 3:
            pred_ids = pred_ids.argmax(axis=-1)

        # Ensure integer type
        pred_ids = pred_ids.astype("int64")

        # IMPORTANT: map any negative prediction IDs (e.g. -100) to pad_token_id
        pred_ids[pred_ids < 0] = processor.tokenizer.pad_token_id

        # Decode predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        # ---- Fix labels the usual way ----
        label_ids = np.array(label_ids)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_cer

# ======================================================
# Main: Evaluate pretrained TrOCR
# ======================================================

def main():

    print("Loading CSV...")
    train_df, test_df_full = load_dataframes()

    split_idx = len(test_df_full) // 2
    val_df = test_df_full.iloc[:split_idx].reset_index(drop=True)
    test_df = test_df_full.iloc[split_idx:].reset_index(drop=True)
    train_root = os.path.join(DatasetConfig.DATA_ROOT, "train_processed", "images")
    test_root = os.path.join(DatasetConfig.DATA_ROOT, "test_processed", "images")

    # Load baseline model for comparison.
    print("Loading PRETRAINED TrOCR model...")
    # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    # model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten").to(device)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    # ===============================================

    compute_cer = compute_cer_builder(processor)

    val_dataset = CustomOCRDataset(test_root, val_df, processor)
    test_dataset = CustomOCRDataset(test_root, test_df, processor)
    train_dataset = CustomOCRDataset(train_root, train_df, processor)

    eval_args = Seq2SeqTrainingArguments(
        output_dir="eval_pretrained",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=eval_args,
        compute_metrics=compute_cer,
        data_collator=default_data_collator,
    )

    # # 1. VALIDATION
    # print("\n===== Pretrained Model: VALIDATION CER =====")
    # val_results = trainer.evaluate(eval_dataset=val_dataset)
    # print("Validation CER =", val_results["eval_cer"])
    # #
    # # 2. TEST
    # print("\n===== Pretrained Model: TEST CER =====")
    # test_results = trainer.evaluate(eval_dataset=test_dataset)
    # print("Test CER =", test_results["eval_cer"])
    #
    # # 3. TRAIN (optional)
    # print("\n===== Pretrained Model: TRAIN SET CER =====")
    # train_results = trainer.evaluate(eval_dataset=train_dataset)
    # print("Train CER =", train_results["eval_cer"])
    #
    # print("\nDone evaluating pretrained TrOCR.")


    # ===============================================================
    # Create raw_notes_baseline.json (same structure as GT/PRED notes)
    # ===============================================================

    print("\nGenerating baseline raw notes (raw_notes_baseline.json)...")
    total_items = len(test_df)
    baseline_notes = {}
    model.eval()
    for idx, row in test_df.iterrows():
        filename = row["image_filename"]
        gt_word = row["text"]
        prefix = filename.rsplit("_", 1)[0].replace(".jpg", "")
        # Initialize list if first time
        if prefix not in baseline_notes:
            baseline_notes[prefix] = []
        # Load image
        img_path = os.path.join(test_root, filename)
        image = Image.open(img_path).convert("RGB")
        # Predict
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        pred_ids = model.generate(pixel_values)[0]
        pred_word = processor.batch_decode([pred_ids], skip_special_tokens=True)[0]
        baseline_notes[prefix].append(pred_word)
        # ----- Logging -----
        if (idx + 1) % 100 == 0 or (idx + 1) == total_items:
            pct = ((idx + 1) / total_items) * 100
            print(f"Processed {idx + 1}/{total_items} images ({pct:.1f}%) | "
                  f"Unique notes so far: {len(baseline_notes)}")

    # Save baseline predictions
    with open("raw_notes_baseline.json", "w") as f:
        json.dump(baseline_notes, f, indent=2)

    print("Saved: raw_notes_baseline.json")


if __name__ == "__main__":
    main()
