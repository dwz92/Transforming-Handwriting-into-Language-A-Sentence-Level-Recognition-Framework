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
    default_data_collator,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ======================================================
# Configuration (matching the training script)
# ======================================================

@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT: str = "input/gnhk_dataset"
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
# Dataset class (same as training script)
# ======================================================

class CustomOCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
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
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ======================================================
# CER Metric
# ======================================================
cer_metric = evaluate.load("cer")

def compute_cer_builder(processor):
    def compute_cer(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_cer


# ======================================================
# Main: Load saved model + evaluate
# ======================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading dataset...")
    train_df, test_df_full = load_dataframes()

    # ---------- Split test_df into validation and test ----------
    split_idx = len(test_df_full) // 2
    val_df = test_df_full.iloc[:split_idx].reset_index(drop=True)
    test_df = test_df_full.iloc[split_idx:].reset_index(drop=True)
    # ------------------------------------------------------------
    # print(val_df.head)
    train_root = os.path.join(DatasetConfig.DATA_ROOT, "train_processed", "images")
    test_root = os.path.join(DatasetConfig.DATA_ROOT, "test_processed", "images")

    print("Loading saved model and processor...")
    processor = TrOCRProcessor.from_pretrained("saved_model")
    model = VisionEncoderDecoderModel.from_pretrained("saved_model").to(device)

    # Build datasets
    val_dataset = CustomOCRDataset(test_root, val_df, processor)
    test_dataset = CustomOCRDataset(test_root, test_df, processor)
    train_dataset = CustomOCRDataset(train_root, train_df, processor)

    compute_cer = compute_cer_builder(processor)

    eval_args = Seq2SeqTrainingArguments(
        output_dir="eval_temp",
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

    # # ========= 1. Evaluate on VALIDATION set =========
    # print("Evaluating on VALIDATION split...")
    # val_results = trainer.evaluate(eval_dataset=val_dataset)
    # print("Validation CER:", val_results["eval_cer"])
    #
    # ========= 2. Evaluate on TEST set =========
    # print("Evaluating fine-tuned model on TEST split...")
    # test_results = trainer.evaluate(eval_dataset=test_dataset)
    # print("Test CER:", test_results["eval_cer"])

    # ========= 3. Evaluate on TRAIN set =========
    # print("Evaluating on FULL TRAINING SET...")
    # train_results = trainer.evaluate(eval_dataset=train_dataset)
    # print("Train CER:", train_results["eval_cer"])

    # ==========================================================
    # Create TWO note dictionaries:
    #   1. notes_gt.json      → ground truth grouped by prefix
    #   2. notes_pred.json    → model predictions grouped by prefix
    # ==========================================================

    # ----- Always start fresh -----
    if os.path.exists("raw_notes_gt.json"):
        os.remove("raw_notes_gt.json")
    if os.path.exists("raw_notes_pred.json"):
        os.remove("raw_notes_pred.json")

    notes_gt = {}
    notes_pred = {}

    print("Creating notes for GT and predictions...")
    model.eval()

    for idx, row in test_df.iterrows():
        filename = row["image_filename"]
        gt_word = row["text"]

        prefix = filename.rsplit("_", 1)[0].replace(".jpg", "")

        # ---- Initialize prefix lists (fresh generation) ----
        if prefix not in notes_gt:
            notes_gt[prefix] = []
        if prefix not in notes_pred:
            notes_pred[prefix] = []

        # ---- Append GT word ----
        notes_gt[prefix].append(gt_word)

        # ---- Predict word ----
        img_path = os.path.join(test_root, filename)
        image = Image.open(img_path).convert("RGB")

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        pred_ids = model.generate(pixel_values)[0]
        pred_word = processor.batch_decode([pred_ids], skip_special_tokens=True)[0]

        notes_pred[prefix].append(pred_word)

    # ----- Save new results -----
    with open("raw_notes_gt.json", "w") as f:
        json.dump(notes_gt, f, indent=2)

    with open("raw_notes_pred.json", "w") as f:
        json.dump(notes_pred, f, indent=2)

    print("Saved fresh raw_notes_gt.json and raw_notes_pred.json")

    # Save back
    with open("raw_notes_gt.json", "w") as f:
        json.dump(notes_gt, f, indent=2)

    with open("raw_notes_pred.json", "w") as f:
        json.dump(notes_pred, f, indent=2)

    print("Saved: raw_notes_gt.json and raw_notes_pred.json")


if __name__ == "__main__":
    main()
