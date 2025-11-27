import os
from dataclasses import dataclass

import torch
import evaluate
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

from transformers import TrainerCallback

class ValidationLoggerCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        if "eval_cer" not in metrics:
            test_cer = metrics.get("test_cer")
            test_loss = metrics.get("test_loss")
            if (test_cer is not None) or (test_loss is not None):
                print(f"[Logger] TEST evaluation → loss={test_loss}, CER={test_cer}")

                # ---- Write TEST results to log file ----
                with open("log.txt", "a") as f:
                    f.write(f"[TEST] loss={test_loss}, CER={test_cer}\n")
            return  # Do NOT fall through to validation block
        # -------- This is VALIDATION evaluation (during training) --------
        cer = metrics.get("eval_cer")
        epoch = state.epoch

        # Get latest epoch-level training loss
        train_loss = None
        for record in reversed(state.log_history):
            if "loss" in record and "epoch" in record:
                train_loss = record["loss"]
                break

        # Write validation log
        with open("log.txt", "a") as f:
            f.write(f"Epoch {epoch:.0f} | Train Loss: {train_loss} | Val CER: {cer}\n")

        print(f"[Logger] Epoch {epoch:.0f} logged: loss={train_loss}, cer={cer}")

# ===================================
# Configuration
# ===================================

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 24 # Remember to adjust this batch size
    EPOCHS: int = 1
    LEARNING_RATE: float = 5e-5


@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT: str = "input/gnhk_dataset"


@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = "microsoft/trocr-small-handwritten"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===================================
# Pre-download everything
# ===================================

def download_dependencies():
    print("===== PRE-DOWNLOAD START =====")

    # 1. Download processor
    print("Downloading processor...")
    _ = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)

    # 2. Download model
    print("Downloading model...")
    _ = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)

    # 3. Download CER metric
    print("Downloading CER metric...")
    _ = evaluate.load("cer")

    print("===== PRE-DOWNLOAD COMPLETE =====")


# ===================================
# Dataset
# ===================================
train_transforms = transforms.Compose([])

class CustomOCRDataset(Dataset):
    """
    Custom dataset class tailored for loading image-text pairs (image and its transcription)
    for TrOCR fine-tuning. It handles image opening, tokenization, and prepares the data
    in the format required by the Hugging Face Seq2Seq Trainer.
    """

    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        # DataFrame containing the image filenames and ground-truth text.
        self.df = df.copy()
        self.processor = processor
        self.max_target_length = max_target_length
        # Replaces any missing text values (NaN) with an empty string to prevent tokenizer errors.
        self.df["text"] = self.df["text"].fillna("")

    def __len__(self):
        # Returns the total number of samples (images) in the dataset.
        return len(self.df)

    def __getitem__(self, idx):
        """
        Standard Python/PyTorch protocol for custom datasets
        This method is called repeatedly by the PyTorch DataLoader to retrieve, open, and preprocess a single image and its corresponding ground-truth label for the model
        During trainer.train() and evaluate()
        """
        # Retrieves the image filename and the ground-truth text for the current index.
        file_name = self.df.loc[idx, "image_filename"]
        text = self.df.loc[idx, "text"]

        image_path = os.path.join(self.root_dir, file_name)
        image = Image.open(image_path).convert("RGB")

        # --- Image Processing ---
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # --- Text Processing ---
        # Tokenizes the ground-truth text, padding it to max_target_length.
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
        ).input_ids

        # --- Loss Masking ---
        # Replaces PAD tokens with -100 so loss ignores them
        # Hugging Face Seq2Seq models use -100 as an index to tell the Cross-Entropy Loss
        # function to ignore the padded tokens, focusing loss only on actual text tokens.
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# ===================================
# Load CSV
# ===================================

def load_dataframes():
    """
    Load the annotation CSV files for the GNHK dataset.
    Each CSV contains two columns:
        - image_filename : name of the cropped handwritten image (e.g., "00123.png")
        - text           : the ground-truth transcription for that image

    The function reads:
        train_processed.csv → training annotations
        test_processed.csv  → test annotations, which we split into:
            - valid_df      : 50% for validation during training
            - final_test_df : remaining 50% kept for later testing
    """
    train_csv = os.path.join(DatasetConfig.DATA_ROOT, "train_processed.csv")
    test_csv = os.path.join(DatasetConfig.DATA_ROOT, "test_processed.csv")

    train_df = pd.read_csv(train_csv, header=None, skiprows=1,
                           names=["image_filename", "text"])
    test_df = pd.read_csv(test_csv, header=None, skiprows=1,
                          names=["image_filename", "text"])

    # Split test_df into 50% validation and 50% final test (deterministic)
    valid_df = test_df.sample(frac=0.5, random_state=42)
    final_test_df = test_df.drop(valid_df.index).reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    return train_df, valid_df, final_test_df


# ===================================
# CER metric
# ===================================

cer_metric = evaluate.load("cer")
def compute_cer_builder(processor):
    """
    Build a CER compute function for HuggingFace Trainer.
    CER (Character Error Rate) tells you how many characters your OCR model gets wrong
    compared to the ground-truth text.
    """
    def compute_cer(pred):
        pred_ids = pred.predictions
        labels_ids = pred.label_ids

        # Decode predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # Replace -100 with pad_token_id before decoding labels
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_cer


# ===================================
# Main
# ===================================

def main():
    print("Device:", device)
    # 1. Pre-download all heavy stuff
    download_dependencies()

    # 2. Load data
    train_df, valid_df, test_df = load_dataframes()

    processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)

    train_root = os.path.join(DatasetConfig.DATA_ROOT, "train_processed", "images")
    test_root = os.path.join(DatasetConfig.DATA_ROOT, "test_processed", "images")

    train_dataset = CustomOCRDataset(train_root, train_df, processor)
    valid_dataset = CustomOCRDataset(test_root, valid_df, processor)
    # ----- Use only ONE sample in train and ONE sample in validation -----
    # train_dataset = CustomOCRDataset(train_root, train_df.iloc[:5], processor)
    # valid_dataset = CustomOCRDataset(test_root, valid_df.iloc[:5], processor)

    # 3. Load model
    model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
    model.to(device)

    # ======== FREEZE EARLY ENCODER LAYERS ========
    print("total number of blocks we have", len(model.encoder.encoder.layer)) # total number of blocks we have
    # Freeze first 3 transformer blocks (for TrOCR small: blocks 0–2)
    for layer in model.encoder.encoder.layer[:2]:
        for param in layer.parameters():
            param.requires_grad = False

    # Generation / decoding config
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0 #  penalizes long generated sequences
    # Sets the maximum number of tokens/characters the decoder is allowed to generate for any single output sequence. This prevents excessively long, garbage predictions.
    model.config.max_length = 64
    # Specifies the beam width (k) for beam search decoding.
    model.config.num_beams = 4  # The model will keep track of the 4 most promising partial sequences at each step to find a higher-quality transcription (trade-off between speed and accuracy).

    compute_cer = compute_cer_builder(processor)

    # 4. Trainer settings
    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpointed_model",
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
        num_train_epochs=TrainingConfig.EPOCHS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        predict_with_generate=True,

        # ----------------------------------------------------
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,  # Log training metrics every 100 steps
        load_best_model_at_end=True,  # Load the model with the best validation CER after training finishes.
        metric_for_best_model="cer",  # Track the best model based on the 'cer' metric.
        # ----------------------------------------------------

        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_cer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
        callbacks=[ValidationLoggerCallback()],
    )

    # 5. Train
    print("Starting training...")
    # ==== resume if checkpoint exists ====
    checkpoint_dir = "checkpointed_model"
    # ensure at least one real checkpoint exists
    has_ckpt = any(
        name.startswith("checkpoint-")
        for name in os.listdir(checkpoint_dir)
    ) if os.path.isdir(checkpoint_dir) else False
    if has_ckpt:
        print("Resuming from last checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoint found — training from scratch...")
        trainer.train()
    # ===========================================

    # ----- SAVE TRAINED MODEL -----
    save_path = "saved_model"
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving fine-tuned model to: {save_path}")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    # --------------------------------

    # 6. Final test evaluation
    print("Evaluating on FINAL TEST set...")
    test_dataset = CustomOCRDataset(test_root, test_df, processor)
    # test_dataset = CustomOCRDataset(test_root, test_df.iloc[:1], processor)

    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print("Test CER:", test_results.get("test_cer"))

    # ----- DELETE CHECKPOINT FOLDER SO FUTURE RUNS DON'T RESUME -----
    import shutil
    shutil.rmtree("checkpointed_model", ignore_errors=True)
    print("Done.")

if __name__ == "__main__":
    main()
