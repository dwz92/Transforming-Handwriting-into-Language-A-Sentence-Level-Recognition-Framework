import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# Import helpers from your existing train_model.py
import train_model as tm


# ----------------------------------------------------
# Hyperparameter configurations to try
# ----------------------------------------------------
EXPERIMENTS = [
    {"name": "E1_ep1_lr5e-5_fr2", "epochs": 1, "lr": 5e-5, "freeze": 2},
    {"name": "E2_ep3_lr5e-5_fr2", "epochs": 3, "lr": 5e-5, "freeze": 2},

    # REQUIRED experiment
    {"name": "E3_ep3_lr3e-5_fr4", "epochs": 3, "lr": 3e-5, "freeze": 4},

    {"name": "E4_ep5_lr3e-5_fr4", "epochs": 5, "lr": 3e-5, "freeze": 4},
    {"name": "E5_ep3_lr1e-5_fr2", "epochs": 3, "lr": 1e-5, "freeze": 2},

    # Ablation: same as E3 but with no frozen encoder blocks
    {"name": "E6_ep3_lr3e-5_fr0", "epochs": 3, "lr": 3e-5, "freeze": 0},
]


def build_datasets(processor):
    """
    Use the same CSVs and structure as train_model.py to build train/val/test datasets.
    """
    train_df, valid_df, test_df = tm.load_dataframes()

    train_root = os.path.join(tm.DatasetConfig.DATA_ROOT, "train_processed", "images")
    test_root = os.path.join(tm.DatasetConfig.DATA_ROOT, "test_processed", "images")

    train_dataset = tm.CustomOCRDataset(train_root, train_df, processor)
    valid_dataset = tm.CustomOCRDataset(test_root, valid_df, processor)
    test_dataset = tm.CustomOCRDataset(test_root, test_df, processor)

    return train_dataset, valid_dataset, test_dataset


def build_model_and_trainer(
    exp_name,
    epochs,
    lr,
    freeze_blocks,
    processor,
    train_dataset,
    valid_dataset,
    device,
):
    """
    Build a fresh TrOCR model + Seq2SeqTrainer for a given hyperparameter config.
    """
    # 1. Load pre-trained TrOCR small (same as in train_model)
    model = VisionEncoderDecoderModel.from_pretrained(tm.ModelConfig.MODEL_NAME)
    model.to(device)

    # 2. Freeze first `freeze_blocks` encoder layers
    total_blocks = len(model.encoder.encoder.layer)
    print(f"[{exp_name}] total encoder blocks: {total_blocks}")
    if freeze_blocks > 0:
        print(f"[{exp_name}] Freezing first {freeze_blocks} encoder blocks...")
        for layer in model.encoder.encoder.layer[:freeze_blocks]:
            for param in layer.parameters():
                param.requires_grad = False
    else:
        print(f"[{exp_name}] No encoder blocks frozen.")

    # 3. Set generation config (copied from train_model.py)
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.max_length = 64
    model.config.num_beams = 4

    # 4. Metrics (reuse CER builder from train_model.py)
    compute_cer = tm.compute_cer_builder(processor)

    # 5. Training arguments
    # Each experiment gets its own output_dir, but we don't keep checkpoints.
    output_dir = f"checkpoint_{exp_name}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=tm.TrainingConfig.BATCH_SIZE,
        per_device_eval_batch_size=tm.TrainingConfig.BATCH_SIZE,
        num_train_epochs=epochs,
        learning_rate=lr,
        predict_with_generate=True,

        eval_strategy="epoch",
        save_strategy="no",          # don't save intermediate checkpoints
        logging_steps=100,
        load_best_model_at_end=False,
        metric_for_best_model="cer",

        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        report_to=[],                # disable wandb/other integrations
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_cer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
        callbacks=[tm.ValidationLoggerCallback()],
    )

    return model, trainer


def run_single_experiment(
    exp,
    processor,
    train_dataset,
    valid_dataset,
    test_dataset,
    device,
):
    """
    Run training + test evaluation for a single hyperparameter configuration.
    Returns the test CER.
    """
    name = exp["name"]
    epochs = exp["epochs"]
    lr = exp["lr"]
    freeze_blocks = exp["freeze"]

    print("\n===============================================")
    print(f"Running experiment: {name}")
    print(f"  epochs={epochs}, lr={lr}, freeze_blocks={freeze_blocks}")
    print("===============================================")

    model, trainer = build_model_and_trainer(
        exp_name=name,
        epochs=epochs,
        lr=lr,
        freeze_blocks=freeze_blocks,
        processor=processor,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        device=device,
    )

    # Train
    trainer.train()

    # Evaluate on FINAL TEST set
    print(f"[{name}] Evaluating on FINAL TEST set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    test_cer = test_results.get("test_cer")

    print(f"[{name}] Test CER: {test_cer}")

    # Free GPU memory between runs
    del model
    torch.cuda.empty_cache()

    return test_cer


def main():
    device = tm.device
    print("Using device:", device)

    # Ensure experiment_result folder exists
    result_dir = Path("experiment_result")
    result_dir.mkdir(exist_ok=True)

    # Shared processor and datasets
    processor = TrOCRProcessor.from_pretrained(tm.ModelConfig.MODEL_NAME)
    train_dataset, valid_dataset, test_dataset = build_datasets(processor)

    # Summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = result_dir / f"hyperparam_results_{timestamp}.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Hyperparameter Tuning Results ({timestamp})\n")
        f.write("===========================================\n\n")
        f.write("Format: name | epochs | lr | freeze | test_cer\n\n")

        for exp in EXPERIMENTS:
            test_cer = run_single_experiment(
                exp,
                processor,
                train_dataset,
                valid_dataset,
                test_dataset,
                device,
            )
            f.write(
                f"{exp['name']} | "
                f"epochs={exp['epochs']} | "
                f"lr={exp['lr']} | "
                f"freeze={exp['freeze']} | "
                f"test_cer={test_cer}\n"
            )

    print(f"\nAll experiments finished. Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
