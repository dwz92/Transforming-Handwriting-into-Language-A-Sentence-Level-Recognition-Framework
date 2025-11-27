import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# ============================================================
# 1. Load local LLM: Phi-3.5 Mini
# ============================================================
print("Loading free local LLM (Phi-3.5-mini-instruct)...")
model_name = "microsoft/Phi-3.5-mini-instruct"  # fully open, no auth needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ============================================================
# 2. Utility: Join + clean raw OCR word list
# ============================================================

def join_and_clean(words):
    text = " ".join(words)
    text = text.replace("SPECIAL_CHARACTER", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# 3. Utility: Rewrite with LLM
# ============================================================

def rewrite_paragraph(raw_text):
    prompt = (
        "You are rewriting handwritten notes.\n"
        "Rewrite the following text into ONE clean, simple, natural English paragraph.\n"
        "IMPORTANT RULES:\n"
        "- Keep ALL meaning from the original.\n"
        "- Do NOT add any new information.\n"
        "- Do NOT invent examples, steps, or explanations.\n"
        "- Do NOT generalize the content.\n"
        "- Only correct grammar, spelling, and flow.\n"
        "- The output must ONLY contain the rewritten paragraph, nothing else.\n\n"
        f"Original text:\n{raw_text}\n\n"
        "Rewritten paragraph:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.0,
        do_sample=False,
    )

    full = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Rewritten paragraph:" in full:
        cleaned = full.split("Rewritten paragraph:", 1)[1].strip()
    else:
        cleaned = full.strip()

    # Strip trailing junk
    for stopper in ["##", "###", "\nRewritten", "\nOriginal", "\nIMPORTANT"]:
        cleaned = cleaned.split(stopper)[0].strip()

    return cleaned

# ============================================================
# 4. Function to process a single raw-notes file
# ============================================================

def process_notes_file(input_file, output_file):
    print(f"\n=== Processing {input_file} → {output_file} ===\n")

    # Load raw notes
    with open(input_file, "r") as f:
        raw_notes = json.load(f)

    # Crash-safe resume support
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            organized = json.load(f)
    else:
        organized = {}

    temp_output = output_file + ".tmp"

    for note_id, words in raw_notes.items():
        print(f"\nProcessing {note_id}...")
        if note_id in organized:
            print(f"Skipping {note_id} (already processed).")
            continue

        raw_text = join_and_clean(words)
        paragraph = rewrite_paragraph(raw_text)

        organized[note_id] = paragraph
        print("Cleaned paragraph:\n", paragraph)
        with open(temp_output, "w") as f:
            json.dump(organized, f, indent=2)

        os.replace(temp_output, output_file)
        print(f"[SAVED] {note_id} into {output_file}")

    print(f"\n=== Finished {output_file} ===\n")
    return organized


# ============================================================
# 5. Organize notes for GT, Pred, and Baseline from the raw notes
# ============================================================

GT_INPUT   = "raw_notes_gt.json"
PRED_INPUT = "raw_notes_pred.json"
BASE_INPUT = "raw_notes_baseline.json"

GT_OUTPUT   = "organized_notes_gt.json"
PRED_OUTPUT = "organized_notes_pred.json"
BASE_OUTPUT = "organized_notes_baseline.json"

organized_gt       = process_notes_file(GT_INPUT, GT_OUTPUT)
organized_pred     = process_notes_file(PRED_INPUT, PRED_OUTPUT)
organized_baseline = process_notes_file(BASE_INPUT, BASE_OUTPUT)

print("\nAll GT, Pred, and Baseline notes processed successfully")
