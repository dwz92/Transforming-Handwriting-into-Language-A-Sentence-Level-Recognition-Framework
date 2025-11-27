import json
from sentence_transformers import SentenceTransformer, util

# ================================================================
# 1. Load organized GT, Pred, Baseline notes
# ================================================================

GT_FILE   = "organized_notes_gt.json"
PRED_FILE = "organized_notes_pred.json"
BASE_FILE = "organized_notes_baseline.json"

with open(GT_FILE, "r") as f:
    notes_gt = json.load(f)

with open(PRED_FILE, "r") as f:
    notes_pred = json.load(f)

with open(BASE_FILE, "r") as f:
    notes_base = json.load(f)

# Only evaluate notes that exist in all three files
note_ids = sorted(set(notes_gt.keys()) & set(notes_pred.keys()) & set(notes_base.keys()))
print(f"Loaded {len(note_ids)} notes to compare.\n")

# ================================================================
# 2. Load semantic embedding model
# ================================================================

print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
model = SentenceTransformer("all-mpnet-base-v2")

# ================================================================
# 3. Compute similarities
# ================================================================

sim_pred_list = []   # GT vs Finetuned
sim_base_list = []   # GT vs Baseline
improvement_list = []

print("\n===== Per-note Similarities =====\n")

for note_id in note_ids:
    gt_text   = notes_gt[note_id].strip()
    pred_text = notes_pred[note_id].strip()
    base_text = notes_base[note_id].strip()

    # Encode
    emb_gt   = model.encode(gt_text, convert_to_tensor=True)
    emb_pred = model.encode(pred_text, convert_to_tensor=True)
    emb_base = model.encode(base_text, convert_to_tensor=True)

    # Compute cosine similarities
    sim_pred = util.cos_sim(emb_gt, emb_pred).item()
    sim_base = util.cos_sim(emb_gt, emb_base).item()
    improvement = sim_pred - sim_base

    sim_pred_list.append(sim_pred)
    sim_base_list.append(sim_base)
    improvement_list.append(improvement)

    print(f"{note_id}:")
    print(f"  GT vs Predicted  = {sim_pred:.4f}")
    print(f"  GT vs Baseline   = {sim_base:.4f}")
    print(f"  Improvement      = {improvement:.4f}")
    print()

# ================================================================
# 4. Summary Statistics
# ================================================================

mean_pred = sum(sim_pred_list) / len(sim_pred_list)
mean_base = sum(sim_base_list) / len(sim_base_list)
mean_improve = sum(improvement_list) / len(improvement_list)

print("===================================")
print("       Overall Similarities        ")
print("===================================")
print(f"Mean GT → Finetuned similarity: {mean_pred:.4f}")
print(f"Mean GT → Baseline similarity:  {mean_base:.4f}")
print("-----------------------------------")
print(f"Overall improvement:             {mean_improve:.4f}")
print("===================================")
