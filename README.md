# 413-Final

This project implements handwriting recognition using a fine-tuned TrOCR model.  
It includes:

- Fine-tuning the model  
- Evaluating CER for both the baseline and fine-tuned models  
- Generating raw and organized predicted notes  
- Comparing similarity between the ground truth notes and model-generated notes  

---

## Work Division
| Week | Member        | Contribution Description                                                                                   | Hours |
|------|----------------|-------------------------------------------------------------------------------------------------------------|-------|
| **Week 1 – Preprocessing** | Shunqi Wang   | Explored dataset and data structure; preprocessed GNHK dataset                                     | ~8    |
|      | Ziheng Zhou   | Data augmentation                                                                                           | ~2    |
|      | Qi Er Teng    | Designed model architecture; explored related models                                                        | ~8    |
|      | Ruilin Tian   | Examined possible ethical issues                                                                            | ~2    |
| **Week 2 – Fine-tuning** | Shunqi Wang   |                                                                                                     |       |
|      | Ziheng Zhou   | Model training, checkpointing, logging; HPO regularization; GPU management; integration management          | ~8    |
|      | Qi Er Teng    |                                                                                                             |       |
|      | Ruilin Tian   |                                                                                                             |       |
| **Week 3 – Evaluation** | Shunqi Wang   |                                                                                                     |       |
|      | Ziheng Zhou   |                                                                                                             |       |
|      | Qi Er Teng    |                                                                                                             |       |
|      | Ruilin Tian   | Conducted evaluations of model accuracy and robustness (~4hrs); discussed limitations (~2hrs)              | ~6    |


## Dataset Setup

1. Download the dataset:  
   https://drive.google.com/file/d/1v2T0rhVals2MHO8DEcDQH_2v7llGGR3v/view?usp=drive_link

2. Unzip the downloaded file.

3. Move the folder named **`input`** into the **`413-Final`** directory so the structure looks like:
```
413-Final/
├── input/
│ ├── train/
│ ├── test/
│ └── metadata.csv
├── train_model.py
├── run_saved_model.py
├── run_baseline_model.py
├── organize_notes.py
└── eval_similarity.py
```
   
## Environment Setup

1. Create the Conda environment:  conda create -n 413Project python=3.10
2. Activate the environment:  conda activate 413Project
3. Install all dependencies: pip install -r requirements.txt
   
## Running the Experiment

1. Run `train_model.py` to train the fine-tuned model.  
   The trained model will be saved in a folder named **`saved_model`**.

2. Run `run_saved_model.py` to generate the **raw_notes** for:  
   - the ground truth  
   - the fine-tuned model predictions  

3. Run `run_baseline_model.py` to generate the **raw_notes** using the baseline pretrained model.

4. Run `organize_notes.py` to convert all generated raw notes into organized paragraphs.

5. Finally, run `eval_similarity.py` to evaluate the similarity between:  
   - the ground truth notes  
   - the notes generated from the fine-tuned model  
   - the notes generated from the baseline model
  
## Script Descriptions

### `train_model.py`
This script fine-tunes Microsoft’s **TrOCR Small Handwritten** model on the GNHK dataset.  
It loads the preprocessed training and validation data, trains the encoder–decoder architecture, and evaluates the **Character Error Rate (CER)** at the end of each epoch.  
After training, the script computes the final test CER and saves the fully fine-tuned model and processor into a directory called **`saved_model/`**.  
It also logs validation performance across epochs, ensuring that the best validation CER is preserved.

---

### `run_saved_model.py`
This script loads the fine-tuned model from **`saved_model/`** and uses it to generate **raw notes** for every handwritten note image in the GNHK dataset.  
For each note ID, it collects predicted words in the correct order and also stores the associated ground-truth words.  
The results are saved as:

- `raw_notes_gt.json` — ground-truth word lists  
- `raw_notes_pred.json` — fine-tuned model predictions  

---

### `run_baseline_model.py`
This script evaluates the original (non-fine-tuned) **pretrained TrOCR model** on the GNHK dataset.  
It reports the baseline validation and test **CER**, showing how the pretrained model performs without any GNHK-specific training.  
It also generates raw predicted words for every note image and saves them in:

- `raw_notes_baseline.json` — baseline model predictions  

These predictions provide a reference for comparison with the fine-tuned model.

---

### `organize_notes.py`
This script uses the **Phi-3.5-mini-instruct** LLM to convert raw OCR word lists into clean, readable paragraphs.  
For each note, the script sends the list of words to the LLM, which rewrites them into a clear, grammatically correct paragraph **without adding new content**.  
It produces:

- `organized_notes_gt.json` — paragraphs rewritten from ground-truth words  
- `organized_notes_pred.json` — paragraphs rewritten from fine-tuned model predictions  
- `organized_notes_baseline.json` — paragraphs rewritten from baseline model predictions  

These organized paragraphs are used for semantic similarity evaluation.

---

### `eval_similarity.py`
This script computes **semantic similarity** between the organized ground-truth paragraphs and the paragraphs generated by both the fine-tuned and baseline models.  
Using the SentenceTransformer model **all-mpnet-base-v2**, it embeds each paragraph and computes cosine similarity for each note.  
It reports both **per-note similarity** and **overall averages**, enabling a direct comparison of semantic accuracy between models and quantifying how much improvement fine-tuning provides.


### Character Error Rate (CER)

Character Error Rate (CER) measures how accurately the model transcribes each cropped word image from the GNHK dataset.  
For every cropped image, the model predicts a word, and CER compares this prediction to its ground-truth label by counting **character substitutions, insertions, and deletions**, normalized by the length of the true word.

A perfect match gives **CER = 0**, while more mistakes increase the score.

During fine-tuning, validation, and testing, CER reflects how close each predicted word is to its corresponding target word.  
It is also used to compare the performance of the pretrained baseline TrOCR model with the fine-tuned version.


