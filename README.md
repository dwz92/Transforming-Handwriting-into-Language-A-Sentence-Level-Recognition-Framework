# 413-Final

This project implements handwriting recognition using a fine-tuned TrOCR model.  
It includes:

- Fine-tuning the model  
- Evaluating CER for both the baseline and fine-tuned models  
- Generating raw and organized predicted notes  
- Comparing similarity between the ground truth notes and model-generated notes  

---

## Dataset Setup

1. Download the dataset:  
   https://drive.google.com/file/d/1v2T0rhVals2MHO8DEcDQH_2v7llGGR3v/view?usp=drive_link

2. Unzip the downloaded file.

3. Move the folder named **`input`** into the **`413-Final`** directory so the structure looks like:
413-Final/ ├── input/ │ ├── train/ │ ├── test/ │ └── metadata.csv ├── train_model.py ├── run_saved_model.py ├── run_baseline_model.py ├── organize_notes.py └── eval_similarity.py
   
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
