# explainability.py
# This script uses SHAP to explain how our model makes its predictions.
# It loads all the saved models and data, and then generates SHAP values
# for a sample clinical note.

import pandas as pd
import numpy as np
import torch
import shap
import joblib
from transformers import AutoTokenizer, AutoModel
import os

print("Loading up all the models and data...")

# Check if we have an Apple Silicon GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon (MPS) device.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# Load the BERT model, the classifier, and the tokenizer
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval() # Set to eval mode

LR_MODEL_PATH = "models/logistic_regression_classifier.joblib"
lr_model = joblib.load(LR_MODEL_PATH)

# We need a small sample of data for the SHAP explainer to use as a
# background reference.
PROCESSED_DATA_PATH = "data/processed/processed_notes.csv"
background_data = pd.read_csv(PROCESSED_DATA_PATH).dropna().sample(100, random_state=42)

print("All models and data loaded.")


def predict_pipeline(text_array):
    """
    This function wraps our entire prediction pipeline, from raw text to
    the final probabilities. SHAP needs this to work its magic.
    """
    # SHAP sometimes passes empty strings, so we'll just replace them
    # with a space to avoid errors.
    text_array = [text if text != "" else " " for text in text_array]
    
    all_embeddings = []
    with torch.no_grad():
        for text in text_array:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embedding)
    
    embeddings = np.vstack(all_embeddings)
    
    # Get the final probabilities from our classifier
    return lr_model.predict_proba(embeddings)


print("\nCreating the SHAP Explainer...")

# This creates the SHAP explainer. It takes our prediction pipeline and
# the tokenizer as input.
explainer = shap.Explainer(predict_pipeline, tokenizer)

print("SHAP Explainer is ready.")

# Let's explain a single prediction
sample_text = background_data.iloc[5]['cleaned_text']
print(f"\nExplaining this text:\n---")
print(sample_text[:300] + "...")
print("---")

# This is where the actual SHAP calculation happens. It can be a bit slow.
shap_values = explainer([sample_text])

print("\nSHAP values have been generated.")
print("You can now use these values to create a visualization.")
print(f"SHAP values shape: {shap_values.shape}")
print(f"Base value: {shap_values.base_values[0]}")
