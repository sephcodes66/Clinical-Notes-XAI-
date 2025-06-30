# FileName: explainability.py
# Description: This script uses the SHAP (SHapley Additive exPlanations) framework to
#              explain predictions from the end-to-end model pipeline. It loads all
#              necessary artifacts, generates SHAP values for a sample clinical note,
#              and prepares them for visualization.

import pandas as pd
import numpy as np
import torch
import shap
import joblib
from transformers import AutoTokenizer, AutoModel
import os

print("Loading all necessary models and data for explainability...")

# Set device for model inference (e.g., Apple Silicon MPS, CUDA, or CPU).
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon (MPS) device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# Load the core components: the BERT model, the trained classifier, and the tokenizer.
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval() # Set model to evaluation mode.

LR_MODEL_PATH = "models/logistic_regression_classifier.joblib"
lr_model = joblib.load(LR_MODEL_PATH)

# Load a small, representative sample of the data to serve as the background dataset for SHAP.
# A small sample is used to make the explainer's computation tractable.
PROCESSED_DATA_PATH = "data/processed/processed_notes.csv"
background_data = pd.read_csv(PROCESSED_DATA_PATH).dropna().sample(100, random_state=42)

print("All artifacts loaded successfully.")


def predict_proba_pipeline(text_array):
    """
    Encapsulates the entire prediction pipeline from raw text to class probabilities.
    This is required by SHAP to understand how inputs map to the final output.
    
    Args:
        text_array (np.array): An array of N text strings provided by SHAP.
        
    Returns:
        np.array: An (N, k) array of prediction probabilities for k classes.
    """
    # SHAP's masking process can generate empty strings; replace them to prevent errors.
    text_array = [text if text != "" else " " for text in text_array]
    
    all_embeddings = []
    # Disable gradient calculations for efficient inference.
    with torch.no_grad():
        # Process each text sample provided by the SHAP explainer.
        for text in text_array:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embedding)
    
    embeddings_stack = np.vstack(all_embeddings)
    
    # Pass the generated embeddings to the trained classifier to get final probabilities.
    return lr_model.predict_proba(embeddings_stack)


print("\nInstantiating SHAP Explainer...")

# Initialize a SHAP explainer for text data. It requires the end-to-end prediction
# function and the tokenizer, which it uses to mask out parts of the text.
explainer = shap.Explainer(predict_proba_pipeline, tokenizer)

print("SHAP Explainer created.")

# Select a single instance from the data to generate a local explanation.
sample_text_to_explain = background_data.iloc[5]['cleaned_text']
print(f"\nGenerating explanation for sample text:\n---")
print(sample_text_to_explain[:300] + "...")
print("---")

# This is the core SHAP calculation, which computes the contribution of each token.
# It can be computationally intensive, especially on the first run.
shap_values = explainer([sample_text_to_explain])

print("\nSHAP values generated successfully.")
print("The 'shap_values' object is now ready for visualization.")
print(f"Shape of SHAP values: {shap_values.shape}")
print(f"Base value (average model output probability): {shap_values.base_values[0]}")

# Example of how to inspect the raw SHAP values for a specific class (e.g., class 0).
# print("\nSHAP values for class 0:")
# print(shap_values.values[0, :, 0])