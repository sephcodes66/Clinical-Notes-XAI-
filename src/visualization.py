# FileName: visualization.py
# Description: This script visualizes the model's behavior using SHAP values. It either
#              loads pre-computed SHAP values or generates them for a background dataset.
#              It then aggregates these values to create a global summary bar plot,
#              highlighting the words with the most impact on predictions for a given class.

import pandas as pd
import numpy as np
import torch
import shap
import joblib
from transformers import AutoTokenizer, AutoModel
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the output path for caching SHAP values to avoid re-computation.
SHAP_VALUES_PATH = "outputs/shap_values.joblib"
os.makedirs("outputs", exist_ok=True)

print("Loading all necessary models and data for visualization...")

# Forcing CPU device for this script to ensure consistency, as it's less memory-intensive.
device = torch.device("cpu")

# Load all required artifacts: the BERT model, the trained classifier, and the tokenizer.
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval()
lr_model = joblib.load("models/logistic_regression_classifier.joblib")

# Load class names for labeling the plot and the background data for the explainer.
class_names = np.load('data/features/label_encoder_classes.npy', allow_pickle=True)
background_data = pd.read_csv("data/processed/processed_notes.csv").dropna().sample(100, random_state=42)
background_text = background_data['cleaned_text'].tolist()

print("All artifacts loaded.")

def predict_proba_pipeline(text_array):
    """
    Encapsulates the entire prediction pipeline from raw text to class probabilities.
    Required by SHAP to map raw text inputs to the final model output.
    """
    # SHAP's masking process can generate empty strings; replace them to prevent errors.
    text_array = [text if text != "" else " " for text in text_array]
    inputs = tokenizer(text_array, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    # Return final probabilities from the downstream classifier.
    return lr_model.predict_proba(cls_embeddings)

# Check for cached SHAP values to avoid lengthy re-computation.
if os.path.exists(SHAP_VALUES_PATH):
    print(f"\nFound existing SHAP values. Loading from '{SHAP_VALUES_PATH}'...")
    shap_values = joblib.load(SHAP_VALUES_PATH)
    print("SHAP values loaded successfully.")
else:
    # If no cached values exist, instantiate the explainer and compute them.
    print("\nInstantiating SHAP Explainer for global summary...")
    explainer = shap.Explainer(predict_proba_pipeline, tokenizer)

    print("Calculating SHAP values for 100 background samples... (This will take a long time)")
    shap_values = explainer(background_text)
    
    print(f"Calculation complete. Saving SHAP values to '{SHAP_VALUES_PATH}'...")
    joblib.dump(shap_values, SHAP_VALUES_PATH)
    print("SHAP values saved successfully.")

# Aggregate SHAP values to create a global feature importance plot.
print("\nGenerating and saving the global summary plot...")

# Define which class prediction to explain.
target_class_index = 0
target_class_name = class_names[target_class_index]

# Manually aggregate the impact of each token across all background samples.
token_impacts = defaultdict(lambda: {'sum': 0.0, 'count': 0})

# Iterate over each sample explanation.
for i in range(len(shap_values)):
    # Isolate SHAP values and corresponding tokens for the target class.
    sample_shap_values = shap_values[i, :, target_class_index].values
    sample_tokens = shap_values[i, :, target_class_index].data

    for token, shap_val in zip(sample_tokens, sample_shap_values):
        # Use the absolute SHAP value as a measure of a token's importance.
        token_impacts[token]['sum'] += abs(shap_val)
        token_impacts[token]['count'] += 1

# Calculate the mean absolute SHAP value for each token across the dataset.
mean_abs_shap = {token: data['sum'] / data['count'] for token, data in token_impacts.items()}

# Identify the top 20 tokens with the highest average impact.
top_tokens = sorted(mean_abs_shap, key=mean_abs_shap.get, reverse=True)[:20]
top_scores = [mean_abs_shap[token] for token in top_tokens]

# Generate the bar plot.
fig, ax = plt.subplots()
y_pos = np.arange(len(top_tokens))

ax.barh(y_pos, top_scores, align='center')
ax.set_yticks(y_pos)
# Reverse the list of labels to display the most important token at the top.
ax.set_yticklabels(reversed(top_tokens))
ax.invert_yaxis()
ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Output)")
ax.set_title(f"Top 20 Most Influential Words for Class: {target_class_name}")

plt.tight_layout()
plt.savefig("outputs/global_summary_plot_example.png")
plt.close()

print(f"\nGlobal summary plot saved successfully to 'outputs/global_summary_plot_example.png'")