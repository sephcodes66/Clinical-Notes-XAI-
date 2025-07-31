# visualization.py
# This script creates a global summary plot of the SHAP values. It shows
# which words have the biggest impact on the model's predictions for a
# given class.

import pandas as pd
import numpy as np
import torch
import shap
import joblib
from transformers import AutoTokenizer, AutoModel
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Path for the cached SHAP values
SHAP_VALUES_PATH = "outputs/shap_values.joblib"
os.makedirs("outputs", exist_ok=True)

print("Loading up the models and data...")

# I'm forcing this to use the CPU because it's not as memory-hungry
# as the other scripts.
device = torch.device("cpu")

# Load up all the things we need
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval()
lr_model = joblib.load("models/logistic_regression_classifier.joblib")

class_names = np.load('data/features/label_encoder_classes.npy', allow_pickle=True)
background_data = pd.read_csv("data/processed/processed_notes.csv").dropna().sample(100, random_state=42)
background_text = background_data['cleaned_text'].tolist()

print("All loaded up.")

def predict_pipeline(text_array):
    """
    A wrapper for our prediction pipeline. SHAP needs this.
    """
    # SHAP can pass in empty strings, so we'll catch that here.
    text_array = [text if text != "" else " " for text in text_array]
    inputs = tokenizer(text_array, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return lr_model.predict_proba(cls_embeddings)

# If we've already calculated the SHAP values, let's just load them.
if os.path.exists(SHAP_VALUES_PATH):
    print(f"\nLoading SHAP values from '{SHAP_VALUES_PATH}'...")
    shap_values = joblib.load(SHAP_VALUES_PATH)
    print("Loaded.")
else:
    # If not, we'll have to calculate them. This takes a while.
    print("\nCreating the SHAP Explainer...")
    explainer = shap.Explainer(predict_pipeline, tokenizer)

    print("Calculating SHAP values... (this might take a while)")
    shap_values = explainer(background_text)
    
    print(f"Done. Saving SHAP values to '{SHAP_VALUES_PATH}'...")
    joblib.dump(shap_values, SHAP_VALUES_PATH)
    print("Saved.")

# Now let's make a plot
print("\nCreating the global summary plot...")

# TODO: Let the user choose which class to explain
target_class_index = 0
target_class_name = class_names[target_class_index]

# This is a bit of a hack to get the global feature importances.
# I'm just averaging the absolute SHAP values for each token.
token_impacts = defaultdict(lambda: {'sum': 0.0, 'count': 0})

for i in range(len(shap_values)):
    sample_shap_values = shap_values[i, :, target_class_index].values
    sample_tokens = shap_values[i, :, target_class_index].data

    for token, shap_val in zip(sample_tokens, sample_shap_values):
        token_impacts[token]['sum'] += abs(shap_val)
        token_impacts[token]['count'] += 1

mean_abs_shap = {token: data['sum'] / data['count'] for token, data in token_impacts.items()}

# Get the top 20 most important tokens
top_tokens = sorted(mean_abs_shap, key=mean_abs_shap.get, reverse=True)[:20]
top_scores = [mean_abs_shap[token] for token in top_tokens]

# Make the plot
fig, ax = plt.subplots()
y_pos = np.arange(len(top_tokens))

ax.barh(y_pos, top_scores, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(reversed(top_tokens)) # Most important at the top
ax.invert_yaxis()
ax.set_xlabel("Average SHAP Value (how much it impacts the prediction)")
ax.set_title(f"Top 20 Words for Class: {target_class_name}")

plt.tight_layout()
plt.savefig("outputs/global_summary_plot_example.png")
plt.close()

print(f"\nPlot saved to 'outputs/global_summary_plot_example.png'")
