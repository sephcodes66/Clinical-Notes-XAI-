# FileName: app.py
# Description: This script builds and runs an interactive Dash web application. The dashboard
#              allows users to input clinical text, receive a prediction from the trained
#              classifier, and visualize the prediction's explanation through highlighted
#              text and a SHAP waterfall plot.

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import torch
import shap
import joblib
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import re

# Load all models and data once at startup to ensure the dashboard is responsive.
print("Loading all models and data for the dashboard...")
device = torch.device("cpu")
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval() # Set model to evaluation mode.
lr_model = joblib.load("models/logistic_regression_classifier.joblib")
class_names = np.load('data/features/label_encoder_classes.npy', allow_pickle=True)
# Load a sample text to pre-populate the text area.
sample_text = pd.read_csv("data/processed/processed_notes.csv").dropna().sample(1, random_state=42).iloc[0]['cleaned_text']
print("Dashboard models loaded successfully.")

def predict_proba_pipeline(text_array):
    """Encapsulates the prediction pipeline from raw text to class probabilities."""
    if isinstance(text_array, np.ndarray):
        text_array = text_array.tolist()
    inputs = tokenizer(text_array, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return lr_model.predict_proba(cls_embeddings)

# Configure the SHAP explainer for text, splitting by whitespace for tokenization.
text_masker = shap.maskers.Text(r"\s+")
explainer = shap.Explainer(predict_proba_pipeline, text_masker)

def shap_to_html(shap_explanation, class_index):
    """
    Converts a SHAP explanation object into an HTML string with highlighted words.
    """
    words = shap_explanation.data
    shap_vals = shap_explanation[:, class_index].values
    
    # Normalize SHAP values to be between -1 and 1 for consistent color scaling.
    max_abs_val = np.abs(shap_vals).max()
    if max_abs_val == 0: max_abs_val = 1 # Avoid division by zero on neutral inputs.
    
    html_elements = []
    for word, val in zip(words, shap_vals):
        normalized_val = val / max_abs_val
        # Assign red for positive contributions and blue for negative contributions.
        if normalized_val > 0:
            color = f"rgba(255, 0, 0, {abs(normalized_val):.3f})" # Red
        else:
            color = f"rgba(0, 0, 255, {abs(normalized_val):.3f})" # Blue
        
        # Create an HTML span for each word with a tooltip showing its exact SHAP value.
        tooltip_text = f"Contribution: {val:.4f}"
        span = html.Span(
            children=word + " ",
            style={'background-color': color, 'padding': '2px', 'margin': '1px', 'border-radius': '3px'},
            title=tooltip_text
        )
        html_elements.append(span)
        
    return html.Div(html_elements)

# Initialize the Dash application with a Bootstrap theme.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout and structure of the web application.
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Explainable AI for Clinical Text Classification", className="text-center my-4"), width=12)),
    dbc.Row([
        dbc.Col([
            html.H5("Enter a Clinical Note:"),
            dcc.Textarea(id='text-input', value=sample_text, style={'width': '100%', 'height': 200}),
            html.Button('Analyze', id='submit-button', n_clicks=0, className="mt-2")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(id="loading-output", children=[
            html.H4("Model Prediction", className="mt-4"),
            html.Div(id='prediction-output', className="lead"),
            
            # Container for the highlighted text explanation.
            html.H4("Highlighted Text Explanation", className="mt-4"),
            html.P("Hover over words to see their contribution value. Red words increase the prediction probability, blue words decrease it."),
            html.Div(id='highlighted-text-output', style={'border': '1px solid #ddd', 'padding': '10px', 'line-height': '2.0'}),
            
            html.H4("SHAP Waterfall Plot", className="mt-4"),
            dcc.Graph(id='shap-plot')
        ]), width=12)
    ]),
], fluid=True)

# Define the server-side callback to link UI components to the prediction logic.
@app.callback(
    Output('prediction-output', 'children'),
    Output('shap-plot', 'figure'),
    Output('highlighted-text-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('text-input', 'value')
)
def update_output(n_clicks, text_input):
    if n_clicks == 0 or not text_input:
        return "Enter text and click 'Analyze'.", go.Figure(), ""

    # Run the full prediction and explanation pipeline on the user's input.
    prediction_probas = predict_proba_pipeline(np.array([text_input]))[0]
    predicted_class_index = np.argmax(prediction_probas)
    predicted_class_name = class_names[predicted_class_index]
    predicted_probability = prediction_probas[predicted_class_index]
    prediction_text = f"Predicted Specialty: {predicted_class_name} (Probability: {predicted_probability:.2%})"

    shap_values = explainer([text_input])
    explanation_for_one_sample = shap_values[0]

    try:
        # Generate the waterfall plot for the top contributing features.
        shap_vals_for_class = explanation_for_one_sample[:, predicted_class_index].values
        words = explanation_for_one_sample.data
        non_zero_indices = np.where(shap_vals_for_class != 0)[0]
        num_features = min(20, len(non_zero_indices))
        sorted_indices = np.argsort(np.abs(shap_vals_for_class[non_zero_indices]))[-num_features:]
        top_words = np.array(words)[non_zero_indices][sorted_indices]
        top_shap_values = shap_vals_for_class[non_zero_indices][sorted_indices]

        fig = go.Figure(go.Waterfall(
            name="SHAP Explanation", orientation="h", y=top_words, x=top_shap_values,
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            increasing={"marker":{"color":"#d62728"}},
            decreasing={"marker":{"color":"#1f77b4"}}
        ))
        fig.update_layout(
            title=f"How Features Contributed to the '{predicted_class_name}' Prediction",
            yaxis_title="Features (Words)", margin=dict(l=150, r=20, t=60, b=20)
        )
        
        # Generate the highlighted text HTML from the SHAP explanation.
        highlighted_text_html = shap_to_html(explanation_for_one_sample, predicted_class_index)

    except Exception as e:
        print(f"Error creating plot: {e}")
        fig = go.Figure().update_layout(title="Could not generate plot.")
        highlighted_text_html = "Error generating highlighted text."
        
    return prediction_text, fig, highlighted_text_html

# Entry point for running the web application server.
if __name__ == '__main__':
    app.run(debug=True)