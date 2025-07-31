# app.py
# This is the main Dash app. It's a simple web interface for playing
# around with the model and seeing how the explanations work.

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

# Load up all the models and data when the app starts.
print("Loading all the things for the dashboard...")
device = torch.device("cpu")
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval()
lr_model = joblib.load("models/logistic_regression_classifier.joblib")
class_names = np.load('data/features/label_encoder_classes.npy', allow_pickle=True)
# I'm just grabbing a random sample to pre-populate the text box.
sample_text = pd.read_csv("data/processed/processed_notes.csv").dropna().sample(1, random_state=42).iloc[0]['cleaned_text']
print("Dashboard is ready to go.")

def predict_pipeline(text_array):
    """A wrapper for the whole prediction pipeline."""
    if isinstance(text_array, np.ndarray):
        text_array = text_array.tolist()
    inputs = tokenizer(text_array, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return lr_model.predict_proba(cls_embeddings)

# I'm using a simple whitespace masker for the text.
text_masker = shap.maskers.Text(r"\s+")
explainer = shap.Explainer(predict_pipeline, text_masker)

def shap_to_html(shap_explanation, class_index):
    """
    Turns a SHAP explanation into a bunch of HTML spans with pretty colors.
    """
    words = shap_explanation.data
    shap_vals = shap_explanation[:, class_index].values
    
    # Normalize the SHAP values so the colors look good.
    max_abs_val = np.abs(shap_vals).max()
    if max_abs_val == 0: max_abs_val = 1
    
    html_elements = []
    for word, val in zip(words, shap_vals):
        normalized_val = val / max_abs_val
        # Red for good, blue for bad (or the other way around, depending
        # on how you look at it).
        if normalized_val > 0:
            color = f"rgba(255, 0, 0, {abs(normalized_val):.3f})" # Red
        else:
            color = f"rgba(0, 0, 255, {abs(normalized_val):.3f})" # Blue
        
        # I'm adding a tooltip so you can see the raw SHAP value.
        tooltip_text = f"SHAP Value: {val:.4f}"
        span = html.Span(
            children=word + " ",
            style={'background-color': color, 'padding': '2px', 'margin': '1px', 'border-radius': '3px'},
            title=tooltip_text
        )
        html_elements.append(span)
        
    return html.Div(html_elements)

# Let's get this app started.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# This is the layout of the app. I'm using Dash Bootstrap Components
# because it makes things look nice without a lot of work.
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Clinical Note Explainer", className="text-center my-4"), width=12)),
    dbc.Row([
        dbc.Col([
            html.H5("Paste a clinical note here:"),
            dcc.Textarea(id='text-input', value=sample_text, style={'width': '100%', 'height': 200}),
            html.Button('Explain', id='submit-button', n_clicks=0, className="mt-2")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(id="loading-output", children=[
            html.H4("Prediction:", className="mt-4"),
            html.Div(id='prediction-output', className="lead"),
            
            html.H4("What the model is thinking:", className="mt-4"),
            html.P("Hover over the words to see how much they contributed to the prediction. Red words pushed the prediction up, blue words pushed it down."),
            html.Div(id='highlighted-text-output', style={'border': '1px solid #ddd', 'padding': '10px', 'line-height': '2.0'}),
            
            html.H4("The most important words:", className="mt-4"),
            dcc.Graph(id='shap-plot')
        ]), width=12)
    ]),
], fluid=True)

# This is the main callback that does all the work.
@app.callback(
    Output('prediction-output', 'children'),
    Output('shap-plot', 'figure'),
    Output('highlighted-text-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('text-input', 'value')
)
def update_output(n_clicks, text_input):
    if n_clicks == 0 or not text_input:
        return "Paste some text and click 'Explain'.", go.Figure(), ""

    # Get the prediction and the SHAP values
    prediction_probas = predict_pipeline(np.array([text_input]))[0]
    predicted_class_index = np.argmax(prediction_probas)
    predicted_class_name = class_names[predicted_class_index]
    predicted_probability = prediction_probas[predicted_class_index]
    prediction_text = f"{predicted_class_name} ({predicted_probability:.2%})"

    shap_values = explainer([text_input])
    explanation = shap_values[0]

    try:
        # Create the waterfall plot
        shap_vals_for_class = explanation[:, predicted_class_index].values
        words = explanation.data
        non_zero_indices = np.where(shap_vals_for_class != 0)[0]
        # I'm only showing the top 20 words to keep the plot clean.
        num_features = min(20, len(non_zero_indices))
        sorted_indices = np.argsort(np.abs(shap_vals_for_class[non_zero_indices]))[-num_features:]
        top_words = np.array(words)[non_zero_indices][sorted_indices]
        top_shap_values = shap_vals_for_class[non_zero_indices][sorted_indices]

        fig = go.Figure(go.Waterfall(
            name="SHAP", orientation="h", y=top_words, x=top_shap_values,
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            increasing={"marker":{"color":"#d62728"}},
            decreasing={"marker":{"color":"#1f77b4"}}
        ))
        fig.update_layout(
            title=f"Top words for the '{predicted_class_name}' prediction",
            yaxis_title="Words", margin=dict(l=150, r=20, t=60, b=20)
        )
        
        # Create the highlighted text
        highlighted_text_html = shap_to_html(explanation, predicted_class_index)

    except Exception as e:
        print(f"Error creating plot: {e}")
        fig = go.Figure().update_layout(title="Couldn't make the plot.")
        highlighted_text_html = "Couldn't create the highlighted text."
        
    return prediction_text, fig, highlighted_text_html

# Let's go!
if __name__ == '__main__':
    app.run(debug=True)