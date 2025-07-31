# data_preparation.py
# This script takes the raw mtsamples data, cleans up the clinical notes,
# and saves a nice, clean version for the next steps.

import pandas as pd
import re
import os

def clean_text(text):
    """A simple text cleaning function for the clinical notes."""
    # Make sure we're working with a string
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Get rid of special characters and extra whitespace
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(input_file, output_file, text_col, label_col):
    """Loads, cleans, and saves the data."""
    print("--- Starting Data Prep ---")
    
    try:
        notes_df = pd.read_csv(input_file)
        print(f"Loaded {len(notes_df)} notes from '{input_file}'.")
    except FileNotFoundError:
        print(f"Error: Can't find the data at {input_file}")
        print("Make sure you've downloaded the mtsamples.csv file.")
        return

    # Drop any rows that are missing text or a label
    notes_df.dropna(subset=[text_col, label_col], inplace=True)
    print(f"Down to {len(notes_df)} notes after dropping NaNs.")

    # Grab the columns we need and give them simpler names
    clean_notes_df = notes_df[[text_col, label_col]].copy()
    clean_notes_df.rename(columns={
        text_col: 'text',
        label_col: 'label'
    }, inplace=True)

    print(f"Cleaning up the '{text_col}' column...")
    clean_notes_df['cleaned_text'] = clean_notes_df['text'].apply(clean_text)
    
    # Just keep the columns we need for modeling
    final_df = clean_notes_df[['cleaned_text', 'label']]
    
    # Make sure the output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print("--- Data Prep Complete ---")
    print(f"Cleaned data saved to: {output_file}")
    print("\nHere's a peek at the cleaned data:")
    print(final_df.head())

if __name__ == "__main__":
    # File paths
    RAW_DATA_PATH = "data/raw/mtsamples.csv"
    PROCESSED_DATA_PATH = "data/processed/processed_notes.csv"
    
    # The columns we care about in the raw CSV
    TEXT_COLUMN = "transcription"
    LABEL_COLUMN = "medical_specialty"
    
    prepare_data(
        input_file=RAW_DATA_PATH,
        output_file=PROCESSED_DATA_PATH,
        text_col=TEXT_COLUMN,
        label_col=LABEL_COLUMN
    )