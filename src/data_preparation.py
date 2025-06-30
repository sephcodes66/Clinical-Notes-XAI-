# FileName: data_preparation.py
# Description: This script loads the raw mtsamples dataset, cleans the clinical text data,
#              and saves a processed version ready for feature extraction and modeling.

import pandas as pd
import re
import os

def clean_clinical_text(text: str) -> str:
    """
    Applies a standard NLP preprocessing pipeline to clinical text.
    """
    # Ensure input is a string to prevent errors with missing data (e.g., NaN).
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Standardize text by removing special characters and normalizing whitespace.
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_data_preparation(
    input_filepath: str, 
    output_filepath: str,
    text_column: str,
    label_column: str
) -> None:
    """
    Orchestrates the data preparation pipeline: loading, cleaning, and saving the data.
    """
    print("Starting Phase 1: Data Preparation...")
    
    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded {len(df)} records from '{input_filepath}'.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {input_filepath}")
        print("Please ensure the raw data file is correctly placed.")
        return

    # Remove rows with missing text or labels, as they cannot be used for training.
    df.dropna(subset=[text_column, label_column], inplace=True)
    print(f"Working with {len(df)} complete records after removing missing values.")

    # Select and rename core columns for consistent naming in subsequent scripts.
    df_processed = df[[text_column, label_column]].copy()
    df_processed.rename(columns={
        text_column: 'text',
        label_column: 'label'
    }, inplace=True)

    print(f"Applying NLP cleaning pipeline to the '{text_column}' column...")
    df_processed['cleaned_text'] = df_processed['text'].apply(clean_clinical_text)
    
    # Retain only the final columns needed for the modeling phase.
    final_df = df_processed[['cleaned_text', 'label']]
    
    # Create the output directory if it doesn't exist to prevent save errors.
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    final_df.to_csv(output_filepath, index=False)
    
    print("Phase 1 complete.")
    print(f"Cleaned data saved to: {output_filepath}")
    print("\nPreview of processed data:")
    print(final_df.head())

if __name__ == "__main__":
    # Define file paths for raw input and processed output.
    RAW_DATA_PATH = "data/raw/mtsamples.csv"
    PROCESSED_DATA_PATH = "data/processed/processed_notes.csv"
    
    # Define the specific column names from the source mtsamples.csv dataset.
    TEXT_COLUMN_NAME = "transcription"
    LABEL_COLUMN_NAME = "medical_specialty"
    
    run_data_preparation(
        input_filepath=RAW_DATA_PATH,
        output_filepath=PROCESSED_DATA_PATH,
        text_column=TEXT_COLUMN_NAME,
        label_column=LABEL_COLUMN_NAME
    )