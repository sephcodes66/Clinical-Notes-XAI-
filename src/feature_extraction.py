# FileName: feature_extraction.py
# Description: This script uses a pre-trained ClinicalBERT model to generate fixed-size
#              vector embeddings from cleaned clinical text. These embeddings serve as
#              features for the downstream classification model.

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
from sklearn.preprocessing import LabelEncoder

def generate_bert_embeddings(
    input_filepath: str,
    embeddings_output_path: str,
    labels_output_path: str,
    model_name: str,
    batch_size: int = 16
):
    """
    Generates [CLS] token embeddings for text data using a specified BERT model.
    Processes data in batches to manage memory usage.

    Args:
        input_filepath (str): Path to the processed CSV file with 'cleaned_text' and 'label' columns.
        embeddings_output_path (str): Path to save the generated embeddings (.npy file).
        labels_output_path (str): Path to save the corresponding encoded labels (.npy file).
        model_name (str): The Hugging Face model identifier.
        batch_size (int): The number of text samples to process at a time.
    """
    print("Starting Phase 2: Feature Extraction with ClinicalBERT...")

    # Set device to GPU for faster processing if available, otherwise default to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading processed data from '{input_filepath}'...")
    df = pd.read_csv(input_filepath).dropna(subset=['cleaned_text', 'label'])
    texts = df['cleaned_text'].tolist()

    # Convert string labels (e.g., 'Cardiovascular / Pulmonary') into integer format for the model.
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['label'])
    # Save the encoder's class mappings to reconstruct original labels later.
    np.save('data/features/label_encoder_classes.npy', label_encoder.classes_)
    
    print(f"Loading tokenizer and model: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    # Set model to evaluation mode, which disables layers like dropout.
    model.eval()

    all_embeddings = []
    print(f"Starting embedding generation in batches of {batch_size}...")

    # Process texts in batches to prevent out-of-memory errors with large datasets.
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Disable gradient calculations to conserve memory and speed up inference.
        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            outputs = model(**inputs)

            # Use the hidden state of the [CLS] token as the sentence-level embedding.
            # This vector is designed to capture the aggregate meaning of the input text.
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        print(f"  Processed batch {i//batch_size + 1} of {len(texts)//batch_size + 1}")

    # Combine the embeddings from all batches into a single numpy array.
    final_embeddings = np.vstack(all_embeddings)

    # Save embeddings and labels to disk. This decouples the computationally expensive
    # feature extraction from the model training and evaluation steps.
    print(f"Saving embeddings to '{embeddings_output_path}'...")
    np.save(embeddings_output_path, final_embeddings)

    print(f"Saving encoded labels to '{labels_output_path}'...")
    np.save(labels_output_path, encoded_labels)

    print("\nPhase 2 Complete.")
    print(f"Embeddings shape: {final_embeddings.shape}")
    print(f"Labels shape: {encoded_labels.shape}")


if __name__ == "__main__":
    PROCESSED_DATA_PATH = "data/processed/processed_notes.csv"
    
    os.makedirs("data/features", exist_ok=True)
    EMBEDDINGS_PATH = "data/features/bert_embeddings.npy"
    LABELS_PATH = "data/features/labels.npy"

    # Specify the pre-trained model to be used for feature extraction.
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    
    # Run the feature extraction process. Note: this is a feature extraction approach,
    # the ClinicalBERT model itself is not being fine-tuned here.
    generate_bert_embeddings(
        input_filepath=PROCESSED_DATA_PATH,
        embeddings_output_path=EMBEDDINGS_PATH,
        labels_output_path=LABELS_PATH,
        model_name=MODEL_NAME
    )