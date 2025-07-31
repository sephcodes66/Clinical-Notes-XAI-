# feature_extraction.py
# This script uses a pre-trained ClinicalBERT model to turn our cleaned text
# into numerical embeddings. These embeddings are what we'll feed into our
# classifier.

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
from sklearn.preprocessing import LabelEncoder

def create_embeddings(
    input_file,
    embeddings_file,
    labels_file,
    model_name,
    batch_size=16
):
    """
    Creates BERT embeddings for our text data. It processes the data in
    batches so we don't run out of memory.
    """
    print("--- Starting Feature Extraction ---")

    # Use a GPU if we have one, otherwise, we'll just use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading data from '{input_file}'...")
    df = pd.read_csv(input_file).dropna(subset=['cleaned_text', 'label'])
    texts = df['cleaned_text'].tolist()

    # Convert the text labels to numbers
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['label'])
    # Save the label encoder so we can decode the labels later
    np.save('data/features/label_encoder_classes.npy', label_encoder.classes_)
    
    print(f"Loading model: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval() # Set the model to evaluation mode

    all_embeddings = []
    print(f"Generating embeddings in batches of {batch_size}...")

    # Process the text in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Don't calculate gradients since we're not training
        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            outputs = model(**inputs)

            # We'll use the [CLS] token embedding as the sentence embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        print(f"  Processed batch {i//batch_size + 1} of {len(texts)//batch_size + 1}")

    # Stack all the embeddings together
    final_embeddings = np.vstack(all_embeddings)

    # Save the embeddings and labels
    print(f"Saving embeddings to '{embeddings_file}'...")
    np.save(embeddings_file, final_embeddings)

    print(f"Saving labels to '{labels_file}'...")
    np.save(labels_file, encoded_labels)

    print("\n--- Feature Extraction Complete ---")
    print(f"Embeddings shape: {final_embeddings.shape}")
    print(f"Labels shape: {encoded_labels.shape}")


if __name__ == "__main__":
    PROCESSED_DATA_PATH = "data/processed/processed_notes.csv"
    
    os.makedirs("data/features", exist_ok=True)
    EMBEDDINGS_PATH = "data/features/bert_embeddings.npy"
    LABELS_PATH = "data/features/labels.npy"

    # This is the model we'll use to create the embeddings
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    
    # TODO: Add a command-line argument to let the user specify the model
    create_embeddings(
        input_file=PROCESSED_DATA_PATH,
        embeddings_file=EMBEDDINGS_PATH,
        labels_file=LABELS_PATH,
        model_name=MODEL_NAME
    )
