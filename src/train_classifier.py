# FileName: train_classifier.py
# Description: This script trains a downstream Logistic Regression classifier using the
#              pre-computed BERT embeddings as features. It evaluates the model on a
#              held-out test set and saves the final trained classifier for inference.

import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_classifier(
    embeddings_path: str,
    labels_path: str,
    model_output_path: str
):
    """
    Loads embeddings and labels, trains a Logistic Regression classifier,
    evaluates it, and saves the trained model.

    Args:
        embeddings_path (str): Path to the saved BERT embeddings (.npy file).
        labels_path (str): Path to the saved encoded labels (.npy file).
        model_output_path (str): Path to save the trained classifier object.
    """
    print("Starting Phase 3: Downstream Classifier Training...")

    # Load the pre-computed ClinicalBERT embeddings (features) and the encoded labels (target).
    print(f"Loading features from '{embeddings_path}'...")
    X = np.load(embeddings_path)
    y = np.load(labels_path)

    # Re-fit the label encoder to ensure labels are in a consistent 0-to-N-1 format.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    print(f"Data loaded. Features shape: {X.shape}, Labels shape: {y.shape}")

    # Split data into training and testing sets, stratifying by label to maintain class distribution.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # A Logistic Regression model is used as a lightweight and interpretable downstream classifier.
    print("Training the Logistic Regression classifier...")
    # Increase max_iter to ensure the model converges with high-dimensional embedding data.
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Save the label encoder classes to map predictions back to original string labels.
    print("Saving label encoder classes...")
    np.save('data/features/label_encoder_classes.npy', label_encoder.classes_)

    # Evaluate model performance on the unseen test set.
    print("Evaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Generate a detailed report of precision, recall, and F1-score for each class.
    print("\nClassification Report:")
    # Note: `y_pred` are integer-encoded; use label_encoder.classes_ to see string labels.
    print(classification_report(y_test, y_pred))

    # Persist the trained model to disk for later use in prediction/inference tasks.
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nTrained model saved successfully to '{model_output_path}'")
    
    print("\nPhase 3 Complete.")


if __name__ == "__main__":
    # Input paths for the features and labels generated in the previous phase.
    EMBEDDINGS_PATH = "data/features/bert_embeddings.npy"
    LABELS_PATH = "data/features/labels.npy"
    
    # Define the output path for the final trained classifier object.
    MODEL_PATH = "models/logistic_regression_classifier.joblib"
    
    train_classifier(
        embeddings_path=EMBEDDINGS_PATH,
        labels_path=LABELS_PATH,
        model_output_path=MODEL_PATH
    )