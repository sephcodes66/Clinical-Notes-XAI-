# train_classifier.py
# This script trains a simple Logistic Regression classifier on our BERT
# embeddings. It also evaluates the model and saves it for later use.

import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model(embeddings_path, labels_path, model_output_path):
    """
    Loads the data, trains a classifier, and saves it.
    """
    print("--- Starting Classifier Training ---")

    # Load the embeddings and labels
    print(f"Loading data from '{embeddings_path}' and '{labels_path}'...")
    X = np.load(embeddings_path)
    y = np.load(labels_path)

    # Just to be safe, let's re-encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    print(f"Data loaded: {X.shape[0]} samples")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data into {len(X_train)} training and {len(X_test)} testing samples.")

    # I'm using a simple Logistic Regression model here. It's fast and
    # works pretty well.
    print("Training the classifier...")
    # I had to increase max_iter to get it to converge.
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_train, y_train)
    print("Done training.")
    
    # Save the label encoder classes so we can get the original labels back
    print("Saving label encoder classes...")
    np.save('data/features/label_encoder_classes.npy', label_encoder.classes_)

    # Let's see how well our model did
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # This will give us a more detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to '{model_output_path}'")
    
    print("\n--- Classifier Training Complete ---")


if __name__ == "__main__":
    # Input files
    EMBEDDINGS_PATH = "data/features/bert_embeddings.npy"
    LABELS_PATH = "data/features/labels.npy"
    
    # Output file
    MODEL_PATH = "models/logistic_regression_classifier.joblib"
    
    train_model(
        embeddings_path=EMBEDDINGS_PATH,
        labels_path=LABELS_PATH,
        model_output_path=MODEL_PATH
    )