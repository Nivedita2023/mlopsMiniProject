import os
import re
import pickle
import logging

import dagshub
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Pre-download and explicitly force-load NLTK datasets
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
wordnet.ensure_loaded()

# Global Lemmatizer & Stopwords initialization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

mlflow.set_tracking_uri('https://dagshub.com/niveditaranjan223883/mlopsMiniProject.mlflow')
dagshub.init(repo_owner='niveditaranjan223883', repo_name='mlopsMiniProject', mlflow=True)

# Initialize Flask app

app = Flask(__name__)

# Base path resolution for local artifacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# ==========================================================
# Text Normalization Utilities
# ==========================================================

def lemmatization(text: str) -> str:
    """Lemmatize words in text safely."""
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])


def remove_stop_words(text: str) -> str:
    """Remove stop words from text."""
    words = [word for word in str(text).split() if word not in stop_words]
    return " ".join(words)


def removing_numbers(text: str) -> str:
    """Remove digits from text."""
    return "".join([char for char in text if not char.isdigit()])


def lower_case(text: str) -> str:
    """Convert text to lowercase."""
    return " ".join([word.lower() for word in text.split()])


def removing_punctuations(text: str) -> str:
    """Remove punctuation and normalize spacing."""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(text: str) -> str:
    """Full text cleaning workflow."""
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_urls(text)
    text = removing_punctuations(text)
    text = lemmatization(text)
    return text

# ==========================================================
# Load Model & Feature Artifacts
# ==========================================================

def load_model_artifact(model_name: str):
    """
    Load model locally from models/model.pkl if available,
    otherwise fallback to DagsHub MLflow Model Registry.
    """
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Loading local model from disk: {LOCAL_MODEL_PATH}")
        with open(LOCAL_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    print(f"Local model not found. Fetching from MLflow Registry for '{model_name}'...")
    client = mlflow.MlflowClient()

    try:
        model_uri = f"models:/{model_name}@staging"
        return mlflow.pyfunc.load_model(model_uri)

    except Exception as e:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_version = max([int(v.version) for v in versions])
            model_uri = f"models:/{model_name}/{latest_version}"
            return mlflow.pyfunc.load_model(model_uri)
        else:
            raise RuntimeError(f"No version found for '{model_name}'.")


# Load Model and Local Vectorizer
model = load_model_artifact("my-model")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ==========================================================
# Flask Routes
# ==========================================================

@app.route('/')
def home():
    """Render main web interface with clean initial state."""
    return render_template('index.html', result=None, user_text="")


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment class for incoming text and preserve user input."""
    if request.method == 'POST':
        raw_text = request.form.get('text', '')

        # Check for empty or whitespace-only inputs
        if not raw_text.strip():
            return render_template('index.html', result="Please enter text.", user_text=raw_text)

        # Clean input text
        clean_text = normalize_text(raw_text)

        # Transform text into feature array
        features = vectorizer.transform([clean_text])
        features_array = features.toarray()

        # Predict class
        prediction = model.predict(features_array)
        raw_prediction = int(prediction[0])

        # Map label: 1 = Positive (happiness), 0 = Negative (sadness)
        prediction_label = "Positive" if raw_prediction == 1 else "Negative"

        # Pass user_text back so the textarea keeps what the user typed
        return render_template('index.html', result=prediction_label, user_text=raw_text)


# ==========================================================
# Entrypoint
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)