"""Load the trained artifacts and run spam classification."""

from pathlib import Path
import pickle

import streamlit as st

from spam_classifier.preprocessing import load_nltk_resources, transform_text


ROOT_DIR = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_classifier():
    """Load trusted, repository-managed model artifacts once per app process."""
    model_path = ROOT_DIR / "model.pkl"
    vectorizer_path = ROOT_DIR / "vectorizer.pkl"

    missing = [
        path.name
        for path in (model_path, vectorizer_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifact(s): " + ", ".join(missing)
        )

    # Pickle files can execute code when loaded. Only load trusted artifacts.
    with model_path.open("rb") as file:
        model = pickle.load(file)
    with vectorizer_path.open("rb") as file:
        vectorizer = pickle.load(file)

    stop_words, stemmer = load_nltk_resources()
    return {
        "model": model,
        "vectorizer": vectorizer,
        "stop_words": stop_words,
        "stemmer": stemmer,
    }


def classify_message(message, classifier):
    """Return the binary prediction: 1 = spam, 0 = ham."""
    processed = transform_text(
        message,
        english_stopwords=classifier["stop_words"],
        stemmer=classifier["stemmer"],
    )
    features = classifier["vectorizer"].transform([processed])
    return int(classifier["model"].predict(features)[0])
