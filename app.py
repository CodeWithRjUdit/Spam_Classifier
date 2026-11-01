from pathlib import Path
import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model_assets():
    """Load the model and its matching TF-IDF vectorizer once per app process."""
    model_path = BASE_DIR / "model.pkl"
    vectorizer_path = BASE_DIR / "vectorizer.pkl"

    if not model_path.is_file() or not vectorizer_path.is_file():
        missing = [
            path.name
            for path in (model_path, vectorizer_path)
            if not path.is_file()
        ]
        raise FileNotFoundError(
            "Required model files are missing: " + ", ".join(missing)
        )

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    with vectorizer_path.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


@st.cache_resource
def load_text_processing_resources():
    """Download NLTK data only when it is not already available."""
    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            downloaded = nltk.download(package_name, quiet=True)
            if not downloaded:
                raise RuntimeError(
                    f"Could not download the NLTK resource: {package_name}"
                )

    english_stopwords = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    return english_stopwords, stemmer


def transform_text(text, english_stopwords, stemmer):
    """Apply the same preprocessing used when the classifier was trained."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalnum()]

    final_tokens = [
        stemmer.stem(word)
        for word in filtered_tokens
        if word not in english_stopwords and word not in string.punctuation
    ]

    return " ".join(final_tokens)


st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="✉️",
    layout="centered",
)

st.title("SMS Spam Classifier")
st.write(
    "Enter a text message below to check whether the trained machine-learning "
    "model classifies it as **spam** or **not spam**."
)
st.caption(
    "Model: Multinomial Naive Bayes · Text features: TF-IDF"
)

try:
    model, vectorizer = load_model_assets()
    english_stopwords, stemmer = load_text_processing_resources()
except Exception as exc:
    st.error("The classifier could not be initialized.")
    st.exception(exc)
    st.stop()

with st.form("spam_classifier_form"):
    message = st.text_area(
        "Message to check",
        placeholder="Example: Congratulations! You have won a free prize...",
        height=150,
    )
    submitted = st.form_submit_button("Classify message", type="primary")

if submitted:
    if not message.strip():
        st.warning("Please enter a message before classifying it.")
    else:
        processed_message = transform_text(
            message,
            english_stopwords,
            stemmer,
        )
        message_vectorized = vectorizer.transform([processed_message])
        prediction = int(model.predict(message_vectorized)[0])

        st.divider()
        if prediction == 1:
            st.error("Spam message")
            st.write(
                "The model classifies this message as **spam**. "
                "Be cautious with links, payment requests, and requests for "
                "personal information."
            )
        else:
            st.success("Not spam")
            st.write("The model classifies this message as **not spam**.")

st.caption(
    "This is a machine-learning demonstration. Predictions can be incorrect; "
    "do not use the result as the only basis for deciding whether a message is safe."
)
