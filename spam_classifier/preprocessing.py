"""Text normalization used by the SMS spam classifier."""

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def load_nltk_resources():
    """Ensure tokenizer and English stop-word resources are available."""
    resources = (
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    )

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(package_name, quiet=True):
                raise RuntimeError(
                    f"Unable to download required NLTK resource: {package_name}"
                )

    return set(stopwords.words("english")), PorterStemmer()


def transform_text(text, english_stopwords=None, stemmer=None):
    """Apply lowercase, word tokenization, filtering, stop-word removal and stemming."""
    if not isinstance(text, str):
        raise TypeError("Input message must be a string.")

    if english_stopwords is None or stemmer is None:
        english_stopwords, stemmer = load_nltk_resources()

    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum()]

    normalized_tokens = [
        stemmer.stem(token)
        for token in filtered_tokens
        if token not in english_stopwords and token not in string.punctuation
    ]
    return " ".join(normalized_tokens)
