# Project Structure and Architecture

## Design principles

- Keep `app.py` at the repository root because Streamlit Community Cloud uses it as the configured entry point.
- Keep reusable NLP preprocessing separate from the UI.
- Keep artifact loading and prediction in a dedicated inference module.
- Keep documentation under `docs/` and maintenance scripts under `scripts/`.
- Keep the model and vectorizer as a versioned pair.

## Module responsibilities

### `app.py`
Owns Streamlit page configuration, user input, submission handling, and result presentation. It should not contain training code.

### `spam_classifier/preprocessing.py`
Owns NLTK resource initialization and the exact text normalization sequence used at inference.

### `spam_classifier/inference.py`
Owns model/vectorizer loading, caching, and the prediction call. It receives raw text and returns the model's binary class.

### `model.pkl` and `vectorizer.pkl`
The trained classifier and fitted TF-IDF vocabulary. These are runtime artifacts, not source code.

### `scripts/verify_artifacts.py`
A lightweight local smoke test to check that the artifacts load and that a sample message can pass through preprocessing, vectorization, and prediction.

## Runtime flow

1. Streamlit executes root-level `app.py`.
2. `load_classifier()` loads the trusted model and vectorizer once and initializes NLTK resources.
3. The user submits a message.
4. `classify_message()` calls `transform_text()`.
5. The fitted vectorizer converts the processed text into the feature space expected by the model.
6. Multinomial Naive Bayes returns class 0 (ham) or 1 (spam).
7. The UI displays the prediction with a safety caveat.

## Change management

If the training preprocessing changes, update `spam_classifier/preprocessing.py` to match. If the vectorizer or classifier is retrained, commit both artifacts from the same training run. Never mix a newly fitted vectorizer with an older classifier.

## Future engineering work

- Move training from the notebook into a reproducible script.
- Add unit tests for normalization and integration tests for prediction.
- Track dataset provenance and licence.
- Add evaluation reports and model artifact metadata.
- Consider a single scikit-learn Pipeline artifact to reduce the risk of model/vectorizer mismatch, after verifying compatibility with the existing training workflow.
