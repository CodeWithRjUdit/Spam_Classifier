# SMS Spam Classifier

A Streamlit web app that classifies a text message as **spam** or **not spam** using a trained Multinomial Naive Bayes model.

## How it works

1. The message is converted to lowercase and tokenized with NLTK.
2. Non-alphanumeric tokens and English stop words are removed.
3. Remaining words are reduced using Porter stemming.
4. The processed message is transformed with the TF-IDF vectorizer saved during training.
5. The trained Multinomial Naive Bayes model predicts the class.

The model was trained using the SMS Spam Collection dataset. The TF-IDF vectorizer uses a maximum vocabulary size of 3,000 features. The preprocessing steps in the app mirror the training notebook.

## Run locally

Use Python 3.11 for the pinned scikit-learn version.

```bash
git clone https://github.com/CodeWithRjUdit/Spam_Classifier.git
cd Spam_Classifier
pip install -r requirements.txt
streamlit run app.py
```

The app opens locally in your browser at the address printed by Streamlit (typically http://localhost:8501).

## Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select **Create app**.
3. Choose the repository `CodeWithRjUdit/Spam_Classifier`.
4. Set the branch to `main` and the main file path to `app.py`.
5. In **Advanced settings**, select Python 3.11, then deploy.
6. Once the build completes, use the app URL provided by Streamlit as the single public link to this project.

## Repository files

- `app.py` — Streamlit interface and inference pipeline.
- `model.pkl` — trained Multinomial Naive Bayes model.
- `vectorizer.pkl` — fitted TF-IDF vectorizer used by the model.
- `Spam Classifier.ipynb` — training and exploratory analysis notebook.
- `spam.csv` — source dataset.
- `requirements.txt` — deployment dependencies.

## Important note

This project is an educational demonstration. The classifier can make mistakes, including on new message styles or messages that differ from the training data. Do not treat its output as a guarantee that a message is safe.
