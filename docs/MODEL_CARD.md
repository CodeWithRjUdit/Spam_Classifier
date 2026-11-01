# Model Card — SMS Spam Classifier

## Summary

This project is a binary text-classification demonstration that predicts whether an SMS/text message is spam or ham (not spam). It uses a saved Multinomial Naive Bayes classifier and a fitted TF-IDF vectorizer.

## Model details

- **Model family:** Multinomial Naive Bayes
- **Feature representation:** TF-IDF
- **Maximum TF-IDF features:** 3,000
- **Class mapping:** 1 = spam; 0 = ham
- **Serialized artifacts:** `model.pkl` and `vectorizer.pkl`
- **Text processing:** lowercase, NLTK word tokenization, alphanumeric-token filtering, English stop-word removal, Porter stemming

The serialized vectorizer and classifier are a matched pair. Do not independently retrain or replace one artifact without regenerating and validating the other.

## Data

The repository contains `spam.csv`, an SMS-labelled dataset, and `Spam Classifier.ipynb`, the original experiment notebook. The notebook should be treated as the primary record of the data preparation, model comparison, and training workflow.

This repository does not currently provide a documented dataset card, dataset licence assessment, or a fully reproducible training pipeline. Verify the source and usage terms of the dataset before redistributing it or using it commercially.

## Intended use

- Demonstration of a classical NLP text-classification workflow.
- Educational exploration of tokenization, stop-word removal, stemming, TF-IDF, and Naive Bayes.
- Low-stakes experimentation with example SMS messages.

## Out-of-scope use

- Sole decision-maker for blocking, reporting, or deleting messages.
- Security, fraud, financial, legal, or other high-impact decisions.
- Determining whether a URL, sender, attachment, or message is safe.
- Claims that a message is definitely malicious or definitely legitimate.

## Performance and evaluation

Do not infer model quality from the algorithm name or from a few successful examples. Consult the notebook for the recorded experiment outputs. A future evaluation should report, on a fixed held-out test set:

- Spam precision and recall.
- Spam F1-score.
- Confusion matrix (especially false negatives and false positives).
- Class distribution and test-set construction.
- Evaluation on recent, out-of-domain, and multilingual examples.

No unverified accuracy or production performance claim is made in this model card.

## Known limitations and risks

- The training data may not represent current spam tactics or all populations and languages.
- Tokenization, stop-word removal, and stemming can discard information that matters for some messages.
- TF-IDF represents token importance, not sender identity, URL reputation, message intent, or real-world context.
- False negatives may allow spam through; false positives may misclassify legitimate messages.
- Short, misspelled, code-switched, or multilingual messages may behave unpredictably.
- A ham prediction is not a safety guarantee.

## Privacy

The demo processes the submitted text to produce a prediction. Do not enter passwords, OTPs, financial details, personal identifiers, or other sensitive content.

## Maintenance

When changing preprocessing or model artifacts:
1. Update the training and inference logic together.
2. Regenerate both `model.pkl` and `vectorizer.pkl` from the same fitted pipeline.
3. Run artifact verification and prediction tests.
4. Record the dataset version, dependency versions, evaluation split, metrics, and artifact version.
