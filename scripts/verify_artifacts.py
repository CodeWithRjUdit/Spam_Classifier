"""Run a local smoke test for the committed spam-classifier artifacts."""

from spam_classifier.inference import classify_message, load_classifier


def main():
    classifier = load_classifier()
    examples = (
        "Congratulations! You have won a free prize. Claim now.",
        "Hey, are we still meeting for lunch today?",
    )

    for message in examples:
        prediction = classify_message(message, classifier)
        label = "spam" if prediction == 1 else "ham"
        print(f"{label:4} | {message}")

    print("\nArtifact loading and inference smoke test completed.")


if __name__ == "__main__":
    main()
