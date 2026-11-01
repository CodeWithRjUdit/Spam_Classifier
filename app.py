import streamlit as st

from spam_classifier.inference import classify_message, load_classifier


st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="✉️",
    layout="centered",
)

st.title("SMS Spam Classifier")
st.write(
    "Check whether a text message resembles spam or legitimate (ham) messages "
    "learned by the machine-learning model."
)
st.caption("Multinomial Naive Bayes · TF-IDF text features")

try:
    classifier = load_classifier()
except Exception as exc:
    st.error("The classifier could not be initialized.")
    st.exception(exc)
    st.stop()

with st.form("spam_classifier_form"):
    message = st.text_area(
        "Message to check",
        placeholder="Paste or type a message here…",
        height=160,
        max_chars=10000,
    )
    submitted = st.form_submit_button("Classify message", type="primary")

if submitted:
    if not message.strip():
        st.warning("Please enter a message before classifying it.")
    else:
        try:
            label = classify_message(message, classifier)
            st.divider()
            if label == 1:
                st.error("Prediction: Spam")
                st.write(
                    "The model identifies this message as spam. Treat links, "
                    "payment requests, and requests for personal information "
                    "with caution."
                )
            else:
                st.success("Prediction: Not spam (ham)")
                st.write("The model identifies this message as a legitimate message.")
        except Exception as exc:
            st.error("The message could not be processed.")
            st.exception(exc)

st.divider()
st.caption(
    "Educational demonstration only. Model predictions can be wrong; "
    "do not use this result as the sole basis for deciding whether a message is safe."
)
