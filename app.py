import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import string
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pickled model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)



# Ensure required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

def transform_text(text):
    # Initialize the Porter Stemmer
    ps = PorterStemmer()
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Filter alphanumeric tokens
    filtered_tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords and punctuation, and apply stemming
    final_tokens = [
        ps.stem(word) for word in filtered_tokens 
        if word not in stopwords.words('english') and word not in string.punctuation
    ]
    
    return " ".join(final_tokens)


# Title for the Streamlit app
st.title("Spam Message Classifier")

# Input for the user to enter a message
message = st.text_area("Enter a message to classify", "")

# Function to predict whether the message is spam or not
def predict_spam(message):
    # Transform the message using the same vectorizer used for training

    processed_message = transform_text(message)
    
    message_vectorized = vectorizer.transform([processed_message])
    
    # Predict using the loaded model
    prediction = model.predict(message_vectorized)
    
    if prediction == 1:
        return "Spam"
    else:
        return "Not Spam"

# Button to classify the message
if st.button("Classify"):
    if message != "":
        result = predict_spam(message)
        st.write(f"The message is: {result}")
    else:
        st.write("Please enter a message to classify.")
