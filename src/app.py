import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import string

# App title and description
st.title("📰 Fake News Detection with LSTM")
st.write("This app analyzes a news article and predicts whether it is **Real** or **Fake**.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\mmmdu\OneDrive\Documents\github\projects\Fake-News-Detection-NLP\models\lstm_fake_news_model.h5")
    return model

@st.cache_resource
def load_tokenizer():
    with open(r"C:\Users\mmmdu\OneDrive\Documents\github\projects\Fake-News-Detection-NLP\models\tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# User input
text = st.text_area("Enter the news article text below:", height=200)

# Predict button
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.success(f"🔍 Prediction: **Real News** ✅ ({prediction:.2f})")
        else:
            st.error(f"🚨 Prediction: **Fake News** ❌ ({prediction:.2f})")
