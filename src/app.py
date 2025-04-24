import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Başlık ve stil
st.title("📰 Fake News Detection with LSTM")
st.write("Bu uygulama, haber metnini analiz ederek **Gerçek** mi yoksa **Sahte** mi olduğunu tahmin eder.")

# Model ve tokenizer yükle
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

# Kullanıcıdan metin al
text = st.text_area("Haberi buraya yazın:", height=200)

# Tahmin yap
if st.button("Tahmin Et"):
    if text.strip() == "":
        st.warning("Lütfen bir metin girin.")
    else:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.success(f"🔍 Tahmin: **Gerçek Haber** ✅ ({prediction:.2f})")
        else:
            st.error(f"🚨 Tahmin: **Sahte Haber** ❌ ({prediction:.2f})")