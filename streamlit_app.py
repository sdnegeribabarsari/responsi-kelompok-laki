import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load Scaler, LSTM model, dan SVM classifier
with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model/lstm_model.pkl', 'rb') as file:
    lstm_model = pickle.load(file)

with open('model/svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Judul Streamlit
st.title('Prediksi apakah Jamur ini bisa dimakan? 🍲😋⁉')

# Input fitur numerik
input_features = []
feature_names = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'max']
for feature in feature_names:
    value = st.number_input(f'Masukkan nilai untuk {feature}', value=0, step=1)
    input_features.append(value)

df = pd.DataFrame([input_features], columns=feature_names)

def prediction(df):
    # Skala fitur numerik menggunakan Scaler yang dimuat
    scaled_data = scaler.transform(df)

    # Bentuk ulang input untuk LSTM
    input_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

    # Ekstrak fitur menggunakan model LSTM
    features = lstm_model.predict(input_data)

    # Lakukan prediksi menggunakan model SVM
    prediction = svm_classifier.predict(features)

    if prediction:
        prediction = 'beracun ☠, jangan ya dek ya 🙅'
    else:
        prediction = 'bisa dimakan 🍲😋, gasin aja bang👍'
    return prediction

if st.button("Prediksi"):
    prediction_result = prediction(df)
    st.write('Jamur ini merupakan jamur yang', prediction_result)