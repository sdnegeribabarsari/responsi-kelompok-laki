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
st.title('Prediksi Tingkat Pencemaran Udara DKI Jakarta 🌫️🌆')

# Input fitur numerik
input_features = []
feature_names = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'max']
st.write("Masukkan nilai parameter pencemaran udara:")
for feature in feature_names:
    value = st.number_input(f'{feature} (masukkan nilai)', value=0, step=1)
    input_features.append(value)

# Konversi input menjadi DataFrame
df = pd.DataFrame([input_features], columns=feature_names)

# Fungsi prediksi
def prediction(df):
    # Skala fitur numerik menggunakan Scaler yang dimuat
    scaled_data = scaler.transform(df)

    # Bentuk ulang input untuk LSTM
    input_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

    # Ekstrak fitur menggunakan model LSTM
    features = lstm_model.predict(input_data)

    # Lakukan prediksi menggunakan model SVM
    prediction = svm_classifier.predict(features)

    # Interpretasi hasil prediksi
    if prediction == 1:
        prediction = 'Tingkat Pencemaran Udara Berbahaya ⚠️'
    else:
        prediction = 'Tingkat Pencemaran Udara Aman ✅'
    return prediction

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    prediction_result = prediction(df)
    st.write('Hasil prediksi:', prediction_result)