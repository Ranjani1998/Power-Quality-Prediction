import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the saved model, scaler, and PCA
model = tf.keras.models.load_model('power_quality_model.h5')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')

# Define the mapping of class indices to categorical values
class_mapping = {
    0: "Normal",
    1: "3rd harmonic wave",
    2: "5th harmonic wave",
    3: "Voltage dip",
    4: "Transient"
}

# Streamlit app
st.title("Power Quality Prediction")

# Input features
st.header("Input features")
mean_val = st.number_input("Mean value", value=0.0)
variance_val = st.number_input("Variance value", value=0.0)
skew_val = st.number_input("Skew value", value=0.0)
kurtosis_val = st.number_input("Kurtosis value", value=0.0)
fft_vals = [st.number_input(f"FFT component {i+1}", value=0.0) for i in range(6)]

# Prepare input for prediction
input_data = np.array([mean_val, variance_val, skew_val, kurtosis_val] + fft_vals).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)
input_data_pca = pca.transform(input_data_scaled)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_pca)
    predicted_class = np.argmax(prediction, axis=1)[0]
    st.write(f"The power quality condition is: {class_mapping[predicted_class]}")

