import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# Load the model
model = tf.keras.models.load_model('power_quality_model.h5')

# Load the fitted scaler and PCA
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

# Debugging: Print input data
st.write("Input data:", input_data)

# Ensure the input data has the correct number of features
if input_data.shape[1] != 10:
    st.error(f"Input data must have 10 features, but has {input_data.shape[1]}")
else:
    # Scale and transform the input data
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)

    # Debugging: Print scaled and PCA-transformed data
    st.write("Scaled data:", input_data_scaled)
    st.write("PCA-transformed data:", input_data_pca)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(input_data_pca)
        predicted_class = np.argmax(prediction, axis=1)[0]
        st.write(f"The power quality condition is: {class_mapping[predicted_class]}")
