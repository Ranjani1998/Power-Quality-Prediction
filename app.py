import streamlit as st
import pandas as pd
import numpy as np

# Define the function to predict the output class
def predict_power_quality(mean, variance, skew, kurtosis, fft1, fft2, fft3, fft4, fft5, fft6):
    # Here you can define your model or logic to predict the class
    # For simplicity, we use a dummy logic here
    # Replace this with your actual model
    if fft1 > 10000:
        return 1  # 3rd harmonic wave
    elif fft3 > 350000:
        return 2  # 5th harmonic wave
    elif mean > 200:
        return 3  # Voltage dip
    elif variance > 18000000:
        return 4  # transient
    else:
        return 0  # Normal

# Input form
st.title("Power Quality Prediction")
st.write("Enter the values:")

input_values = st.text_input("Input values (separated by spaces):", "78.48 17087176.82 0.0927 -1.2422 10123.58 12483.36 370271.93 9458.98 8836.6 8812.29")

# Convert input values to a list of floats
values = list(map(float, input_values.split()))

if len(values) == 10:
    mean, variance, skew, kurtosis, fft1, fft2, fft3, fft4, fft5, fft6 = values

    # Create a DataFrame to display the input values
    data = {
        "Mean Value": [mean],
        "Variance Value": [variance],
        "Skew Value": [skew],
        "Kurtosis Value": [kurtosis],
        "FFT Component 1": [fft1],
        "FFT Component 2": [fft2],
        "FFT Component 3": [fft3],
        "FFT Component 4": [fft4],
        "FFT Component 5": [fft5],
        "FFT Component 6": [fft6]
    }

    df = pd.DataFrame(data)

    st.write("Input Values:")
    st.table(df)

    # Predict the output
    predicted_label = predict_power_quality(mean, variance, skew, kurtosis, fft1, fft2, fft3, fft4, fft5, fft6)

    # Define the mapping of output labels
    label_mapping = {
        0: "Normal",
        1: "3rd harmonic wave",
        2: "5th harmonic wave",
        3: "Voltage dip",
        4: "transient"
    }

    st.write("Predicted Power Quality Condition:")
    st.write(label_mapping[predicted_label])
else:
    st.write("Please enter 10 values separated by spaces.")
