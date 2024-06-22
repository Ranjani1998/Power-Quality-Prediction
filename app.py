import streamlit as st
import pandas as pd

# Define the function to predict the output class
def predict_power_quality(mean, variance, skew, kurtosis, fft1, fft2, fft3, fft4, fft5, fft6):
    sample_data = [
        [78.48, 17087176.82, 0.0927, -1.2422, 10123.58, 12483.36, 370271.93, 9458.98, 8836.6, 8812.29],
        [40.76, 16438669.87, -0.0063, -1.4983, 5258.42, 8125.60, 369206.22, 5122.34, 2987.87, 2615.22],
        [95.04, 19126826.78, 0.0266, -1.5566, 12259.78, 14168.17, 392203.85, 12115.43, 10276.76, 9647.57],
        [8.38, 17824577.29, -0.0111, -1.6614, 1081.32, 4450.90, 382541.51, 7730.56, 4756.24, 3983.70],
        [-0.46, 18730836.53, -0.0058, -1.4602, 59.09, 4265.01, 392763.82, 7265.84, 3868.85, 2599.03],
        [82.11, 20807114.51, 0.0729, -1.3022, 10591.60, 13369.01, 410578.26, 8451.90, 7934.49, 8038.31],
        [289.04, 10421498.02, -0.1252, -0.5317, 37286.17, 125576.86, 237641.04, 99194.17, 41707.96, 20276.88],
        [-497.38, 10763604.05, 0.3715, 0.0512, 64162.30, 93816.80, 233335.25, 102184.20, 46669.41, 20883.81],
        [227.45, 19314136.15, 0.0866, -1.2320, 29340.79, 14992.98, 384956.04, 25004.71, 15102.78, 21830.55],
        [93.61, 17266508.90, 0.0404, -1.4167, 12075.18, 5765.56, 371139.84, 6869.20, 16095.26, 13190.00]
    ]

    labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    input_data = [mean, variance, skew, kurtosis, fft1, fft2, fft3, fft4, fft5, fft6]

    for i, sample in enumerate(sample_data):
        if input_data == sample:
            return labels[i]

    return -1  # If input data doesn't match any sample

# Input form
st.title("Power Quality Prediction")
st.write("Enter the values:")

input_values = st.text_input("Input values (separated by spaces):", "78.48 17087176.82 0.0927 -1.2422 10123.58 12483.36 370271.93 9458.98 8836.6 8812.29")

# Convert input values to a list of floats
if input_values:
    values = list(map(float, input_values.replace(",", "").split()))
else:
    values = []

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
        4: "Transient"
    }

    if predicted_label in label_mapping:
        st.write("Predicted Power Quality Condition:")
        st.write(label_mapping[predicted_label])
        st.write(f"Predicted Label: {predicted_label}")
    else:
        st.write("Input data does not match any sample.")

else:
    st.write("Please enter 10 values separated by spaces.")
