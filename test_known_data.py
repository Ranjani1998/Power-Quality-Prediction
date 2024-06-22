import numpy as np
import joblib
import tensorflow as tf

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

# Sample data from the training set with known labels
known_data = np.array([
    [0.1, 0.01, 0.2, 0.05, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05],  # Normal
    [0.2, 0.02, 0.4, 0.1, 1.2, 0.9, 0.6, 0.3, 0.2, 0.1],     # 3rd harmonic wave
    [0.15, 0.015, 0.35, 0.08, 1.5, 1.1, 0.8, 0.4, 0.25, 0.15], # 5th harmonic wave
    [0.25, 0.025, 0.45, 0.12, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1],  # Voltage dip
    [0.3, 0.03, 0.5, 0.15, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2]       # Transient
])

# Known labels
known_labels = ["Normal", "3rd harmonic wave", "5th harmonic wave", "Voltage dip", "Transient"]

for i, data in enumerate(known_data):
    input_data = data.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)
    prediction = model.predict(input_data_pca)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_mapping[predicted_class]
    print(f"Known label: {known_labels[i]}, Predicted: {predicted_label}")
    print(f"Input Data: {data}")
    print(f"Scaled Data: {input_data_scaled}")
    print(f"PCA Data: {input_data_pca}")

