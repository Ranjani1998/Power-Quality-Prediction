import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:\Users\Subaranjani\Desktop\Research Clients\Vasanth\PowerQualityDistributionDataset1.csv')

# Handling missing values (if any)
df.fillna(df.mean(), inplace=True)

# Adjust labels from [1, 2, 3, 4, 5] to [0, 1, 2, 3, 4]
df['output'] = df['output'] - 1

# Separate features and target
X = df.iloc[:, :-1].values  # Columns Col1 to Col128
y = df.iloc[:, -1].values   # Column 'output'

# Feature Engineering
def extract_features(data):
    features = []
    for row in data:
        row_features = []
        row_features.append(np.mean(row))
        row_features.append(np.var(row))
        row_features.append(skew(row))
        row_features.append(kurtosis(row))
        row_features.extend(np.abs(fft(row))[:64])  # First 64 FFT components
        features.append(row_features)
    return np.array(features)

X_features = extract_features(X)

# Normalize the data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_features)

# Dimensionality Reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Load the trained model
model = tf.keras.models.load_model('power_quality_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', '3rd harmonic wave', '5th harmonic wave', 'Voltage dip', 'Transient'], yticklabels=['Normal', '3rd harmonic wave', '5th harmonic wave', 'Voltage dip', 'Transient'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualization of one wave from each class after applying Fourier transformation
classes = ['Normal', '3rd harmonic wave', '5th harmonic wave', 'Voltage dip', 'Transient']
for i, cls in enumerate(classes):
    wave = X[i][0:128]
    wave[0:128] = np.abs(fft(wave[0:128]))
    xf = fftfreq(128, 1/128)
    plt.plot(xf, wave)
    plt.title(f"Class {cls} wave")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()
