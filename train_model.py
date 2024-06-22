import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
import joblib
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\Subaranjani\Desktop\Research Clients\Vasanth\PowerQualityDistributionDataset1.csv')

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
        row_features.extend(np.abs(fft(row))[:6])  # First 6 FFT components
        features.append(row_features)
    return np.array(features)

X_features = extract_features(X)

# Normalize the data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_features)

# Save the fitted scaler
joblib.dump(scaler, 'scaler.joblib')

# Dimensionality Reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Save the fitted PCA
joblib.dump(pca, 'pca.joblib')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define class weights to handle class imbalance
class_weights = {0: 1.0, 1: 1.0, 2: 4.0, 3: 1.0, 4: 1.0}

# Define a more complex MLP model for tabular data
def create_mlp(input_shape, num_classes, learning_rate, dropout_rate):
    inputs = Input(shape=input_shape)
    x = Dense(512, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter tuning with Optuna
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = create_mlp((X_train_smote.shape[1],), num_classes=5, learning_rate=learning_rate, dropout_rate=dropout_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

    history = model.fit(X_train_smote, y_train_smote,
                        epochs=50, batch_size=batch_size, validation_split=0.2,
                        class_weight=class_weights,
                        callbacks=[early_stopping, reduce_lr], verbose=0)

    val_loss = min(history.history['val_loss'])
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Train final model with best hyperparameters
best_params = study.best_params
model = create_mlp((X_train_smote.shape[1],), num_classes=5, learning_rate=best_params['learning_rate'], dropout_rate=best_params['dropout_rate'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

history = model.fit(X_train_smote, y_train_smote,
                    epochs=50, batch_size=best_params['batch_size'], validation_split=0.2,
                    class_weight=class_weights,
                    callbacks=[early_stopping, reduce_lr], verbose=1)

# Save the model
model.save('power_quality_model.h5')
