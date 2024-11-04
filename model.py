from sklearn.datasets import load_diabetes
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# Carica il dataset di esempio
diabetes = load_diabetes()
data, targets = diabetes.data, diabetes.target

# Dividi i dati in train e test
train_data, test_data = data[:400], data[400:]
train_targets, test_targets = targets[:400], targets[400:]

# Normalizza i dati
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Definisci il modello
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Crea il modello
model = build_model()

# Addestra il modello
model.fit(train_data, train_targets, epochs=100, batch_size=16, verbose=1, validation_split=0.2)

# Valuta il modello
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(f"Mean Absolute Error on test data: {test_mae_score}")

