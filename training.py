import os
import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import tandardScaler  

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#Import the data into a dataframe
def model_run(
    path_to_dataset: str,
    model_dir: str,
):
    try:
        # Carica il dataset Iris
        iris = load_iris()
        X, y = iris.data, iris.target

        # Dividi il dataset in set di addestramento e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizza le caratteristiche
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Crea e addestra il modello SVM
        model = SVC(kernel='linear', random_state=42)
        model.fit(X_train, y_train)

        # Effettua previsioni sul set di test
        y_pred = model.predict(X_test)

        # Calcola l'accuratezza del modello
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuratezza del modello: {accuracy:.2f}")

        model.save_weights(rf'{model_dir}/my_model_weights.weights.h5')
        model.save(rf'{model_dir}/my_model.keras')

        return 'Ha funzionato'
    
    except Exception as e:
        return str(e)

model_dir = os.environ['SM_MODEL_DIR']

print('#####################################')

input_dir = os.environ['SM_INPUT_DIR']
output = model_run(
    rf"{input_dir}/data/training/cancer_data.csv",
    model_dir
)
    
with open(model_dir + '/output_model.txt', 'w') as f:
    f.write(output)
    
output_dir = os.environ['SM_OUTPUT_DIR']
with open(output_dir + '/output.txt', 'w') as f:
    f.write('Ciao sono i log del training')