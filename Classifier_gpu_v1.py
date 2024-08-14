from __future__ import print_function
import os
import sys
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from joblib import dump

# Ensure TensorFlow uses GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load data
dataset = r"KFF.csv"
data = pd.read_csv(dataset)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.drop(["URLs", "Type"], axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)

print(f"Dataset loaded and preprocessed.")

X = data.iloc[:, 1:].astype(np.float32)  
X = np.nan_to_num(X)
Y = data.iloc[:, 0].astype(int)  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

# Define MLP using TensorFlow
def MLP_Classifier(X_train, Y_train, X_test, Y_test):
    model = Sequential([
        Dense(150, input_dim=X_train.shape[1], activation='relu'),
        Dense(150, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    start = timer()
    model.fit(X_train, Y_train, epochs=10, batch_size=10, verbose=1)
    end = timer()
    
    Y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average="macro"),
        "recall": recall_score(Y_test, Y_pred, average="macro"),
        "time": end - start
    }

# Run the MLP Classifier
print("********************************")
print("MLP Classifier")
results = MLP_Classifier(X_train, Y_train, X_test, Y_test)
print("Total Execution Time:", results["time"])
print("Accuracy:", results["accuracy"])
print("Precision:", results["precision"])
print("Recall:", results["recall"])
