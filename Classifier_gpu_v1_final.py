from __future__ import print_function
import os
import sys
from timeit import default_timer as timer
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# from utils import track_memory, limit_memory, reset_limit_memory


# x_mb = int(sys.argv[2])

# Ensure TensorFlow uses GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
print("GPUs", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load data
label_data_divide = {0: 99998, 1: 10000}
dataset = sys.argv[1]
label_data_divided_filename = (
    f"{dataset}__0_{label_data_divide[0]}__1_{label_data_divide[1]}.csv"
)

print(f"Dataset: {dataset}")

if os.path.exists(label_data_divided_filename):
    print("Extracted samples' file already exist. Loading the file.")
    df = pd.read_csv(label_data_divided_filename)
else:
    print("Extracted samples' does not exist. Creating the samples' file.")
    data = pd.read_csv(f"{dataset}.csv")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.drop(["URLs", "Type"], axis=1)
    data = data.reset_index(drop=True)
    label_0_sample_data = data[data["Label"] == 0].sample(
        n=label_data_divide[0], random_state=100
    )
    label_1_sample_data = data[data["Label"] == 1].sample(
        n=label_data_divide[1], random_state=100
    )
    df = pd.concat([label_0_sample_data, label_1_sample_data])
    df.to_csv(label_data_divided_filename)

print(f"Label 0 count", df[df["Label"] == 0].shape[0])
print(f"Label 1 count", df[df["Label"] == 1].shape[0])

X = df.iloc[:, 2:].astype(np.float32)
X = np.nan_to_num(X)
Y = df["Label"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=109
)


# Define MLP using TensorFlow
# @track_memory
def MLP_Classifier(X_train, Y_train, X_test, Y_test):
    model = Sequential(
        [
            Dense(150, input_dim=X_train.shape[1], activation="relu"),
            Dense(150, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    start = timer()
    history = model.fit(X_train, Y_train, epochs=10, batch_size=10, verbose=1)
    end = timer()

    Y_pred = (model.predict(X_test) > 0.5).astype(int)

    return {
        "history": history.history,
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average="macro"),
        "recall": recall_score(Y_test, Y_pred, average="macro"),
        "time": end - start,
    }
    reset_limit_memory()


accuracies, precisions, recalls, times = [], [], [], []

# Run the MLP Classifier
print("********************************")
print("MLP Classifier")
rh_df = pd.DataFrame()
for i in range(3):
    results = MLP_Classifier(X_train, Y_train, X_test, Y_test)
    times.append(results["time"])
    accuracies.append(results["accuracy"])
    precisions.append(results["precision"])
    recalls.append(results["recall"])
    history = {"run": [i for _ in range(len(results["history"]["accuracy"]))]}
    history.update(
        {"epoch": [k for k in range(1, len(results["history"]["accuracy"]) + 1)]}
    )
    history.update({**results["history"]})
    rh_df = pd.concat([rh_df, pd.DataFrame(history)])
    # print("Total Execution Time:", results["time"])
    # print("Accuracy:", results["accuracy"])
    # print("Precision:", results["precision"])
    # print("Recall:", results["recall"])

rh_df.reset_index(inplace=True)
rh_df.to_csv(f"{label_data_divided_filename}_training_history.csv")

# Print aggregated results from three runs
print("Total Execution Time:", sum(times))
print("Total Accuracy:", sum(accuracies))
print("Total Precision:", sum(precisions))
print("Total Recall:", sum(recalls))
print("Average Execution Time:", np.mean(times))
print("Average Accuracy:", np.mean(accuracies))
print("Average Precision:", np.mean(precisions))
print("Average Recall:", np.mean(recalls))

# memory_info = tf.config.experimental.get_memory_info("GPU:0")
# print("Memory info:", memory_info)
