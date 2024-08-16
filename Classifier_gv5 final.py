from __future__ import print_function
import os
import sys
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from joblib import dump

from utils import limit_memory, track_memory

x_mb = int(sys.argv[2])

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


def run_classifier_three_times(classifier_func, *args):
    accuracies, precisions, recalls, times = [], [], [], []
    for _ in range(3):  # Run the classifier three times
        results = classifier_func(*args)
        accuracies.append(results["accuracy"])
        precisions.append(results["precision"])
        recalls.append(results["recall"])
        times.append(results["time"])

    # Print aggregated results from three runs
    print("Total Execution Time:", sum(times))
    print("Total Accuracy:", sum(accuracies))
    print("Total Precision:", sum(precisions))
    print("Total Recall:", sum(recalls))
    print("Average Execution Time:", np.mean(times))
    print("Average Accuracy:", np.mean(accuracies))
    print("Average Precision:", np.mean(precisions))
    print("Average Recall:", np.mean(recalls))


# Define each classifier function including MLP
# @track_memory
def MLP_Classifier(X_train, Y_train, X_test, Y_test):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(150, 150),
                    alpha=0.0005,
                    activation="relu",
                    learning_rate_init=0.001,
                    solver="adam",
                    max_iter=300,
                    random_state=21,
                    verbose=True,
                ),
            ),
        ]
    )

    # Training the model
    start = timer()
    pipeline.fit(X_train, Y_train)
    end = timer()

    # Evaluating the model
    Y_pred = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average="macro"),
        "recall": recall_score(Y_test, Y_pred, average="macro"),
        "time": end - start,
    }

    # Save the trained model

    # dump(pipeline, 'mlp_classifier.joblib')
    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('mlp', MLPClassifier(hidden_layer_sizes=(150, 150), alpha=0.0005, activation='relu',
    #                           learning_rate_init=0.001, solver='adam', max_iter=300, random_state=21))
    # ])
    # param_grid = {
    #     'mlp__hidden_layer_sizes': [(150, 150), (200, 200)],
    #     'mlp__alpha': [0.0001, 0.0005, 0.001],
    #     'mlp__activation': ['relu', 'tanh'],
    #     'mlp__learning_rate_init': [0.001, 0.01],
    #     'mlp__solver': ['adam']
    # }
    # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
    # grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
    # start = timer()
    # grid_search.fit(X_train, Y_train)
    # end = timer()
    # best_model = grid_search.best_estimator_
    # Y_pred = best_model.predict(X_test)


# Remaining classifiers
def Rf():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100)
    return run_classifier_three_times(
        Rf, X_train_scaled, Y_train, X_test_scaled, Y_test
    )


def Svm():
    clf = svm.SVC()
    return run_classifier_three_times(Svm, X_train, Y_train, X_test, Y_test)


def LR():
    clf = LogisticRegression()
    return run_classifier_three_times(LR, X_train, Y_train, X_test, Y_test)


def KNN():
    clf = KNeighborsClassifier(n_neighbors=5)
    return run_classifier_three_times(KNN, X_train, Y_train, X_test, Y_test)


def NB():
    clf = GaussianNB()
    return run_classifier_three_times(NB, X_train, Y_train, X_test, Y_test)


limit_memory(x_mb * 1024 * 1024)

# Execute classifiers
print("********************************")
print("MLP Classifier")
run_classifier_three_times(MLP_Classifier, X_train, Y_train, X_test, Y_test)
# print("********************************")
# print("Naive Bayes")
# run_classifier_three_times(NB, X_train, Y_train, X_test, Y_test)
# print("********************************")
# print("Random Forest Classifier")
# run_classifier_three_times(Rf, X_train, Y_train, X_test, Y_test)
# print("********************************")
# print("Logistic Regression")
# run_classifier_three_times(LR, X_train, Y_train, X_test, Y_test)
# print("********************************")
# print("KNN Classifier")
# run_classifier_three_times(KNN, X_train, Y_train, X_test, Y_test)
# print("********************************")
# print("SVM")
# run_classifier_three_times(Svm, X_train, Y_train, X_test, Y_test)
