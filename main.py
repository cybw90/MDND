import os
import sys
import numpy as np
import pandas as pd
import sklearn

from timeit import default_timer as timer
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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


# # Take equal proportion of test data
# df_0 = df[df["Label"] == 0]
# df_1 = df[df["Label"] == 1]

# X_0 = np.nan_to_num(df_0.iloc[:, 2:].astype(np.float32))
# X_1 = np.nan_to_num(df_1.iloc[:, 2:].astype(np.float32))

# Y_0 = df_0["Label"]
# Y_1 = df_1["Label"]

# X_train_0, X_test_0, Y_train_0, Y_test_0 = train_test_split(
#     X_0, Y_0, test_size=0.15, random_state=109
# )
# X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(
#     X_1, Y_1, test_size=0.15, random_state=109
# )

# X_train = np.concat([X_train_0, X_train_1])
# X_test = np.concat([X_test_0, X_test_1])
# Y_train = np.concat([Y_train_0, Y_train_1])
# Y_test = np.concat([Y_test_0, Y_test_1])


def run_classifier_three_times(classifier, X_train, Y_train, X_test, Y_test):
    # with sklearn.config_context(working_memory=1):
    accuracies, precisions, recalls, times = [], [], [], []
    for _ in range(3):  # Run the classifier three times
        start = timer()
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        end = timer()

        accuracies.append(metrics.accuracy_score(Y_test, Y_pred))
        precisions.append(metrics.precision_score(Y_test, Y_pred))
        recalls.append(metrics.recall_score(Y_test, Y_pred))
        times.append(end - start)

    # Print aggregated results from three runs
    print("Total Execution Time:", sum(times))
    print("Total Accuracy:", sum(accuracies))
    print("Total Precision:", sum(precisions))
    print("Total Recall:", sum(recalls))
    print("Average Execution Time:", np.mean(times))
    print("Average Accuracy:", np.mean(accuracies))
    print("Average Precision:", np.mean(precisions))
    print("Average Recall:", np.mean(recalls))


# Random Forest Classifier
#@track_memory
def Rf():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100)
    run_classifier_three_times(clf, X_train_scaled, Y_train, X_test_scaled, Y_test)


# SVM Classifier
#@track_memory
def Svm():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with sklearn.config_context(working_memory=1):
        clf = svm.SVC()
        run_classifier_three_times(clf, X_train_scaled, Y_train, X_test_scaled, Y_test)


# Logistic Regression
#@track_memory
def LR():
    clf = LogisticRegression()
    run_classifier_three_times(clf, X_train, Y_train, X_test, Y_test)


# KNN Classifier
#@track_memory
def KNN():
    clf = KNeighborsClassifier(n_neighbors=5)
    run_classifier_three_times(clf, X_train, Y_train, X_test, Y_test)


# Naive Bayes Classifier
#@track_memory
def NB():
    # with sklearn.config_context(working_memory=0):
    clf = GaussianNB()
    run_classifier_three_times(clf, X_train, Y_train, X_test, Y_test)


limit_memory(x_mb * 1024 * 1024)

# # Execute classifiers
#print("********************************")
#print("Random Forest Classifier")
#Rf()
#print("********************************")
#print("Logistic Regression")
#LR()
#print("********************************")
#print("SVM")
#Svm()
print("********************************")
print("KNN Classifier")
KNN()
#print("********************************")
#print("Naive Bayes")
#NB()
