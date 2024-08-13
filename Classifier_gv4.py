from __future__ import print_function
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load data
dataset = r"50_Features.csv"
data = pd.read_csv(dataset)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.drop(["URLs", "Type"], axis=1)
data = data.reset_index(drop=True)

label_data_divide = {0: 99998, 1: 10000}
label_0_sample_data = data[data["Label"] == 0].sample(n=label_data_divide[0])
label_1_sample_data = data[data["Label"] == 1].sample(n=label_data_divide[1])

df = pd.concat([label_0_sample_data, label_1_sample_data])

print(f"Dataset: {dataset}")
print(f"Label 0 count", df[df["Label"] == 0].shape[0])
print(f"Label 1 count", df[df["Label"] == 1].shape[0])

X = df.iloc[:, 2:].astype(np.float32)
X = np.nan_to_num(X)
Y = df["Label"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

def run_classifier_three_times(classifier, X_train, Y_train, X_test, Y_test):
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
def Rf():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100)
    run_classifier_three_times(clf, X_train_scaled, Y_train, X_test_scaled, Y_test)

# SVM Classifier
def Svm():
    clf = svm.SVC()
    run_classifier_three_times(clf, X_train, Y_train, X_test, Y_test)

# Logistic Regression
def LR():
    clf = LogisticRegression()
    run_classifier_three_times(clf, X_train, Y_train, X_test, Y_test)

# KNN Classifier
def KNN():
    clf = KNeighborsClassifier(n_neighbors=5)
    run_classifier_three_times(clf, X_train, Y_train, X_test, Y_test)

# Naive Bayes Classifier
def NB():
    clf = GaussianNB()
    run_classifier_three_times(clf, X_train, Y_train, X_test, Y_test)

# Execute classifiers
# print("********************************")
# print("Naive Bayes")
# NB()
# print("********************************")
# print("Random Forest Classifier")
# Rf()
# print("********************************")
# print("Logistic Regression")
# LR()
# print("********************************")
# print("KNN Classifier")
# KNN()
print("********************************")
print("SVM")
Svm()
