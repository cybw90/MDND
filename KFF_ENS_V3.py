from __future__ import print_function
import os
import sys
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from joblib import dump

# Function to calculate weighted averages
def calculate_weighted_average(sensitivity, specificity, weights=[0.2, 0.4, 0.5, 0.6, 0.8]):
    weighted_averages = {}
    for w in weights:
        weighted_average = w * sensitivity + (1 - w) * specificity
        weighted_averages[w] = weighted_average
    return weighted_averages

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
    accuracies, precisions, recalls, times, weighted_results = [], [], [], [], []
    for _ in range(3):  # Run the classifier three times
        results = classifier_func(*args)
        accuracies.append(results["accuracy"])
        precisions.append(results["precision"])
        recalls.append(results["recall"])
        times.append(results["time"])
        weighted_results.append(results["weighted_averages"])

    # Print aggregated results from three runs
    print("Total Execution Time:", sum(times))
    print("Total Accuracy:", sum(accuracies))
    print("Total Precision:", sum(precisions))
    print("Total Recall:", sum(recalls))
    print("Average Execution Time:", np.mean(times))
    print("Average Accuracy:", np.mean(accuracies))
    print("Average Precision:", np.mean(precisions))
    print("Average Recall:", np.mean(recalls))

    # Print the weighted averages from the runs
    for i, weighted_avg in enumerate(weighted_results, start=1):
        print(f"Weighted Averages for Run {i}: {weighted_avg}")

def NB(X_train, Y_train, X_test, Y_test):
    clf = GaussianNB()
    start = timer()
    clf.fit(X_train, Y_train)
    end = timer()
    Y_pred = clf.predict(X_test)

    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate weighted averages for different weights
    weighted_averages = calculate_weighted_average(sensitivity, specificity)

    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average="macro"),
        "recall": recall_score(Y_test, Y_pred, average="macro"),
        "time": end - start,
        "weighted_averages": weighted_averages
    }

def Rf(X_train, Y_train, X_test, Y_test):
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-1)  # Optimized criterion and parallel processing
    start = timer()
    clf.fit(X_train, Y_train)
    end = timer()
    Y_pred = clf.predict(X_test)

    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate weighted averages for different weights
    weighted_averages = calculate_weighted_average(sensitivity, specificity)

    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average="macro"),
        "recall": recall_score(Y_test, Y_pred, average="macro"),
        "time": end - start,
        "weighted_averages": weighted_averages
    }

def Svm(X_train, Y_train, X_test, Y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = svm.LinearSVC(max_iter=1000)  # Using LinearSVC for faster computation and early stopping
    start = timer()
    clf.fit(X_train_scaled, Y_train)
    end = timer()
    Y_pred = clf.predict(X_test_scaled)

    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate weighted averages for different weights
    weighted_averages = calculate_weighted_average(sensitivity, specificity)

    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average="macro"),
        "recall": recall_score(Y_test, Y_pred, average="macro"),
        "time": end - start,
        "weighted_averages": weighted_averages
    }

def LR(X_train, Y_train, X_test, Y_test):
    clf = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')  # Adjusted class weights for better recall
    start = timer()
    clf.fit(X_train, Y_train)
    end = timer()
    Y_pred = clf.predict(X_test)

    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate weighted averages for different weights
    weighted_averages = calculate_weighted_average(sensitivity, specificity)

    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average="macro"),
        "recall": recall_score(Y_test, Y_pred, average="macro"),
        "time": end - start,
        "weighted_averages": weighted_averages
    }

def KNN(X_train, Y_train, X_test, Y_test):
    clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')  # Reduced neighbors and optimized algorithm
    start = timer()
    clf.fit(X_train, Y_train)
    end = timer()
    Y_pred = clf.predict(X_test)

    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate weighted averages for different weights
    weighted_averages = calculate_weighted_average(sensitivity, specificity)

    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average="macro"),
        "recall": recall_score(Y_test, Y_pred, average="macro"),
        "time": end - start,
        "weighted_averages": weighted_averages
    }

def Ensemble_Classifier(X_train, Y_train, X_test, Y_test):
    # Define the base classifiers
    clf1 = RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-1)
    clf2 = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
    clf3 = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    clf4 = GaussianNB()

    # Create a Voting Classifier with optimized weights
    ensemble = VotingClassifier(
        estimators=[('rf', clf1), ('lr', clf2), ('knn', clf3), ('nb', clf4)],
        voting='soft',  # Use 'soft' voting to account for probability estimates
        weights=[2, 1, 1, 1]  # Giving more weight to Random Forest
    )

    # Training the ensemble model
    start = timer()
    ensemble.fit(X_train, Y_train)
    end = timer()

    # Evaluating the ensemble model
    Y_pred = ensemble.predict(X_test)

    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate weighted averages for different weights
    weighted_averages = calculate_weighted_average(sensitivity, specificity)

    # Compare Precision and Recall
    precision = precision_score(Y_test, Y_pred, average="macro")
    recall = recall_score(Y_test, Y_pred, average="macro")
    print(f"Precision: {precision}, Recall: {recall}")

    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision,
        "recall": recall,
        "time": end - start,
        "weighted_averages": weighted_averages
    }

# Execute classifiers
print("********************************")
print("Naive Bayes")
run_classifier_three_times(NB, X_train, Y_train, X_test, Y_test)
print("********************************")
print("Random Forest Classifier")
run_classifier_three_times(Rf, X_train, Y_train, X_test, Y_test)
print("********************************")
print("Logistic Regression")
run_classifier_three_times(LR, X_train, Y_train, X_test, Y_test)
print("********************************")
print("KNN Classifier")
run_classifier_three_times(KNN, X_train, Y_train, X_test, Y_test)
print("********************************")
print("SVM")
run_classifier_three_times(Svm, X_train, Y_train, X_test, Y_test)
print("********************************")
print("Ensemble Classifier")
run_classifier_three_times(Ensemble_Classifier, X_train, Y_train, X_test, Y_test)
