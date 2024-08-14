from __future__ import print_function
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.svm import SVC
from joblib import dump

# Load data
dataset = "CA_505.csv"
data = pd.read_csv(dataset)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.drop(["URL", "Type"], axis=1)
data = data.reset_index(drop=True)

label_data_divide = {0: 99998, 1: 10000}
label_0_sample_data = data[data["label"] == 0].sample(n=label_data_divide[0])
label_1_sample_data = data[data["label"] == 1].sample(n=label_data_divide[1])

df = pd.concat([label_0_sample_data, label_1_sample_data])

print(f"Dataset: {dataset}")
print(f"label 0 count", df[df["label"] == 0].shape[0])
print(f"label 1 count", df[df["label"] == 1].shape[0])

X = df.iloc[:, 2:].astype(np.float32)
X = np.nan_to_num(X)
Y = df["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

def run_classifier(classifier, X_train, Y_train, X_test, Y_test):
    # Start timing
    start = timer()
    # Train the classifier
    classifier.fit(X_train, Y_train)
    # Predict using the trained classifier
    Y_pred = classifier.predict(X_test)
    # End timing
    end = timer()
    # Calculate metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test, Y_pred, average='macro')
    # Print results
    print(f"Execution Time: {end - start} seconds")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

# Initialize the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Classifier
print("SVM Classifier")
svm_clf = SVC()
run_classifier(svm_clf, X_train_scaled, Y_train, X_test_scaled, Y_test)

# # Random Forest Classifier
# print("Random Forest Classifier")
# rf_clf = RandomForestClassifier(n_estimators=100)
# run_classifier(rf_clf, X_train, Y_train, X_test, Y_test)

# # Logistic Regression
# print("Logistic Regression")
# lr_clf = LogisticRegression()
# run_classifier(lr_clf, X_train_scaled, Y_train, X_test_scaled, Y_test)

# # K-Nearest Neighbors
# print("K-Nearest Neighbors")
# knn_clf = KNeighborsClassifier(n_neighbors=5)
# run_classifier(knn_clf, X_train_scaled, Y_train, X_test_scaled, Y_test)

# # Naive Bayes
# print("Naive Bayes")
# nb_clf = GaussianNB()
# run_classifier(nb_clf, X_train, Y_train, X_test, Y_test)
