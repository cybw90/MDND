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

# Load data
data = pd.read_csv(r"Final_dataset.csv")
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.drop(['URLs', 'Type'], axis=1)
data = data.reset_index()
X = data.iloc[:, 2:].astype(np.float32)
X = np.nan_to_num(X)
Y = data['Label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

# Random Forest Classifier
def Rf():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100)
    start = timer()
    clf.fit(X_train_scaled, Y_train)
    Y_pred = clf.predict(X_test_scaled)
    end = timer()
    print("RF Execution Time:", end - start)
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    print("Recall:", metrics.recall_score(Y_test, Y_pred))

# SVM Classifier
def Svm():
    clf = svm.SVC()
    start = timer()
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    end = timer()
    print("SVM Execution Time:", end - start)
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    print("Recall:", metrics.recall_score(Y_test, Y_pred))

# Logistic Regression
def LR():
    logreg = LogisticRegression()
    start = timer()
    logreg.fit(X_train, Y_train)
    y_pred = logreg.predict(X_test)
    end = timer()
    print("LR Execution Time:", end - start)
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    print("Precision:", metrics.precision_score(Y_test, y_pred))
    print("Recall:", metrics.recall_score(Y_test, y_pred))

# KNN Classifier
def KNN():
    knn = KNeighborsClassifier(n_neighbors=5)
    start = timer()
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    end = timer()
    print("KNN Execution Time:", end - start)
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    print("Precision:", metrics.precision_score(Y_test, y_pred))
    print("Recall:", metrics.recall_score(Y_test, y_pred))

# Naive Bayes Classifier
def NB():
    gnb = GaussianNB()
    start = timer()
    gnb.fit(X_train, Y_train)
    y_pred = gnb.predict(X_test)
    end = timer()
    print("NB Execution Time:", end - start)
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    print("Precision:", metrics.precision_score(Y_test, y_pred))
    print("Recall:", metrics.recall_score(Y_test, y_pred))

# Execute classifiers
print("Naive Bayes")
NB()
print("SVM Classifier")
Svm()
print("Random Forest Classifier")
Rf()
print("Logistic Regression")
LR()
print("KNN Classifier")
KNN()
