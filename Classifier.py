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
from sklearn.ensemble import VotingClassifier
import numpy as numm
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

# from genetic_selection import GeneticSelectionCV
data = pd.read_csv(r"Final_Features.csv")
print(data.dtypes)
print(data.shape)
lbl = data['Type'].value_counts()
print(lbl)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
print("***********")
data = data.drop(['URLs', 'Type'], axis=1)

data = data.reset_index()
print(data.shape)
print(data.columns)
# np.any(np.isnan(data))

X = data.iloc[:, 2:]
print(X.head())
print(type(X))

X = np.nan_to_num(X.astype(np.float32))
print(type(X))
Y = data['Label']
X_train, X_test, Y_train, Y_test = train_test_split(data, data.Label, test_size=0.3,
                                                    random_state=109)  # 70% training and 30% test

from sklearn.preprocessing import StandardScaler

def Rf():
    print("Training Columns:", X_train.columns)
    print("Training Data:", X_train.head())  # Displays the first few rows of X_train
    print("Testing Columns:", X_test.columns)
    
    # Create a scaler object and fit it to the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100)
    start = timer()
    # Train the model using the scaled training sets
    clf.fit(X_train_scaled, Y_train)

    Y_pred = clf.predict(X_test_scaled)
    end = timer()
    print(end - start)

    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(Y_test, Y_pred))

def Svm(X_train, X_test, Y_train, Y_test):
    clf1 = svm.SVC()  # Linear Kernel

    # Train the model using the training sets
    clf1.fit(X_train, Y_train)

    # Predict the response for test dataset
    Y_pred = clf1.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(Y_test, Y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(Y_test, Y_pred))


def LR():
    logreg = LogisticRegression()
    start = timer()
    # fit the model with data
    logreg.fit(X_train, Y_train)
    y_pred = logreg.predict(X_test)
    end = timer()
    print(end - start)

    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(Y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(Y_test, y_pred))
    cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
    print(cnf_matrix)


def KNN():
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    start = timer()
    # Train the model using the training sets
    knn.fit(X_train, Y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    end = timer()
    print(end - start)

    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(Y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(Y_test, y_pred))


def NB():
    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, Y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(Y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(Y_test, y_pred))


print("********************************")
print("NB")
NB()
print("********************************")
print('\n')
print("SVM Classifier")
Svm(X_train, X_test, Y_train, Y_test)
print("********************************")
print("RF classifier")
Rf()
print("********************************")
print("LR")
LR()
print("********************************")
print("KNN")
KNN()
