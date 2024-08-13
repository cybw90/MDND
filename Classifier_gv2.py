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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from joblib import dump


# Load data
data = pd.read_csv(r"UCIF.csv")
# data = pd.read_csv(r"demo.csv")
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.drop(['URLs', 'Type'], axis=1)
data = data.reset_index()
X = data.iloc[:, 2:].astype(np.float32)
X = np.nan_to_num(X)
Y = data['Label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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


#******************************************* MLP ***********************************************


def MLP_Classifier(X, y):
    # Split data into training and testing sets
    #X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=109)

    # Pipeline setup: includes data scaling and the MLP classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=300, random_state=21, verbose=True))
    ])

    # Define the hyperparameter grid for GridSearchCV
  
    param_grid = {
        'mlp__hidden_layer_sizes': [(150, 150), (200, 200)],  #  network configuraiton
        'mlp__alpha': [0.0001, 0.0005, 0.001],  # L2 penalty regularization 
        'mlp__activation': ['relu', 'tanh'],  # Activation functions
        'mlp__learning_rate_init': [0.001, 0.01],  # Initial learning rate values
        'mlp__solver': ['adam']  # weight optimization
    }

    # cross-validation using stratified k-folds
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
    start = timer()
    grid_search.fit(X_train, Y_train)
    end = timer()

    # Best model extraction from grid search
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)

    # Evaluation metrics
    print("Best Model Parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    print("ccuracy:", accuracy_score(Y_test, Y_pred))
    print("Precision:", precision_score(Y_test, Y_pred, average='macro'))
    print("Recall:", recall_score(Y_test, Y_pred, average='macro'))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    print("Time taken: {:.2f} seconds".format(end - start))

    # Save the trained model for deployment
    dump(best_model, 'final_mlp_classifier.joblib')


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
# print("********************************")
# print("SVM")
# Svm()
print("********************************")
print("MLP Classifier")
MLP_Classifier(X,Y)