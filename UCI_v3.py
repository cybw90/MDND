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
from joblib import dump

# Load data
data = pd.read_csv(r"CA_505.csv")
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.drop(['URL', 'Type'], axis=1)
data = data.reset_index(drop=True)
X = data.iloc[:, 1:].astype(np.float32)
X = np.nan_to_num(X)
Y = data['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def run_classifier_three_times(classifier_func, *args):
    accuracies, precisions, recalls, times = [], [], [], []
    for _ in range(3):  # Run the classifier three times
        results = classifier_func(*args)
        accuracies.append(results['accuracy'])
        precisions.append(results['precision'])
        recalls.append(results['recall'])
        times.append(results['time'])

    # Print aggregated results from three runs
    print("Total Execution Time:", sum(times))
    print("Total Accuracy:", sum(accuracies))
    print("Total Precision:", sum(precisions))
    print("Total Recall:", sum(recalls))
    print("Average Execution Time:", np.mean(times))
    print("Average Accuracy:", np.mean(accuracies))
    print("Average Precision:", np.mean(precisions))
    print("Average Recall:", np.mean(recalls))

def MLP_Classifier(X_train_scaled, Y_train, X_test_scaled, Y_test):
    pipeline = Pipeline([
        ('mlp', MLPClassifier(hidden_layer_sizes=(150, 150), alpha=0.0005, activation='relu',
                              learning_rate_init=0.001, solver='adam', max_iter=300, random_state=21))
    ])
    param_grid = {
        'mlp__hidden_layer_sizes': [(150, 150), (200, 200)],
        'mlp__alpha': [0.0001, 0.0005, 0.001],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__learning_rate_init': [0.001, 0.01],
        'mlp__solver': ['adam']
    }
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
    start = timer()
    grid_search.fit(X_train_scaled, Y_train)
    end = timer()
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test_scaled)
    return {
        'accuracy': accuracy_score(Y_test, Y_pred),
        'precision': precision_score(Y_test, Y_pred, average='macro'),
        'recall': recall_score(Y_test, Y_pred, average='macro'),
        'time': end - start
    }

# Execute and print results for each classifier
print("********************************")
print("Random Forest")
run_classifier_three_times(random_forest, X_train_scaled, Y_train, X_test_scaled, Y_test)
print("********************************")
print("SVM Classifier")
run_classifier_three_times(svm_classifier, X_train_scaled, Y_train, X_test_scaled, Y_test)
print("********************************")
print("Logistic Regression")
run_classifier_three_times(logistic_regression, X_train_scaled, Y_train, X_test_scaled, Y_test)
print("********************************")
print("KNN Classifier")
run_classifier_three_times(knn_classifier, X_train_scaled, Y_train, X_test_scaled, Y_test)
print("********************************")
print("Naive Bayes Classifier")
run_classifier_three_times(naive_bayes, X_train_scaled, Y_train, X_test_scaled, Y_test)
print("********************************")
print("MLP Classifier")
run_classifier_three_times(MLP_Classifier, X_train_scaled, Y_train, X_test_scaled, Y_test)
