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

# Load data
data = pd.read_csv(r"CA_505.csv")
# data = pd.read_csv(r"demo.csv")
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.drop(['URL', 'Type'], axis=1)
data = data.reset_index()
X = data.iloc[:, 2:].astype(np.float32)
X = np.nan_to_num(X)
Y = data['label']
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

# MLP Classifier **************************************** 94% in 163.5 *******************************

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, auc
# from timeit import default_timer as timer
# from joblib import dump

# def MLP_Classifier(X, y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=109)
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('mlp', MLPClassifier(hidden_layer_sizes=(150, 150), alpha=0.0005,
#                               activation='relu', learning_rate_init=0.001,
#                               solver='adam', max_iter=300, random_state=21, verbose=True))
#     ])
    
#     # Training the model
#     start = timer()
#     pipeline.fit(X_train, Y_train)
#     end = timer()

#     # Evaluating the model
#     Y_pred = pipeline.predict(X_test)
#     print("Test Set Accuracy:", accuracy_score(Y_test, Y_pred))
#     print("Test Set Precision:", precision_score(Y_test, Y_pred, average='macro'))
#     print("Test Set Recall:", recall_score(Y_test, Y_pred, average='macro'))
#     print("Classification Report:\n", classification_report(Y_test, Y_pred))
#     print("Time taken: {:.2f} seconds".format(end - start))

#     # Save the trained model
#     dump(pipeline, 'mlp_classifier.joblib')

# # Example usage
# # Assuming X and Y are defined as your dataset features and target
# MLP_Classifier(X, Y)


#*************************************** Best Time/ 94% ***************************************************

# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
# from timeit import default_timer as timer
# import numpy as np
# import pandas as pd

# def MLP_Classifier(X, y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=109)

#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('mlp', MLPClassifier(max_iter=300, random_state=21, verbose=True))
#     ])

#     param_grid = {
#         'mlp__hidden_layer_sizes': [(150, 150)],
#         'mlp__alpha': [0.0005],
#         'mlp__activation': ['relu'],
#         'mlp__learning_rate_init': [0.001],
#         'mlp__solver': ['adam']
#     }

#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
#     grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
#     start = timer()
#     grid_search.fit(X_train, Y_train)
#     end = timer()

#     print("Best Model Parameters:", grid_search.best_params_)
#     print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
#     Y_pred = grid_search.best_estimator_.predict(X_test)

#     print("Test Set Accuracy:", accuracy_score(Y_test, Y_pred))
#     print("Test Set Precision:", precision_score(Y_test, Y_pred, average='macro'))
#     print("Test Set Recall:", recall_score(Y_test, Y_pred, average='macro'))
#     print("Classification Report:\n", classification_report(Y_test, Y_pred))
#     print("Time taken: {:.2f} seconds".format(end - start))

# # Example usage with your dataset
# MLP_Classifier(X, Y)

#******************************************* 94.5%  with Hyper tuning Paper version ***********************************************

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from joblib import dump
from timeit import default_timer as timer
import numpy as np
import pandas as pd

def MLP_Classifier(X, y):
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=109)

    # Pipeline setup: includes data scaling and the MLP classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=300, random_state=21, verbose=True))
    ])

    # Define the hyperparameter grid for GridSearchCV
    # Grid Search will explore different combinations of parameters to find the best configuration
    param_grid = {
        'mlp__hidden_layer_sizes': [(150, 150), (200, 200)],  # Varying the network structure
        'mlp__alpha': [0.0001, 0.0005, 0.001],  # L2 penalty (regularization term)
        'mlp__activation': ['relu', 'tanh'],  # Activation functions
        'mlp__learning_rate_init': [0.001, 0.01],  # Initial learning rate values
        'mlp__solver': ['adam']  # Solver for weight optimization
    }

    # Setup for cross-validation using stratified k-folds
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
    print("Test Set Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Test Set Precision:", precision_score(Y_test, Y_pred, average='macro'))
    print("Test Set Recall:", recall_score(Y_test, Y_pred, average='macro'))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    print("Time taken: {:.2f} seconds".format(end - start))

    # Save the trained model for deployment
    dump(best_model, 'final_mlp_classifier.joblib')

# # Example usage with your dataset
# # Assuming X and Y are defined as your dataset features and target
# MLP_Classifier(X, Y)
#***************************************** 94.6 with timer  3050 seconds ************************************************#
# import threading
# import time
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
# import numpy as np
# import pandas as pd

# def print_time_elapsed(start_time, keep_running):
#     """ Function to print time elapsed in seconds, will run in a separate thread. """
#     while keep_running['active']:
#         elapsed_time = time.time() - start_time
#         print(f"Time elapsed: {elapsed_time:.2f} seconds")
#         time.sleep(2)  # Update every 2 seconds

# def MLP_Classifier(X, y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=109)

#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('mlp', MLPClassifier(max_iter=500, random_state=21, verbose=True))
#     ])

#     param_grid = {
#         'mlp__hidden_layer_sizes': [(150, 150, 150), (200, 200, 200)],
#         'mlp__alpha': [0.0001, 0.0005],
#         'mlp__activation': ['relu'],
#         'mlp__learning_rate_init': [0.0005, 0.001],
#         'mlp__solver': ['adam']
#     }

#     kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)
#     grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)

#     # Start the timer thread
#     keep_running = {'active': True}
#     start_time = time.time()
#     timer_thread = threading.Thread(target=print_time_elapsed, args=(start_time, keep_running))
#     timer_thread.start()

#     # Begin model training
#     grid_search.fit(X_train, Y_train)

#     # Stop the timer thread
#     keep_running['active'] = False
#     timer_thread.join()

#     # Evaluation and output
#     Y_pred = grid_search.best_estimator_.predict(X_test)
#     print("Best Model Parameters:", grid_search.best_params_)
#     print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
#     print("Test Set Accuracy:", accuracy_score(Y_test, Y_pred))
#     print("Test Set Precision:", precision_score(Y_test, Y_pred, average='macro'))
#     print("Test Set Recall:", recall_score(Y_test, Y_pred, average='macro'))
#     print("Classification Report:\n", classification_report(Y_test, Y_pred))
#     print(f"Total Time taken: {time.time() - start_time:.2f} seconds")

# # Example usage with your dataset
# # Assuming X and Y are defined as your dataset features and target
# MLP_Classifier(X, Y)

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
# from sklearn.decomposition import PCA
# from timeit import default_timer as timer

# def MLP_Classifier(X, y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=109)

#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('features', PolynomialFeatures(degree=2)),
#         ('pca', PCA(n_components=0.95)),  # Reduce dimensions while retaining 95% variance
#         ('mlp', MLPClassifier(max_iter=500, random_state=21, verbose=True))
#     ])

#     param_grid = {
#         'mlp__hidden_layer_sizes': [(200, 200, 200), (250, 200, 150)],
#         'mlp__alpha': [0.0001, 0.0005],
#         'mlp__activation': ['relu'],
#         'mlp__learning_rate_init': [0.0001, 0.0005],
#         'mlp__solver': ['adam']
#     }

#     kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)
#     grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
#     start_time = timer()
#     grid_search.fit(X_train, Y_train)
#     print(f"Total Time taken: {timer() - start_time:.2f} seconds")

#     Y_pred = grid_search.best_estimator_.predict(X_test)
#     print("Best Model Parameters:", grid_search.best_params_)
#     print("Test Set Accuracy:", accuracy_score(Y_test, Y_pred))
#     print("Test Set Precision:", precision_score(Y_test, Y_pred, average='macro'))
#     print("Test Set Recall:", recall_score(Y_test, Y_pred, average='macro'))
#     print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Assuming X and Y are defined as your dataset features and target
# MLP_Classifier(X, Y)


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