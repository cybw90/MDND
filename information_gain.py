from __future__ import print_function
import csv
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as num
from sklearn import datasets, linear_model
#from genetic_selection import GeneticSelectionCV

data = pd.read_csv(r"Extracted_Features.csv")

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

#Information gain

importance = mutual_info_classif(X,Y)
feat_importance = pd.Series(importance, data.columns[0:len(data.columns)-2])
print(feat_importance)
feat_importance.plot(kind= 'barh', color = 'teal')
plt.show()
plt.figure(figsize=(10,6))