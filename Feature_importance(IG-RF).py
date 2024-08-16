from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv(r"50_Features.csv")

# handling missing values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Check for and remove constant features
n_unique = data.apply(pd.Series.nunique)
cols_to_drop = n_unique[n_unique == 1].index
data.drop(cols_to_drop, axis=1, inplace=True)


data = data.drop(['URLs', 'Type'], axis=1)
data = data.reset_index(drop=True)

# Extract features and labels
X = data.iloc[:, 2:].astype(np.float32)  
Y = data['Label']

# Calculate mutual information 
mi_importance = mutual_info_classif(X, Y)
mi_feat_importance = pd.Series(mi_importance, index=X.columns)

# Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, Y)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

# Combine both feature importances into a DataFrame
importance_df = pd.DataFrame({
    'Mutual Information': mi_feat_importance,
    'Random Forest': rf_importance
})

# Plotting
plt.figure(figsize=(20, 10))
importance_df.plot(kind='bar', width=0.8, figsize=(20, 10))
plt.ylabel('Importance')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

important_features_mi = mi_feat_importance[mi_feat_importance >= 0.04]
print("Features with MI importance >= 0.04:")
print(important_features_mi)

important_features_rf = rf_importance.sort_values(ascending=False).head(10)
print("Features importance with Random Forest:")
print(important_features_rf)
