import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

#all samples loaded
df = pd.read_csv('157_all_samples_tpm.tsv', sep = "\t", index_col=0)

#all labels
labels = pd.read_csv("metadata_157_all.csv",  sep = ",", index_col=0)

#a bit of cleaning below
keep_nonzero = (df > 0).sum(axis=1) > 0
df = df.loc[keep_nonzero]

#trasverse
df_T = df.T
df_T.index.name = "Samples"

# if your samples are the index after transpose
df_T = df_T.reset_index().rename(columns={'index': 'Samples'})

labels = labels.reset_index().rename(columns={'index': 'Samples'})

merged_df = pd.merge(df_T, labels, on='Samples', how='inner')

X = merged_df.iloc[:,1:26147]

y = merged_df.iloc[:, 26148]

#no validation set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#checking split of training and test dataset
from collections import Counter

print("Training set:", Counter(y_train))
print("Testing set:", Counter(y_test))


#this is just encoding characters to numerals like case/control - 1/0.
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

#first check without any hypertuning and cross validation 
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

classification_rep = classification_report(y_test, y_pred)

print(f"test Accuracy (first check - without hypertuning): {test_accuracy:.4f}")

print("\nClassification Report (first check):\n", classification_rep)

print("\nConfusion Matrix (first check):\n", confusion_matrix(y_test, y_pred))

#cross validation without hypertuning

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=cv,
    scoring='accuracy'
)

print("Cross validation (CV) Accuracy (Second check - without hypertuning):", scores.mean())
print("Std Dev in cross-validation (Second check - without hypertuning):", scores.std())


# assume X_train, y_train already loaded then above code would not be required

rf_model = RandomForestClassifier(random_state=42)

print("\nhypertuning model")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'max_features': ["sqrt", "log2"],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True,False],
    'class_weight': [None, 'balanced']
}

print("\nhypertuning done")
print("trying every combination in param_grid and for each combination - splitting the traiing data into 5 folds, training on 4 folds and validating on 1 fold, repeating 5 times and providing average performance")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nTest results with CV and best tuned parameters")
print("Test Accuracy with best tuned parameters:", accuracy_score(y_test, y_pred))

print("\nClassification Report for test accuracy with best tuned parameters:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix for test accuracy with best tuned parameters:\n", confusion_matrix(y_test, y_pred))

joblib.dump(best_model, "rf_model.pkl")
joblib.dump(grid_search, "grid_search.pkl")


print("\nFinished cross validation")

print("\nInitiating nested cross validation")
#nested cross validation 
#below is for nested cross-validation 

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'max_features': ["sqrt", "log2"],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True],
    'class_weight': [None, "balanced"]
}

# INNER CV (for tuning)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=inner_cv,
    scoring='accuracy',
    n_jobs=-1
)

# OUTER CV (for evaluation)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_scores = cross_val_score(
    grid_search,
    X,
    y,
    cv=outer_cv
)

print("Nested CV Accuracy:", nested_scores.mean())
print("Std Dev:", nested_scores.std())


print("pipeline ran successfully")
