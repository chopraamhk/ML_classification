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
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import cross_val_predict

print("\nRunning Randomforest on 157 samples")

#all samples loaded
df = pd.read_csv('/data2/mchopra/yale_rnaseq/rnaseq_outputs/157_all_samples_tpm.tsv', sep = "\t", index_col=0)

#all labels
labels = pd.read_csv("/data2/mchopra/yale_rnaseq/rnaseq_outputs/metadata_157_all.csv",  sep = ",", index_col=0)

print("\nRemoving genes with 0 value in count matrix")

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

print("\nX = 26147 features (genes)")
print("\ny = 157 cases/controls")

X = merged_df.iloc[:,1:26147]
y = merged_df.iloc[:, 26148]


print("\nsplitting the samples to 80:20 ratio (training:testing)")

#no validation set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#checking split of training and test dataset
from collections import Counter

print("\nTraining set:", Counter(y_train))
print("\nTesting set:", Counter(y_test))


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
print("\n FIRST CHECK")
print(f"test Accuracy (first check - without hypertuning): {test_accuracy:.4f}")
print("\nClassification Report (first check):\n", classification_rep)
print("\nConfusion Matrix (first check):\n", confusion_matrix(y_test, y_pred))
print("\n FINISHED FIRST CHECK")

#cross validation without hypertuning

print("\nInitiating cross validation below")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#y_enc = LabelEncoder().fit.transform(y)

scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=cv,
    scoring='accuracy'
)

print("\nCross validation (CV) Accuracy (Second check - without hypertuning):", scores.mean())
print("\nStd Dev in cross-validation (Second check - without hypertuning):", scores.std())


# assume X_train, y_train already loaded then above code would not be required

rf_model = RandomForestClassifier(random_state=42)

print("\nhypertuning model initiating")

param_grid = {
    'n_estimators': [10, 100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'max_features': ["sqrt", "log2", 0.5],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False],
    'class_weight': ['balanced']
}

print("\nhypertuning done")

print("\ntrying every combination in param_grid and for each combination - splitting the traiing data into 5 folds, training on 4 folds and validating on 1 fold, repeating 5 times and providing average performance")

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

print("\nBest parameters: ", grid_search.best_params_)
print("\nBest score: ", grid_search.best_score_)

print("\nTest results with CV and best grid tuned parameters")

print("\nTest Accuracy with best grid tuned parameters:", accuracy_score(y_test, y_pred))

print("\nClassification Report for test accuracy with best tuned parameters:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix for test accuracy with best tuned parameters:\n", confusion_matrix(y_test, y_pred))

joblib.dump(best_model, "rf_model.pkl")
joblib.dump(grid_search, "rf_grid_search.pkl")

print("\nFinished hyperparameter tuning and saving models - rf_model.pkl (best_model) and rf_grid_search.pkl - grid search model")

print("\nFinished cross validation")

print("\nInitiating nested cross validation")
#nested cross validation 
#below is for nested cross-validation 

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

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

print("\nNested CV estimates how good your modeling strategy is.")

print("\nNested CV Accuracy:", nested_scores.mean())
print("\nStd Dev:", nested_scores.std())

print("\npipeline ran successfully")
