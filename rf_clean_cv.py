#below was for 118 samples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
from collections import Counter
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
from sklearn.pipeline import Pipeline

print("\nRunning Randomforest on 118 samples")

#all samples loaded
df = pd.read_csv('salmon_gene_tpm_qc_passed.txt', sep = "\t", index_col=0)

#all labels
labels = pd.read_csv("metadata_only_qc_passed.txt",  sep = ",", index_col=0)

print("\nRemoving genes with 0 value in count matrix")

#a bit of cleaning below
keep_nonzero = (df > 0).sum(axis=1) > 0
df = df.loc[keep_nonzero]

#Apply Log2 transformation (Standard for gene expresson to stabilize variance)
df_log = np.log2(df + 1)


#transpose
df_T = df_log.T

df_T.index.name = "Samples"

# if your samples are the index after transpose
df_T = df_T.reset_index().rename(columns={'index': 'Samples'})

labels = labels.reset_index().rename(columns={'index': 'Samples'})

merged_df = pd.merge(df_T, labels, on='Samples', how='inner')

print("\nX = 25765 features (genes)")
print("\ny = 118 cases/controls")

X = merged_df.iloc[:,1:25765]
y = merged_df.iloc[:, 25765]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nsplitting the samples to 80:20 ratio (training:testing)")

#no validation set 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

#checking split of training and test dataset

print("\nTraining set:", Counter(y_train))
print("\nTesting set:", Counter(y_test))

#this is just encoding characters to numerals like case/control - 1/0.
#label_encoder = LabelEncoder()
#y_train = label_encoder.fit_transform(y_train)
#y_test = label_encoder.transform(y_test)

#first check without any hypertuning and cross validation 
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")

#For imbalanced gene-expression classification, 
#using class_weight='balanced' with Random Forest is usually a good default. 
#It often improves recall for the minority class with minimal extra tuning.rf_model.fit(X_train, y_train)
#gives more weightage to minority class samples 

y_pred = rf_model.predict(X_test)

print("\n--- FIRST CHECK (Without Tuning) ---")

test_accuracy = accuracy_score(y_test, y_pred)

classification_rep = classification_report(y_test, y_pred)
print("\n FIRST CHECK")
print(f"test Accuracy (first check - without hypertuning): {test_accuracy:.4f}")
print("\nClassification Report (first check):\n", classification_rep)
print("\nConfusion Matrix (first check):\n", confusion_matrix(y_test, y_pred))
print("\n FINISHED FIRST CHECK")

# cross-validation without hypertuning

print("\nInitiating cross validation below")


# Use a Pipeline. Even if you don't scale, it's safer for future additions.
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Keeps scaling logic inside CV folds
    ('rf', RandomForestClassifier(random_state=42, class_weight="balanced"))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#y_enc = LabelEncoder().fit.transform(y)

scores = cross_val_score(
    pipeline,
    X,
    y_encoded,
    cv=cv,
    scoring='accuracy'
)

print("\nCross validation (CV) Accuracy (Second check - without hypertuning):", scores.mean())
print("\nStd Dev in cross-validation (Second check - without hypertuning):", scores.std())


# assume X_train, y_train already loaded then above code would not be required

#rf_model = RandomForestClassifier(random_state=42)

print("\nhypertuning model initiating")

param_grid = {
    'rf__n_estimators': [100, 300, 500],
    'rf__max_depth': [5, 10, 20],
    'rf__max_features': ["sqrt", "log2"],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__bootstrap': [True],
    'rf__class_weight': ['balanced']
}

print("\nhypertuning done")

print("\ntrying every combination in param_grid and for each combination - splitting the traiing data into 5 folds, training on 4 folds and validating on 1 fold, repeating 5 times and providing average performance")

#cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=2
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters: ", grid_search.best_params_)

y_pred_tuned = best_model.predict(X_test)

print("\nBest score: ", grid_search.best_score_)

print("\nTest results with CV and best grid tuned parameters")

print("\nTest Accuracy with best grid tuned parameters:", accuracy_score(y_test, y_pred_tuned))

print("\nClassification Report for test accuracy with best tuned parameters:\n", classification_report(y_test, y_pred_tuned))

print("\nConfusion Matrix for test accuracy with best tuned parameters:\n", confusion_matrix(y_test, y_pred_tuned))

joblib.dump(best_model, "rf_pipeline_model.pkl")
joblib.dump(grid_search, "rf_grid_search_pipeline.pkl")

print("\nFinished hyperparameter tuning and saving models - rf_model_pipeline.pkl (best_model) and rf_grid_search_pipeline.pkl - grid search model")

print("\nFinished cross validation")

print("\nInitiating nested cross validation")
#nested cross-validation 
#below is for nested cross-validation 

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

#rf_model = RandomForestClassifier(random_state=42)

# INNER CV (for tuning)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_gs = GridSearchCV(
    pipeline,
    param_grid,
    cv=inner_cv,
    scoring='accuracy',
    n_jobs=-1
)

# OUTER CV (for evaluation)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_scores = cross_val_score(
    nested_gs,
    X,
    y_encoded,
    cv=outer_cv
)

print("\nNested CV estimates how good your modeling strategy is.")

print("\nNested CV Accuracy:", nested_scores.mean())
print("\nStd Dev:", nested_scores.std())

print("\npipeline ran successfully")
