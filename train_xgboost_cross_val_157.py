import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV, cross_val_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =============================
# Load data

df = pd.read_csv('157_all_samples_tpm.tsv', sep="\t", index_col=0)

labels = pd.read_csv("metadata_157_all.csv", sep=",", index_col=0)

# Remove genes with all zero expression
keep_nonzero = (df > 0).sum(axis=1) > 0
df = df.loc[keep_nonzero]

# Transpose: samples as rows
df_T = df.T.reset_index().rename(columns={'index': 'Samples'})
labels = labels.reset_index().rename(columns={'index': 'Samples'})

# Merge expression + metadata
merged_df = pd.merge(df_T, labels, on='Samples', how='inner')

# Define X and y
X = merged_df.iloc[:, 1:26147]
y = merged_df.iloc[:, 26148]

# Encode labels FIRST (important)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# =============================
# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set:", Counter(y_train))
print("Testing set:", Counter(y_test))

# Handle class imbalance for XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# =============================
# Baseline XGBoost (no tuning)

xgb_model = XGBClassifer(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

print(f"\nTest Accuracy (baseline XGBoost): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report (baseline):\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix (baseline):\n", confusion_matrix(y_test, y_pred))

# =============================
# Cross-validation (no tuning)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(xgb_model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print("\nCV Accuracy (no tuning):", scores.mean())
print("CV Std Dev:", scores.std())

# =============================
# Hyperparameter tuning (GridSearch)

print("\nStarting hyperparameter tuning")

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1, 10, 100],
    'reg_lambda': [0.5, 0.7, 1, 1.3]
}

xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1)

grid_search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nTest Accuracy (tuned XGBoost):", accuracy_score(y_test, y_pred))
print("\nClassification Report (tuned):\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix (tuned):\n", confusion_matrix(y_test, y_pred))

joblib.dump(best_model, "xgb_model.pkl")
joblib.dump(grid_search, "xgb_grid_search.pkl")

print("\nFinished hyperparameter tuning and saving models - xgb_model.pkl (best_model) and xgb_grid_search.pkl - grid search model")

# =============================
# Nested cross-validation

print("\nInitiating nested cross-validation")

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1)

grid_search = GridSearchCV(xgb_model, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)

nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, n_jobs=-1)

np.save("nested_cv_accuracy_scores_xgboost.npy", nested_scores)
print("\nNested CV estimates how good your modeling strategy is.")
print("\nNested CV Accuracy:", nested_scores.mean())
print("\nNested CV Std Dev:", nested_scores.std())

print("\nPipeline ran successfully")
