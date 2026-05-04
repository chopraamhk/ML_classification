import pandas as pd
import numpy as np
import joblib
import xgboost 
from collections import Counter
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV, cross_val_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import balanced_accuracy_score

# Load data
df = pd.read_csv('salmon_gene_tpm_qc_passed.txt', sep="\t", index_col=0)

labels = pd.read_csv("metadata_only_qc_passed.txt", sep=",", index_col=0)

# Remove genes with all zero expression
keep_nonzero = (df > 0).sum(axis=1) > 0
df = df.loc[keep_nonzero]

#Keep genes expressed in at least 20% of samples with a value > 0
threshold = 1
min_percent = 0.20
n_samples = df.shape[1]

# Logic: Sum how many samples are > 1, then check if that sum is >= 20% of total samples
keep_expressed = (df > threshold).sum(axis=1) > (min_percent * n_samples)
df_filtered = df.loc[keep_expressed]

print(f"Original genes: {len(keep_nonzero)}")
print(f"Genes after 20% threshold filter: {len(df_filtered)}")

#Apply Log2 transformation (Standard for gene expression to stabilise variance)
df_log = np.log2(df_filtered + 1)

print(f"Genes after log2 transformation: {len(df_log)}")

#transpose so samples are rows and genes are columns 
df_T = df_log.T
df_T.index.name = "Samples"

# if your samples are the index after transpose
df_T = df_T.reset_index().rename(columns={'index': 'Samples'})

labels = labels.reset_index().rename(columns={'index': 'Samples'})

merged_df = pd.merge(df_T, labels, on='Samples', how='inner')

print(f"\nX = {len(df_log)} features (genes)")
print(f"\ny = {len(labels)} cases/controls")

X = merged_df.iloc[:, 1:len(df_log) + 1]

y = merged_df.iloc[:, len(df_log) + 2]

#----------------------------------
#XGBoost can be picky about gene names (special characters)
X.columns = [str(c).replace('[', '').replace(']', '').replace('<', '') for c in X.columns]

# Encode labels FIRST (important)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split

print("\nsplitting the samples to 80:20 ratio (training:testing)")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Handle class imbalance for XGBoost
#Dynamic scale_pos_weight (Calculate based on the current training split)
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

print("\nTraining set:", Counter(y_train))
print("Testing set:", Counter(y_test))

#----------------------
# On training data
#rv = X_train.var(axis=0)
#q75 = rv.quantile(0.75)
#selected_genes = X_train.columns[rv >= q75]

# Apply SAME genes to BOTH train and test
#X_train = X_train[selected_genes]
#X_test = X_test.reindex(columns=selected_genes, fill_value=0)

#variances in final set
#print(X_train.var(axis=0).describe())

#------------------------------

# Baseline XGBoost (no tuning)

pipeline = Pipeline([
    ('select', SelectPercentile(f_classif, percentile=25)),  # Top 25% by case/control separation
    ('xgb', XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1))
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\n FIRST CHECK")
print(f"\nTest Accuracy (baseline XGBoost): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report (baseline):\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix (baseline):\n", confusion_matrix(y_test, y_pred))
print("\n FINISHED FIRST CHECK")

# Cross-validation (no tuning)

print("\nInitiating cross validation below")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring="balanced_accuracy", n_jobs=2)
#balanced_accuracy avoids inflated performance estimates on imbalanced datasets

print("\nCross validation (CV) Accuracy (Second check - without hypertuning):", scores.mean())
print("\nStd Dev in cross-validation (Second check - without hypertuning):", scores.std())

# Hyperparameter tuning (GridSearch)

print("\nStarting hyperparameter tuning")

param_grid = {
     'select__percentile': [20, 25, 45, 50]  
    'xgb__n_estimators': [100, 300],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__max_depth': [3, 6],
    'xgb__subsample': [0.7, 1.0],
    'xgb__colsample_bytree': [0.5, 1.0]
}

print("\ntrying every combination in param_grid and for each combination - splitting the traiing data into 5 folds, training on 4 folds and validating on 1 fold, repeating 5 times and providing average performance")

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=2, verbose=1)

grid_search.fit(X_train, y_train)

print("\nhypertuning done")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nBest parameters: ", grid_search.best_params_)
print("\nBest score: ", grid_search.best_score_)

print("\nTest results with CV and best grid tuned parameters")

print("\nTest Balance Accuracy with best grid tuned parameters:", accuracy_score_score(y_test, y_pred))

print("\nClassification Report for test accuracy with best tuned parameters:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix for test accuracy with best tuned parameters:\n", confusion_matrix(y_test, y_pred))

importances = best_model.named_steps['xgb'].feature_importances_
top_genes = X_train.columns[np.argsort(importances)[-20:]].tolist()
print("Top 20 genes:", top_genes)
joblib.dump(top_genes, "top_xgb_genes.pkl")

joblib.dump(best_model, "xgb_model_pipeline.pkl")
joblib.dump(grid_search, "xgb_grid_search_pipeline.pkl")

print("\nFinished hyperparameter tuning and saving models - xgb_model_pipeline.pkl (best_model) and xgb_grid_search_pipeline.pkl - grid search model")

# Nested cross-validation

print("\nInitiating nested cross-validation")

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search_nested = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="balanced_accuracy", n_jobs=2)

nested_scores = cross_val_score(grid_search_nested, X, y_encoded, cv=outer_cv, n_jobs=2)

np.save("nested_cv_accuracy_scores_xgboost_pipeline.npy", nested_scores)
print("\nNested CV estimates how good your modeling strategy is.")
print("\nNested CV Accuracy:", nested_scores.mean())
print("\nNested CV Std Dev:", nested_scores.std())

print("\nPipeline ran successfully")
