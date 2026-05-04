#score = balanced_accuracy for imbalancedness and accuracy for other times
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
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import balanced_accuracy_score


print("\nRunning Randomforest on 118 samples")

#all samples loaded
df = pd.read_csv('salmon_gene_tpm_qc_passed.txt', sep = "\t", index_col=0)

#all labels
labels = pd.read_csv("metadata_only_qc_passed.txt",  sep = ",", index_col=0)

print("\nRemoving genes with 0 value in count matrix")

#Remove genes that are 0 in all samples
#a bit of cleaning below
keep_nonzero = (df > 0).sum(axis=1) > 0
df = df.loc[keep_nonzero]

# Keep genes expressed in at least 20% of samples with a value > 0
threshold = 1
min_percent = 0.20
n_samples = df.shape[1]

# Logic: Sum how many samples are > 0, then check if that sum is >= 20% of total samples
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

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nsplitting the samples to 80:20 ratio (training:testing)")

#no validation set 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

#checking split of training and test dataset

print("\nTraining set:", Counter(y_train))
print("\nTesting set:", Counter(y_test))

# On training data
#rv = X_train.var(axis=0)
#q75 = rv.quantile(0.75)
#selected_genes = X_train.columns[rv >= q75]

# Apply SAME genes to BOTH train and test
X_train = X_train[selected_genes]
X_test = X_test.reindex(columns=selected_genes, fill_value=0)

# Summary of variances in the final set
print(X_train.var(axis=0).describe())

#first check without any hypertuning and cross-validation 
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")

# Now train and predict
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

#For imbalanced gene-expression classification, 
#using class_weight='balanced' with Random Forest is usually a good default. 
#It often improves recall for the minority class with minimal extra tuning.
#gives more weightage to minority class samples 

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

# pipeline: it's safer for future additions.
pipeline = Pipeline([
    ('select', SelectPercentile(f_classif, percentile=25)),  # ~top 25% by ANOVA F-test
 #  ('var', VarianceThreshold(threshold=q75)),
    ('rf', RandomForestClassifier(random_state=42, class_weight="balanced"))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#y_enc = LabelEncoder().fit.transform(y)

scores = cross_val_score(
    pipeline,
    X,
    y_encoded,
    cv=cv,
    scoring='balanced_accuracy'
)

#balanced_accuracy avoids inflated performance estimates on imbalanced datasets

print("\nCross validation (CV) Accuracy (Second check - without hypertuning):", scores.mean())
print("\nStd Dev in cross-validation (Second check - without hypertuning):", scores.std())

#rf_model = RandomForestClassifier(random_state=42)

print("\nhypertuning model initiating")

param_grid = {
    'select__percentile': [20, 25, 45, 50, 70, 80]  
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
    scoring='balanced_accuracy',
    n_jobs=2
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters: ", grid_search.best_params_)

y_pred_tuned = best_model.predict(X_test)

print("\nBest score: ", grid_search.best_score_)

print("\nTest results with CV and best grid tuned parameters")

print("\nTest Balanced Accuracy with best grid tuned parameters:", balanced_accuracy_score(y_test, y_pred_tuned))

print("\nClassification Report for test accuracy with best tuned parameters:\n", classification_report(y_test, y_pred_tuned))

print("\nConfusion Matrix for test accuracy with best tuned parameters:\n", confusion_matrix(y_test, y_pred_tuned))

joblib.dump(best_model, "rf_pipeline_model.pkl")
joblib.dump(grid_search, "rf_grid_search_pipeline.pkl")

print("\nFinished hyperparameter tuning and saving models - rf_model_pipeline.pkl (best_model) and rf_grid_search_pipeline.pkl - grid search model")

print("\nFinished cross validation")

print("\nInitiating nested cross validation")
#nested cross-validation 
#below is for nested cross-validation 

#rf_model = RandomForestClassifier(random_state=42)

# INNER CV (for tuning)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_gs = GridSearchCV(
    pipeline,
    param_grid,
    cv=inner_cv,
    scoring='balanced_accuracy',
    n_jobs=-1
)

# OUTER CV (for evaluation)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_scores = cross_val_score(
    nested_gs,
    X,
    y_encoded,
    cv=outer_cv,
    scoring="balanced_accuracy"
)

print("\nNested CV estimates how good your modeling strategy is.")

print("\nNested CV Accuracy:", nested_scores.mean())
print("\nStd Dev:", nested_scores.std())

print("\npipeline ran successfully")
