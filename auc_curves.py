import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load model
best_model = joblib.load("118_samples/xgb_model_pipeline.pkl")

# Reload SAME test data
#X_test = pd.read_csv("X_test.csv", index_col=0)
#y_test = pd.read_csv("y_test.csv", index_col=0).values.ravel()

# Probabilities
y_prob = best_model.predict_proba(X_test)[:, 1]

# AUC
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()
