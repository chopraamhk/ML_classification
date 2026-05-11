import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Load fitted pipeline/model
best_rf = joblib.load(
    "filtering_manuscript_version/118_samples/rf_pipeline_model.pkl"
)

# CV setup
cv_outer = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

tprs = []
precisions = []
aucs = []
aps = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for i, (train_idx, test_idx) in enumerate(cv_outer.split(X, y_encoded)):

    # Clone model to avoid contamination
    from sklearn.base import clone
    model = clone(best_rf)

    # Train
    model.fit(
        X.iloc[train_idx],
        y_encoded[train_idx]
    )

    # Predict probabilities
    probas_ = model.predict_proba(
        X.iloc[test_idx]
    )[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(
        y_encoded[test_idx],
        probas_
    )

    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    ax1.plot(
        fpr,
        tpr,
        lw=1,
        alpha=0.3,
        label=f'Fold {i+1} (AUC={roc_auc:.2f})'
    )

    # PR
    precision, recall, _ = precision_recall_curve(
        y_encoded[test_idx],
        probas_
    )

    ap = average_precision_score(
        y_encoded[test_idx],
        probas_
    )

    aps.append(ap)

    interp_precision = np.interp(
        mean_recall,
        recall[::-1],
        precision[::-1]
    )

    precisions.append(interp_precision)

    ax2.plot(
        recall,
        precision,
        lw=1,
        alpha=0.3,
        label=f'Fold {i+1} (AP={ap:.2f})'
    )

# Mean ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0

mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

ax1.plot(
    mean_fpr,
    mean_tpr,
    lw=2,
    label=f'Mean ROC (AUC={mean_auc:.2f} ± {std_auc:.2f})'
)

std_tpr = np.std(tprs, axis=0)

ax1.fill_between(
    mean_fpr,
    np.maximum(mean_tpr - std_tpr, 0),
    np.minimum(mean_tpr + std_tpr, 1),
    alpha=0.2
)

ax1.plot([0, 1], [0, 1], linestyle='--')

ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("Cross-Validated ROC")
ax1.legend()

# Mean PR
mean_precision = np.mean(precisions, axis=0)

mean_ap = np.mean(aps)
std_ap = np.std(aps)

ax2.plot(
    mean_recall,
    mean_precision,
    lw=2,
    label=f'Mean PR (AP={mean_ap:.2f} ± {std_ap:.2f})'
)

std_precision = np.std(precisions, axis=0)

ax2.fill_between(
    mean_recall,
    np.maximum(mean_precision - std_precision, 0),
    np.minimum(mean_precision + std_precision, 1),
    alpha=0.2
)

ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Cross-Validated Precision-Recall")
ax2.legend()

plt.tight_layout()
plt.show()
