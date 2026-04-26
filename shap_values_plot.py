import shap
import matplotlib.pyplot as plt
import joblib

# Load GridSearchCV
grid = joblib.load("118_samples/xgb_grid_search_pipeline.pkl")

# Extract pipeline
pipeline = grid.best_estimator_

# Extract XGBoost model (CHECK STEP NAME)
xgb_model = pipeline.named_steps["xgb"]

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# Use held-out test data or representative subset
X_shap = X

# Compute SHAP values
shap_values = explainer.shap_values(X_shap)

# Global importance plot (SAFE choice)
shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    max_display=20
)
