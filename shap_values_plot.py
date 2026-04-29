import shap
import matplotlib.pyplot as plt
import joblib

# Load GridSearchCV
grid = joblib.load("118_samples/xgb_grid_search_pipeline.pkl")

# Extract pipeline
pipeline = grid.best_estimator_

# Extract XGBoost model (CHECK STEP NAME)
xgb_model = pipeline.named_steps["xgb"]
explainer = shap.TreeExplainer(xgb_model)
X_shap = X

shap_values = explainer.shap_values(X_shap)
shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    max_display=20
)
