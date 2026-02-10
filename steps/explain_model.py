import numpy as np
import pandas as pd
import shap
from zenml import step

@step
def explain_model(model, X_train: pd.DataFrame, top_k: int = 10):
    """
    Returns a list of top features by mean(|SHAP|).
    Works well for LogisticRegression by explaining predict_proba.
    """
    # explain probabilities (more standard for classification)
    explainer = shap.Explainer(model.predict_proba, X_train)
    shap_values = explainer(X_train)

    # shap_values.values shape: (n_samples, n_features, n_classes)
    vals = shap_values.values
    if vals.ndim == 3:
        # take class 1 if binary, else average across classes
        if vals.shape[2] >= 2:
            vals = vals[:, :, 1]
        else:
            vals = vals.mean(axis=2)

    mean_abs = np.abs(vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    return [
        {"feature": X_train.columns[i], "importance": float(mean_abs[i])}
        for i in top_idx
    ]
