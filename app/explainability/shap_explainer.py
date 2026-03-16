from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model: Any,
    X_sample: pd.DataFrame,
    max_samples: int = 200,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Compute SHAP values and global feature importance for a fitted model.

    For tree-based models SHAP uses TreeExplainer. For other models where
    SHAP is problematic, we gracefully fall back to simple importance proxies.
    """
    X_sample = X_sample.copy()
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=0)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # Fallback: use model-native importances if available, otherwise zeros.
        if hasattr(model, "feature_importances_"):
            importance = np.abs(np.asarray(model.feature_importances_))
            # Broadcast to match expected return format (samples x features)
            shap_values_arr = np.zeros((len(X_sample), len(X_sample.columns)))
            feature_importance = pd.Series(
                importance, index=X_sample.columns
            ).sort_values(ascending=False)
            return feature_importance, shap_values_arr

        if hasattr(model, "coef_"):
            coef = np.asarray(getattr(model, "coef_"))
            if coef.ndim > 1:
                coef = coef[0]
            importance = np.abs(coef)
            shap_values_arr = np.zeros((len(X_sample), len(X_sample.columns)))
            feature_importance = pd.Series(
                importance, index=X_sample.columns
            ).sort_values(ascending=False)
            return feature_importance, shap_values_arr

        # Last-resort: zero importance
        shap_values_arr = np.zeros((len(X_sample), len(X_sample.columns)))
        feature_importance = pd.Series(
            0.0, index=X_sample.columns
        ).sort_values(ascending=False)
        return feature_importance, shap_values_arr

    if isinstance(shap_values, list):
        shap_values_arr = np.array(shap_values[0])
    else:
        shap_values_arr = np.array(shap_values)

    importance = np.abs(shap_values_arr).mean(axis=0)
    feature_importance = pd.Series(importance, index=X_sample.columns).sort_values(ascending=False)

    return feature_importance, shap_values_arr


