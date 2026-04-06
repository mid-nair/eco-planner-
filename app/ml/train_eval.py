from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.config import MLConfig
from app.ml.models import build_model


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> Tuple[object, Dict[str, float]]:
    """Train a regression model and return fitted model plus performance metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MLConfig.test_size, random_state=MLConfig.random_state
    )

    model = build_model(model_name, random_state=MLConfig.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }
    return model, metrics


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: List[str],
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Train/evaluate multiple models across repeated train/test splits.
    Returns a DataFrame sorted by best mean R² (descending).
    """
    split_seeds = [MLConfig.random_state + (i * 7) for i in range(max(1, n_repeats))]
    rows = []
    for name in model_names:
        r2_vals: List[float] = []
        mae_vals: List[float] = []
        rmse_vals: List[float] = []

        for seed in split_seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=MLConfig.test_size, random_state=seed
            )
            model = build_model(name, random_state=MLConfig.random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_vals.append(float(r2_score(y_test, y_pred)))
            mae_vals.append(float(mean_absolute_error(y_test, y_pred)))
            rmse_vals.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))

        rows.append(
            {
                "model": name,
                "r2_mean": float(np.mean(r2_vals)),
                "r2_std": float(np.std(r2_vals)),
                "mae_mean": float(np.mean(mae_vals)),
                "rmse_mean": float(np.mean(rmse_vals)),
            }
        )

    df = pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


