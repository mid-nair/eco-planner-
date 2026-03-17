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
) -> pd.DataFrame:
    """
    Train/evaluate multiple models on the exact same split.
    Returns a DataFrame sorted by best R² (descending).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MLConfig.test_size, random_state=MLConfig.random_state
    )

    rows = []
    for name in model_names:
        model = build_model(name, random_state=MLConfig.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append(
            {
                "model": name,
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }
        )

    df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


