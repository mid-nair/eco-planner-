from __future__ import annotations

from typing import Dict, Tuple

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


