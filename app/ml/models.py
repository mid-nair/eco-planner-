from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from app.ml.ensemble import BWFERegressor


@dataclass
class ModelSpec:
    name: str
    params: Dict[str, Any]


def build_model(model_name: str, random_state: int = 42) -> Any:
    """Factory for supported regression models."""
    name = model_name.lower()
    if name == "random forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "ann":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=2500,
            tol=1e-3,
            early_stopping=False,
            random_state=random_state,
        )
    if name == "arima-bpnn":
        # Approximated ARIMA-BPNN: use a deep MLP as a backpropagation neural network
        # keeping the same regression interface as other models.
        return MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=3000,
            tol=1e-3,
            early_stopping=False,
            random_state=random_state,
        )
    if name == "svr":
        return SVR(kernel="rbf", C=10.0, epsilon=0.1)
    if name == "bwfe":
        # Bayesian-Weighted Full Ensemble combining XGBoost, RF, ANN, and ARIMA-BPNN
        return BWFERegressor(random_state=random_state)


    raise ValueError(f"Unsupported model: {model_name}")


