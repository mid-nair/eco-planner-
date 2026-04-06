from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


class BWFERegressor(BaseEstimator, RegressorMixin):
    """
    Strong BWFE via stacked generalization:
    base estimators (XGB, RF, ANN, ARIMA-BPNN) + RidgeCV meta-learner.
    `passthrough=True` lets the meta-learner use both base predictions and
    original features, which generally improves robustness and performance.
    """

    def __init__(self, random_state: int = 42, n_splits: int = 5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.model_: Optional[StackingRegressor] = None
        self.weights_: Optional[np.ndarray] = None

    def _build_stacker(self) -> StackingRegressor:
        estimators = [
            (
                "xgboost",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=2500,
                    tol=1e-3,
                    early_stopping=False,
                    random_state=self.random_state,
                ),
            ),
            (
                "arima_bpnn",
                MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=3000,
                    tol=1e-3,
                    early_stopping=False,
                    random_state=self.random_state,
                ),
            ),
        ]

        meta = RidgeCV(alphas=np.logspace(-4, 4, 40), fit_intercept=True)
        return StackingRegressor(
            estimators=estimators,
            final_estimator=meta,
            cv=self.n_splits,
            passthrough=True,
            n_jobs=-1,
        )

    def _as_train_xy(
        self, X: Union[pd.DataFrame, np.ndarray], y: Any
    ) -> tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        y_arr = np.asarray(y, dtype=float).ravel()
        return X, y_arr

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any) -> "BWFERegressor":
        X_loc, y_arr = self._as_train_xy(X, y)
        self.model_ = self._build_stacker()
        self.model_.fit(X_loc, y_arr)

        # Expose first four meta coefficients (base prediction weights) for inspection.
        final_est = getattr(self.model_, "final_estimator_", None)
        coef = getattr(final_est, "coef_", None)
        if coef is not None and len(np.asarray(coef).ravel()) >= 4:
            self.weights_ = np.asarray(coef).ravel()[:4]
        else:
            self.weights_ = None
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("BWFERegressor is not fitted.")
        return self.model_.predict(X).astype(float)
