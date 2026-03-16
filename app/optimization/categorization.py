from __future__ import annotations

import numpy as np
import pandas as pd


def categorize_plans(plans: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each plan as one of:
    - Low carbon plan
    - Low cost plan
    - High strength plan
    - Balanced plan
    """
    df = plans.copy()

    # Normalize core metrics
    def _z(col: str) -> np.ndarray:
        v = df[col].astype(float).values
        return (v - v.mean()) / (v.std() + 1e-9)

    z_carbon = _z("predicted_total_co2")
    z_cost = _z("total_cost")
    z_strength = _z("material_strength_score")

    labels = []
    for c, cost, s in zip(z_carbon, z_cost, z_strength):
        is_low_carbon = c <= -0.5
        is_low_cost = cost <= -0.5
        is_high_strength = s >= 0.5

        if is_low_carbon and not is_low_cost and not is_high_strength:
            labels.append("Low carbon plan")
        elif is_low_cost and not is_low_carbon and not is_high_strength:
            labels.append("Low cost plan")
        elif is_high_strength and not is_low_carbon and not is_low_cost:
            labels.append("High strength plan")
        else:
            labels.append("Balanced plan")

    df["plan_category"] = labels
    return df


