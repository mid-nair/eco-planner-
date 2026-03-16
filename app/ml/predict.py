from __future__ import annotations

from typing import Any

import pandas as pd

from app.data.emissions import compute_emission_breakdown


def predict_emissions(
    model: Any,
    X_scaled: pd.DataFrame,
    scenarios_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run predictions for a set of scenarios and attach breakdown + basic KPIs.

    X_scaled: feature matrix already transformed with the training scaler.
    scenarios_df: original scenarios rows to which predictions will be attached.
    """
    y_pred = model.predict(X_scaled)

    df = scenarios_df.copy()
    df["predicted_total_co2"] = y_pred

    # Add simple breakdown columns
    breakdowns = df.apply(compute_emission_breakdown, axis=1)
    df["co2_materials"] = [b["Materials"] for b in breakdowns]
    df["co2_energy"] = [b["Energy"] for b in breakdowns]
    df["co2_transport"] = [b["Transportation"] for b in breakdowns]

    return df


