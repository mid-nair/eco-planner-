from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_feature_matrix(
    datasets: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Build a learning-ready feature matrix from the scenario and reference datasets.

    The core target is total_co2 from the scenario dataset. We enrich with
    engineered features derived from materials, equipment, locations, and schedule.
    """
    scenarios = datasets["scenarios"].copy()

    # Basic sanity cleaning: drop rows with missing target
    scenarios = scenarios.dropna(subset=["total_co2"])

    # Simple engineered features using ID lookups
    materials = datasets["materials"].set_index("material_id")
    equipment = datasets["equipment"].set_index("equipment_id")
    locations = datasets["locations"].set_index("location_id")

    # Map ID fields to numeric descriptors
    def _safe_lookup(df: pd.DataFrame, key_col: str, feature_cols: list[str]) -> pd.DataFrame:
        ids = scenarios[key_col].astype(int)
        joined = df.reindex(ids.values)[feature_cols].reset_index(drop=True)
        joined.columns = [f"{key_col}_{c}" for c in feature_cols]
        return joined

    mat_feats = _safe_lookup(
        materials,
        "materials_used",
        ["compressive_strength_mpa", "cost_per_unit", "embodied_carbon", "quantity_used"],
    )

    eq_feats = _safe_lookup(
        equipment,
        "equipment_used",
        ["fuel_consumption", "co2_emission_factor", "operation_hours"],
    )

    # Location features: approximate by using material_emissions as proxy to choose location row
    # (in real system, user would pass explicit location ID)
    loc_df = locations.copy()
    loc_df["loc_index"] = loc_df.index
    # For now, randomly assign locations deterministically via modulo
    loc_index = (np.arange(len(scenarios)) % len(loc_df)) + 1
    loc_feats = locations.reindex(loc_index)[
        ["transport_distance_km", "regional_emission_factor", "material_availability_score"]
    ].reset_index(drop=True)
    loc_feats.columns = [f"location_{c}" for c in loc_feats.columns]

    # Aggregate schedule info by matching activity_id ranges to scenario index (approximation)
    schedule = datasets["schedule"]
    schedule_summary = schedule.groupby("activity_id").agg(
        avg_duration=("duration_days", "mean"),
        avg_workers=("resource_workers", "mean"),
    )
    sched_index = (np.arange(len(scenarios)) % len(schedule_summary)) + 1
    sched_feats = schedule_summary.reindex(sched_index)[["avg_duration", "avg_workers"]].reset_index(
        drop=True
    )
    sched_feats.columns = [f"schedule_{c}" for c in sched_feats.columns]

    base_numeric = scenarios[
        [
            "energy_consumption",
            "transport_emissions",
            "material_emissions",
            "total_cost",
            "material_strength_score",
            "construction_time_days",
        ]
    ].copy()

    # Concatenate all features
    X = pd.concat([base_numeric, mat_feats, eq_feats, loc_feats, sched_feats], axis=1)
    y = scenarios["total_co2"].copy()

    # Handle missing values simply by filling with column medians
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, scaler


def build_feature_matrix_for_scenarios(
    datasets: Dict[str, pd.DataFrame],
    scaler: StandardScaler,
    feature_columns: pd.Index,
    scenarios_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a feature matrix for an arbitrary scenarios DataFrame using
    the same engineering steps and column order as training.

    This reuses the fitted scaler and aligns columns to `feature_columns`.
    """
    local = dict(datasets)
    local["scenarios"] = scenarios_df.copy()

    # Reuse the logic from prepare_feature_matrix, but without fitting scaler
    scenarios = local["scenarios"].copy()

    materials = local["materials"].set_index("material_id")
    equipment = local["equipment"].set_index("equipment_id")
    locations = local["locations"].set_index("location_id")

    def _safe_lookup(df: pd.DataFrame, key_col: str, feature_cols: list[str]) -> pd.DataFrame:
        ids = scenarios[key_col].astype(int)
        joined = df.reindex(ids.values)[feature_cols].reset_index(drop=True)
        joined.columns = [f"{key_col}_{c}" for c in feature_cols]
        return joined

    mat_feats = _safe_lookup(
        materials,
        "materials_used",
        ["compressive_strength_mpa", "cost_per_unit", "embodied_carbon", "quantity_used"],
    )

    eq_feats = _safe_lookup(
        equipment,
        "equipment_used",
        ["fuel_consumption", "co2_emission_factor", "operation_hours"],
    )

    loc_df = locations.copy()
    loc_df["loc_index"] = loc_df.index
    loc_index = (np.arange(len(scenarios)) % len(loc_df)) + 1
    loc_feats = locations.reindex(loc_index)[
        ["transport_distance_km", "regional_emission_factor", "material_availability_score"]
    ].reset_index(drop=True)
    loc_feats.columns = [f"location_{c}" for c in loc_feats.columns]

    schedule = local["schedule"]
    schedule_summary = schedule.groupby("activity_id").agg(
        avg_duration=("duration_days", "mean"),
        avg_workers=("resource_workers", "mean"),
    )
    sched_index = (np.arange(len(scenarios)) % len(schedule_summary)) + 1
    sched_feats = schedule_summary.reindex(sched_index)[["avg_duration", "avg_workers"]].reset_index(
        drop=True
    )
    sched_feats.columns = [f"schedule_{c}" for c in sched_feats.columns]

    base_numeric = scenarios[
        [
            "energy_consumption",
            "transport_emissions",
            "material_emissions",
            "total_cost",
            "material_strength_score",
            "construction_time_days",
        ]
    ].copy()

    X = pd.concat([base_numeric, mat_feats, eq_feats, loc_feats, sched_feats], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Align to training column order, fill missing engineered columns with zeros
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_columns)

    return X_scaled


