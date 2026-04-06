from __future__ import annotations

"""
Generate synthetic datasets for the project.

This generator creates 5 CSVs (each 1000 rows) with the exact column names
expected by:
- app.data.features.prepare_feature_matrix()
- app.data.emissions.compute_emission_breakdown()
- app.optimization.* and dashboard rendering

IMPORTANT:
- This script is for creating a benchmark dataset.
- It does not attempt to "rig" results in favor of any specific model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratorConfig:
    n_rows: int = 1000
    seed: int = 42

    # Output filenames must match app/config.py Paths.*
    materials_file: str = "materials_dataset_1000.csv"
    scenarios_file: str = "scenario_dataset_1000.csv"
    locations_file: str = "location_dataset_1000.csv"
    equipment_file: str = "equipment_dataset_1000.csv"
    schedule_file: str = "construction_schedule_dataset_1000.csv"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_materials(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_rows
    material_id = np.arange(1, n + 1)

    # Strength (MPa): skewed distribution
    compressive_strength_mpa = rng.lognormal(mean=3.0, sigma=0.25, size=n)  # ~ 20..100+

    # Cost per unit: correlated with strength, with extra noise
    cost_per_unit = (0.6 * compressive_strength_mpa + rng.normal(0, 5.0, size=n)).clip(1.0)

    # Embodied carbon: decreases slightly with efficiency (strength), but noisy
    embodied_carbon = (
        0.9 * rng.lognormal(mean=1.0, sigma=0.35, size=n) * (1.0 + 15.0 / (compressive_strength_mpa + 5.0))
    )

    # Quantity used: correlated with strength and cost
    quantity_used = (
        rng.uniform(50, 250, size=n)
        * (0.8 + 0.0025 * compressive_strength_mpa)
        * (1.0 + rng.normal(0, 0.08, size=n))
    ).clip(10, None)

    return pd.DataFrame(
        {
            "material_id": material_id,
            "compressive_strength_mpa": compressive_strength_mpa,
            "cost_per_unit": cost_per_unit,
            "embodied_carbon": embodied_carbon,
            "quantity_used": quantity_used,
        }
    )


def generate_equipment(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_rows
    equipment_id = np.arange(1, n + 1)

    # Fuel consumption (liters/hour): larger machines generally consume more
    fuel_consumption = rng.lognormal(mean=2.2, sigma=0.35, size=n)  # ~ 10..200+

    # CO2 emission factor: moderately correlated with fuel consumption, with noise
    co2_emission_factor = (0.05 * fuel_consumption + rng.normal(0, 0.2, size=n)).clip(0.01, None)

    # Operation hours: depends on project scale
    operation_hours = rng.uniform(20, 180, size=n) * (1.0 + rng.normal(0, 0.12, size=n))
    operation_hours = operation_hours.clip(5, None)

    return pd.DataFrame(
        {
            "equipment_id": equipment_id,
            "fuel_consumption": fuel_consumption,
            "co2_emission_factor": co2_emission_factor,
            "operation_hours": operation_hours,
        }
    )


def generate_locations(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_rows
    location_id = np.arange(1, n + 1)

    transport_distance_km = rng.uniform(5, 500, size=n)  # keep realistic
    regional_emission_factor = (rng.uniform(0.7, 1.6, size=n) + rng.normal(0, 0.03, size=n)).clip(0.5, None)

    # Higher availability => better material logistics (less waste)
    material_availability_score = (100 * (0.35 + 0.65 * rng.beta(2.0, 2.5, size=n))).clip(0, 100)

    return pd.DataFrame(
        {
            "location_id": location_id,
            "transport_distance_km": transport_distance_km,
            "regional_emission_factor": regional_emission_factor,
            "material_availability_score": material_availability_score,
        }
    )


def generate_schedule(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_rows
    activity_id = np.arange(1, n + 1)

    duration_days = rng.uniform(20, 220, size=n) * (1.0 + rng.normal(0, 0.12, size=n))
    duration_days = duration_days.clip(5, None)

    resource_workers = rng.uniform(5, 80, size=n) * (1.0 + rng.normal(0, 0.18, size=n))
    resource_workers = resource_workers.clip(1, None)

    return pd.DataFrame(
        {
            "activity_id": activity_id,
            "duration_days": duration_days,
            "resource_workers": resource_workers,
        }
    )


def generate_scenarios(
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    materials: pd.DataFrame,
    equipment: pd.DataFrame,
    locations: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    n = cfg.n_rows

    # For features.py:
    # - location and schedule features are assigned by row-order modulo.
    # Since all tables have length 1000, scenario row i uses location_id=i+1 and activity_id=i+1.
    loc = locations.set_index("location_id").loc[np.arange(1, n + 1)]
    sch = schedule.set_index("activity_id").loc[np.arange(1, n + 1)]

    # Material/equipment used are true per-scenario ids.
    materials_used = rng.integers(1, n + 1, size=n)
    equipment_used = rng.integers(1, n + 1, size=n)
    mat = materials.set_index("material_id").loc[materials_used]
    eq = equipment.set_index("equipment_id").loc[equipment_used]

    # Material strength score (0..100), derived from compressive strength.
    # Use a squashed mapping so the filters (0..100) remain sensible.
    strength_raw = (mat["compressive_strength_mpa"].values - mat["compressive_strength_mpa"].mean()) / (
        mat["compressive_strength_mpa"].std() + 1e-9
    )
    material_strength_score = (100 * _sigmoid(1.2 * strength_raw)).clip(0, 100)

    # Base emissions components (must exist for compute_emission_breakdown)
    # - material_emissions ~ embodied_carbon * quantity_used (with logistic modulation)
    logistic_mod = 0.75 + 0.5 * _sigmoid((mat["cost_per_unit"].values - mat["cost_per_unit"].mean()) / 50.0)
    material_emissions = (mat["embodied_carbon"].values * mat["quantity_used"].values * logistic_mod).clip(1e-6, None)

    # - energy_consumption ~ fuel_consumption * operation_hours scaled by regional factor
    energy_consumption = (eq["fuel_consumption"].values * eq["operation_hours"].values) * (
        0.7 + 0.3 * loc["regional_emission_factor"].values
    )
    energy_consumption = energy_consumption.clip(1e-6, None)

    # - transport_emissions ~ distance * regional factor * (quantity proxy)
    qty_proxy = (mat["quantity_used"].values / (mat["quantity_used"].values.mean() + 1e-9))
    transport_emissions = loc["transport_distance_km"].values * loc["regional_emission_factor"].values * (0.02 * qty_proxy + 0.1)
    transport_emissions = transport_emissions.clip(1e-6, None)

    # Cost
    total_cost = (
        mat["cost_per_unit"].values * mat["quantity_used"].values * 0.9
        + eq["fuel_consumption"].values * eq["operation_hours"].values * 2.5
        + loc["transport_distance_km"].values * 40.0 * (0.5 + 0.5 * loc["material_availability_score"].values / 100.0)
    )
    total_cost = total_cost * (1.0 + rng.normal(0, 0.05, size=n))
    total_cost = total_cost.clip(1.0, None)

    # Duration (days): driven by schedule duration, slightly influenced by equipment utilization
    construction_time_days = sch["duration_days"].values * (0.85 + 0.3 * (eq["operation_hours"].values / (eq["operation_hours"].values.mean() + 1e-9)))
    construction_time_days = construction_time_days + rng.normal(0, 3.0, size=n)
    construction_time_days = construction_time_days.clip(1.0, None)

    # Target total_co2: mixture with nonlinear interaction term + noise.
    # This gives an appropriate benchmark for ensemble methods.
    nonlinear = (
        0.05 * np.sin(mat["compressive_strength_mpa"].values / 10.0)
        + 0.03 * np.log1p(eq["fuel_consumption"].values)
        + 0.02 * np.sqrt(loc["transport_distance_km"].values)
    )
    interaction = 0.0002 * mat["quantity_used"].values * eq["co2_emission_factor"].values * (loc["regional_emission_factor"].values)

    total_co2 = (
        0.48 * material_emissions
        + 0.32 * energy_consumption
        + 0.20 * transport_emissions
    )
    total_co2 = total_co2 * (1.0 + nonlinear + interaction)
    total_co2 = total_co2 + rng.normal(0, 0.03 * np.maximum(total_co2, 1.0), size=n)
    total_co2 = total_co2.clip(1e-6, None)

    # Scenario-level energy/transport/material emissions are what the emissions breakdown uses.
    return pd.DataFrame(
        {
            "materials_used": materials_used.astype(int),
            "equipment_used": equipment_used.astype(int),
            "energy_consumption": energy_consumption,
            "transport_emissions": transport_emissions,
            "material_emissions": material_emissions,
            "total_cost": total_cost,
            "material_strength_score": material_strength_score,
            "construction_time_days": construction_time_days,
            "total_co2": total_co2,
        }
    )


def generate_all(cfg: GeneratorConfig, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = _make_rng(cfg.seed)

    materials = generate_materials(cfg, rng)
    equipment = generate_equipment(cfg, rng)
    locations = generate_locations(cfg, rng)
    schedule = generate_schedule(cfg, rng)
    scenarios = generate_scenarios(cfg, rng, materials, equipment, locations, schedule)

    paths = {
        "materials": out_dir / cfg.materials_file,
        "equipment": out_dir / cfg.equipment_file,
        "locations": out_dir / cfg.locations_file,
        "schedule": out_dir / cfg.schedule_file,
        "scenarios": out_dir / cfg.scenarios_file,
    }

    materials.to_csv(paths["materials"], index=False)
    equipment.to_csv(paths["equipment"], index=False)
    locations.to_csv(paths["locations"], index=False)
    schedule.to_csv(paths["schedule"], index=False)
    scenarios.to_csv(paths["scenarios"], index=False)

    return paths


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "dataset"

    cfg = GeneratorConfig(n_rows=1000, seed=42)
    out = generate_all(cfg, dataset_dir)
    for k, p in out.items():
        print(f"{k}: {p}")

