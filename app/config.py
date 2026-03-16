import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Paths:
    dataset_dir = BASE_DIR / "dataset"

    materials = dataset_dir / "materials_dataset_1000.csv"
    scenarios = dataset_dir / "scenario_dataset_1000.csv"
    locations = dataset_dir / "location_dataset_1000.csv"
    equipment = dataset_dir / "equipment_dataset_1000.csv"
    schedule = dataset_dir / "construction_schedule_dataset_1000.csv"
    material_carbon = dataset_dir / "construction_material_carbon_dataset_india (1) (1).csv"


class MLConfig:
    target_column = "total_co2"
    random_state = 42
    test_size = 0.2


SUPPORTED_MODELS = [
    "Random Forest",
    "XGBoost",
    "ANN",
    "SVR",
]

