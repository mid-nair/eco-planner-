from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from app.config import Paths


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """Load all core CSV datasets into a dictionary of DataFrames."""
    return {
        "materials": _read_csv(Paths.materials),
        "scenarios": _read_csv(Paths.scenarios),
        "locations": _read_csv(Paths.locations),
        "equipment": _read_csv(Paths.equipment),
        "schedule": _read_csv(Paths.schedule),
    }


def load_scenarios() -> pd.DataFrame:
    return _read_csv(Paths.scenarios)


