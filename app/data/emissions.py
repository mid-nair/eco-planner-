from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_emission_breakdown(row: pd.Series) -> Dict[str, float]:
    """
    Compute a simple carbon emission breakdown for a single scenario row.

    We use the existing columns:
    - material_emissions
    - energy_consumption
    - transport_emissions
    and scale them so their sum matches total_co2 if available.
    """
    materials = float(row.get("material_emissions", np.nan))
    energy = float(row.get("energy_consumption", np.nan))
    transport = float(row.get("transport_emissions", np.nan))

    parts = np.array([materials, energy, transport], dtype=float)
    parts[np.isnan(parts)] = 0.0

    total = float(row.get("total_co2", np.nansum(parts)))
    s = parts.sum()
    if s <= 0:
        # Fallback: evenly split
        parts = np.array([total / 3.0] * 3)
    else:
        parts = parts / s * total

    return {
        "Materials": float(parts[0]),
        "Energy": float(parts[1]),
        "Transportation": float(parts[2]),
        "Total": float(total),
    }


