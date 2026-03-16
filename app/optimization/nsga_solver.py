from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ObjectiveWeights:
    carbon: float = 1.0
    cost: float = 1.0
    strength: float = 1.0
    duration: float = 1.0


def _normalize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        col = out[c].astype(float)
        mn, mx = col.min(), col.max()
        if mx > mn:
            out[c] = (col - mn) / (mx - mn)
        else:
            out[c] = 0.0
    return out


def _fast_nondominated_sort(values: np.ndarray) -> List[np.ndarray]:
    """
    Basic non-dominated sorting similar to NSGA-II to obtain Pareto fronts.

    values: shape (n_samples, n_objectives) to be minimized.
    Returns: list of arrays of indices, each array is one front.
    """
    n = values.shape[0]
    dominates = [set() for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: List[np.ndarray] = []

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(values[p] <= values[q]) and np.any(values[p] < values[q]):
                dominates[p].add(q)
            elif np.all(values[q] <= values[p]) and np.any(values[q] < values[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            # belong to first front
            pass

    current_front = np.where(domination_count == 0)[0]
    assigned = set(current_front.tolist())
    fronts.append(current_front)

    while True:
        next_front: List[int] = []
        for p in current_front:
            for q in dominates[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0 and q not in assigned:
                    next_front.append(q)
                    assigned.add(q)
        if not next_front:
            break
        current_front = np.array(next_front, dtype=int)
        fronts.append(current_front)

    return fronts


def optimize_plans_nsga(
    plans: pd.DataFrame,
    n_best: int = 10,
) -> pd.DataFrame:
    """
    Multi-objective optimization over existing plans using NSGA-II style
    non-dominated sorting on four objectives:

    - minimize carbon emissions
    - minimize cost
    - maximize material strength
    - minimize duration

    Returns the n_best Pareto-efficient plans with diversity across the fronts.
    """
    work = plans.copy()
    cols = [
        "predicted_total_co2",
        "total_cost",
        "material_strength_score",
        "construction_time_days",
    ]
    work = _normalize_columns(work, cols)

    # convert max strength to a minimization objective
    f_carbon = work["predicted_total_co2"].values
    f_cost = work["total_cost"].values
    f_strength = 1.0 - work["material_strength_score"].values
    f_duration = work["construction_time_days"].values

    objs = np.vstack([f_carbon, f_cost, f_strength, f_duration]).T

    fronts = _fast_nondominated_sort(objs)

    selected_indices: List[int] = []
    for front in fronts:
        for idx in front:
            selected_indices.append(int(idx))
            if len(selected_indices) >= n_best:
                break
        if len(selected_indices) >= n_best:
            break

    optimized = plans.iloc[selected_indices].copy()
    optimized["pareto_rank"] = np.repeat(range(1, len(fronts) + 1), repeats=1)[: len(optimized)]
    return optimized


