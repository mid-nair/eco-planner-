from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def topsis_rank(
    plans: pd.DataFrame,
    benefit_cols: List[str],
    cost_cols: List[str],
    weights: Dict[str, float] | None = None,
    score_col: str = "topsis_score",
) -> pd.DataFrame:
    """
    Apply TOPSIS ranking to the given plans.

    benefit_cols: columns where higher is better (e.g., strength, sustainability)
    cost_cols: columns where lower is better (e.g., cost, emissions, duration)
    """
    df = plans.copy()
    all_cols = benefit_cols + cost_cols
    mat = df[all_cols].astype(float).values

    # Normalize columns
    norm = np.linalg.norm(mat, axis=0)
    norm[norm == 0] = 1.0
    r = mat / norm

    # Weights
    if weights is None:
        w = np.ones(len(all_cols)) / len(all_cols)
    else:
        w = np.array([weights.get(c, 1.0) for c in all_cols], dtype=float)
        if w.sum() > 0:
            w = w / w.sum()

    v = r * w

    # Ideal best/worst
    benefit_idx = [all_cols.index(c) for c in benefit_cols]
    cost_idx = [all_cols.index(c) for c in cost_cols]

    v_plus = np.zeros(v.shape[1])
    v_minus = np.zeros(v.shape[1])

    for j in range(v.shape[1]):
        if j in benefit_idx:
            v_plus[j] = v[:, j].max()
            v_minus[j] = v[:, j].min()
        else:
            v_plus[j] = v[:, j].min()
            v_minus[j] = v[:, j].max()

    # Distances
    d_plus = np.linalg.norm(v - v_plus, axis=1)
    d_minus = np.linalg.norm(v - v_minus, axis=1)

    # Score between 0 and 1
    s = d_minus / (d_plus + d_minus + 1e-9)
    df[score_col] = s

    # Higher score is better
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["topsis_rank"] = np.arange(1, len(df) + 1)

    # Map to a 0–100 sustainability score
    df["sustainability_score"] = (df[score_col] * 100).round(2)

    return df


