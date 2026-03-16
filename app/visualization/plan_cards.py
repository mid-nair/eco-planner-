from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from app.data.emissions import compute_emission_breakdown


def render_plan_card(plan: pd.Series):
    co2 = float(plan["predicted_total_co2"])
    cost = float(plan["total_cost"])
    strength = float(plan["material_strength_score"])
    time_days = float(plan["construction_time_days"])
    category = plan.get("plan_category", "Plan")
    score = plan.get("sustainability_score", None)

    with st.container(border=True):
        st.subheader(f"Plan #{int(plan['plan_id'])} – {category}")
        cols = st.columns(4)
        cols[0].metric("Carbon (t CO₂)", f"{co2:,.2f}")
        cols[1].metric("Cost (₹)", f"{cost:,.0f}")
        cols[2].metric("Strength score", f"{strength:.2f}")
        cols[3].metric("Duration (days)", f"{time_days:.0f}")

        if score is not None:
            st.markdown(f"**Sustainability score:** {score:.1f} / 100")

        with st.expander("View detailed optimization impact"):
            breakdown = compute_emission_breakdown(plan)
            st.write(
                {
                    "Materials CO2": breakdown["Materials"],
                    "Energy CO2": breakdown["Energy"],
                    "Transport CO2": breakdown["Transportation"],
                }
            )

            if "baseline_total_co2" in plan:
                base = float(plan["baseline_total_co2"])
                diff = base - co2
                pct = (diff / base * 100.0) if base > 0 else 0.0
                st.write(f"**Emission reduction vs original:** {diff:.2f} t CO₂ ({pct:.1f}%)")

            if "baseline_total_cost" in plan:
                base_c = float(plan["baseline_total_cost"])
                delta_c = cost - base_c
                st.write(f"**Cost difference vs original:** {delta_c:,.0f} ₹")

            if "baseline_strength" in plan:
                base_s = float(plan["baseline_strength"])
                delta_s = strength - base_s
                st.write(f"**Strength improvement vs original:** {delta_s:.2f}")


def render_plan_cards_grid(plans: pd.DataFrame, cards_per_row: int = 3):
    plans = plans.reset_index(drop=True)
    for i in range(0, len(plans), cards_per_row):
        row = plans.iloc[i : i + cards_per_row]
        cols = st.columns(len(row))
        for col, (_, plan_row) in zip(cols, row.iterrows()):
            with col:
                render_plan_card(plan_row)


