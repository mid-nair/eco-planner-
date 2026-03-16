from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px


def carbon_breakdown_chart(plan_row: pd.Series):
    df = pd.DataFrame(
        {
            "Source": ["Materials", "Energy", "Transportation"],
            "CO2": [
                plan_row["co2_materials"],
                plan_row["co2_energy"],
                plan_row["co2_transport"],
            ],
        }
    )
    fig = px.bar(df, x="Source", y="CO2", title="Carbon Emission Breakdown")
    return fig


def cost_vs_carbon_chart(plans: pd.DataFrame):
    fig = px.scatter(
        plans,
        x="total_cost",
        y="predicted_total_co2",
        color="plan_category",
        hover_data=["material_strength_score", "construction_time_days"],
        title="Cost vs Carbon",
    )
    return fig


def strength_vs_carbon_chart(plans: pd.DataFrame):
    fig = px.scatter(
        plans,
        x="material_strength_score",
        y="predicted_total_co2",
        color="plan_category",
        title="Material Strength vs CO2",
    )
    return fig


def pareto_front_chart(plans: pd.DataFrame):
    fig = px.scatter(
        plans,
        x="predicted_total_co2",
        y="total_cost",
        size="material_strength_score",
        color="construction_time_days",
        title="Optimization Results (Pareto-style)",
        labels={
            "predicted_total_co2": "Carbon emissions",
            "total_cost": "Total cost",
            "construction_time_days": "Duration (days)",
        },
    )
    return fig


def shap_importance_chart(feature_importance: pd.Series, top_n: int = 15):
    top = feature_importance.head(top_n).sort_values()
    fig = px.bar(
        x=top.values,
        y=top.index,
        orientation="h",
        title="SHAP Feature Importance",
        labels={"x": "Mean |SHAP value|", "y": "Feature"},
    )
    return fig


def topsis_ranking_chart(plans_ranked: pd.DataFrame, top_n: int = 20):
    df = plans_ranked.head(top_n)
    fig = px.bar(
        df,
        x="plan_id",
        y="sustainability_score",
        title="TOPSIS Sustainability Ranking",
        labels={"plan_id": "Plan", "sustainability_score": "Score"},
    )
    return fig


def carbon_reduction_chart(original_co2: float, optimized_plans: pd.DataFrame):
    best = optimized_plans["predicted_total_co2"].min()
    reduction = max(original_co2 - best, 0.0)
    df = pd.DataFrame(
        {
            "Scenario": ["Original plan", "Best optimized"],
            "CO2": [original_co2, best],
        }
    )
    fig = px.bar(df, x="Scenario", y="CO2", title="Carbon Reduction from Optimization")
    fig.add_annotation(
        x=0.5,
        y=max(df["CO2"]) * 0.9,
        text=f"Reduction: {reduction:.2f} t CO2",
        showarrow=False,
    )
    return fig


