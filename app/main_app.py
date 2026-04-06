from __future__ import annotations

import io
import os
import sys
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
import streamlit as st
import plotly.express as px


# Ensure project root is on sys.path so `import app.*` works
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.config import SUPPORTED_MODELS
from app.data.features import prepare_feature_matrix, build_feature_matrix_for_scenarios
from app.data.loaders import load_all_datasets, load_scenarios
from app.explainability.shap_explainer import compute_shap_values
from app.ml.predict import predict_emissions
from app.ml.train_eval import compare_models, train_and_evaluate
from app.optimization.categorization import categorize_plans
from app.optimization.nsga_solver import optimize_plans_nsga
from app.optimization.topsis import topsis_rank
from app.visualization.charts import (
    carbon_breakdown_chart,
    carbon_reduction_chart,
    cost_vs_carbon_chart,
    pareto_front_chart,
    shap_importance_chart,
    strength_vs_carbon_chart,
    topsis_ranking_chart,
)
from app.visualization.plan_cards import render_plan_cards_grid


st.set_page_config(
    page_title="Sustainable Construction Optimizer",
    layout="wide",
)

st.title("Machine Learning–based Carbon Emission Optimization for Construction Projects")
st.caption("Decision support dashboard for low‑carbon, cost‑effective, and resilient construction planning.")


@st.cache_resource(show_spinner=False)
def _load_and_prepare():
    datasets = load_all_datasets()
    X, y, scaler = prepare_feature_matrix(datasets)
    return datasets, X, y, scaler


@st.cache_data(show_spinner=False)
def _compare_all_models_cached(X: pd.DataFrame, y: pd.Series, model_names: list[str]) -> pd.DataFrame:
    # Cache results so users can re-open without retraining everything
    return compare_models(X, y, model_names=model_names, n_repeats=5)


def sidebar_controls():
    st.sidebar.header("Project & Model Settings")
    building_type = st.sidebar.selectbox(
        "Building type",
        ["Residential", "Commercial", "Industrial"],
    )
    building_size = st.sidebar.number_input("Building size (m²)", min_value=50.0, value=1000.0, step=50.0)
    floor_area = st.sidebar.number_input("Floor area (m²)", min_value=50.0, value=800.0, step=50.0)

    st.sidebar.subheader("Machine Learning Model")
    model_name = st.sidebar.selectbox("Select algorithm", SUPPORTED_MODELS, index=0)

    st.sidebar.subheader("Upload additional scenarios (optional)")
    uploaded = st.sidebar.file_uploader("Upload scenario CSV", type=["csv"], accept_multiple_files=False)

    st.sidebar.subheader("Upload project documents (PDF)")
    project_docs = st.sidebar.file_uploader(
        "Upload drawings, BOQs, specs, etc.",
        type=["pdf"],
        accept_multiple_files=True,
    )

    return {
        "building_type": building_type,
        "building_size": building_size,
        "floor_area": floor_area,
        "model_name": model_name,
        "uploaded_scenarios": uploaded,
        "project_docs": project_docs,
    }


def main():
    controls = sidebar_controls()

    with st.spinner("Loading datasets and preparing features..."):
        datasets, X, y, scaler = _load_and_prepare()

    # Show uploaded PDFs (metadata only for now)
    if controls["project_docs"]:
        st.subheader("Uploaded project documents")
        for doc in controls["project_docs"]:
            st.write(f"- {doc.name}")

    st.subheader("1. Train emission prediction model")
    col_train_left, col_train_right = st.columns([1, 2])
    with col_train_left:
        if st.button("Train model", type="primary"):
            st.session_state["model_training_requested"] = True

        if st.button("Compare algorithms"):
            st.session_state["model_comparison_requested"] = True

    if st.session_state.get("model_comparison_requested"):
        with st.spinner("Training and evaluating all algorithms across repeated splits..."):
            comparison_df = _compare_all_models_cached(X, y, SUPPORTED_MODELS)
            st.session_state["comparison_df"] = comparison_df

        best = st.session_state["comparison_df"].iloc[0]
        st.success(
            f"Best model: {best['model']} (Mean R²={best['r2_mean']:.3f} ± {best['r2_std']:.3f}, "
            f"Mean RMSE={best['rmse_mean']:.3f})"
        )
        st.markdown("**Algorithm performance comparison (5 repeated train/test splits)**")
        st.dataframe(st.session_state["comparison_df"], width="stretch")

        fig = px.bar(
            st.session_state["comparison_df"],
            x="model",
            y="r2_mean",
            title="Mean R² by algorithm (higher is better)",
            text="r2_mean",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            yaxis_title="Mean R²",
            xaxis_title="Algorithm",
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )
        st.plotly_chart(fig, width="stretch")

    if st.session_state.get("model_training_requested"):
        with st.spinner(f"Training {controls['model_name']} model..."):
            model, metrics = train_and_evaluate(X, y, controls["model_name"])
            st.session_state["trained_model"] = model
            st.session_state["metrics"] = metrics
            st.session_state["feature_columns"] = X.columns
            st.success("Model trained.")

    metrics = st.session_state.get("metrics")
    if metrics:
        with col_train_right:
            st.markdown("**Model performance**")
            st.write(
                {
                    "R²": round(metrics["r2"], 3),
                    "MAE": round(metrics["mae"], 3),
                    "RMSE": round(metrics["rmse"], 3),
                }
            )

    st.subheader("2. Predict carbon emissions and optimize plans")
    model = st.session_state.get("trained_model")
    if model is None:
        st.info("Train a model first to enable optimization.")
        return

    # Load base scenarios
    base_scenarios = load_scenarios()
    if controls["uploaded_scenarios"] is not None:
        try:
            user_df = pd.read_csv(controls["uploaded_scenarios"])
            base_scenarios = pd.concat([base_scenarios, user_df], ignore_index=True)
        except Exception as e:
            st.warning(f"Failed to read uploaded scenarios: {e}")

    # Add plan_id if not present
    if "plan_id" not in base_scenarios.columns:
        base_scenarios = base_scenarios.copy()
        base_scenarios["plan_id"] = base_scenarios.index + 1

    feature_columns = st.session_state["feature_columns"]

    with st.spinner("Predicting emissions for all scenarios..."):
        X_pred = build_feature_matrix_for_scenarios(
            datasets=datasets,
            scaler=scaler,
            feature_columns=feature_columns,
            scenarios_df=base_scenarios,
        )
        predicted_plans = predict_emissions(model, X_pred, base_scenarios)

    with st.spinner("Performing multi-objective optimization (NSGA-style)..."):
        optimized = optimize_plans_nsga(predicted_plans, n_best=10)
        optimized = categorize_plans(optimized)
        ranked = topsis_rank(
            optimized,
            benefit_cols=["material_strength_score"],
            cost_cols=["predicted_total_co2", "total_cost", "construction_time_days"],
        )

    st.markdown("### Optimized plans (TOPSIS-ranked)")
    render_plan_cards_grid(ranked)

    # Scenario comparison and filters
    st.subheader("3. Scenario comparison and filters")
    filter_cols = st.columns(4)
    max_cost = float(predicted_plans["total_cost"].quantile(0.75))
    max_co2 = float(predicted_plans["predicted_total_co2"].quantile(0.75))
    min_strength = float(predicted_plans["material_strength_score"].quantile(0.25))
    max_duration = float(predicted_plans["construction_time_days"].quantile(0.75))

    cost_filter = filter_cols[0].slider("Max cost (₹)", 0.0, max_cost * 2, max_cost)
    co2_filter = filter_cols[1].slider("Max carbon (t CO₂)", 0.0, max_co2 * 2, max_co2)
    strength_filter = filter_cols[2].slider("Min strength score", 0.0, 100.0, min_strength)
    duration_filter = filter_cols[3].slider("Max duration (days)", 0.0, max_duration * 2, max_duration)

    filtered = ranked[
        (ranked["total_cost"] <= cost_filter)
        & (ranked["predicted_total_co2"] <= co2_filter)
        & (ranked["material_strength_score"] >= strength_filter)
        & (ranked["construction_time_days"] <= duration_filter)
    ]

    st.write(f"Showing {len(filtered)} plans after filtering.")

    # Charts
    st.subheader("4. Visualization dashboard")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Carbon breakdown",
            "Cost vs Carbon",
            "Strength vs CO₂",
            "Optimization (Pareto)",
            "SHAP importance",
            "TOPSIS ranking",
            "Carbon reduction",
        ]
    )

    # Select a reference plan (first optimized)
    ref_plan = ranked.iloc[0]

    with tab1:
        st.plotly_chart(carbon_breakdown_chart(ref_plan), width="stretch")
    with tab2:
        st.plotly_chart(cost_vs_carbon_chart(filtered), width="stretch")
    with tab3:
        st.plotly_chart(strength_vs_carbon_chart(filtered), width="stretch")
    with tab4:
        st.plotly_chart(pareto_front_chart(ranked), width="stretch")
    with tab6:
        st.plotly_chart(topsis_ranking_chart(ranked), width="stretch")
    with tab7:
        original_co2 = float(predicted_plans["predicted_total_co2"].median())
        st.plotly_chart(
            carbon_reduction_chart(original_co2, ranked),
            width="stretch",
        )

    # SHAP explainability
    with tab5:
        with st.spinner("Computing SHAP feature importance on a sample..."):
            # Use the same engineered, scaled features used for prediction
            X_sample = X_pred.sample(min(200, len(X_pred)), random_state=0)
            feature_importance, _ = compute_shap_values(model, X_sample=X_sample)
        st.plotly_chart(shap_importance_chart(feature_importance), width="stretch")

    # Scenario comparison tool
    st.subheader("5. Scenario comparison tool")
    selectable = ranked[["plan_id", "plan_category", "predicted_total_co2", "total_cost"]]
    selected_ids = st.multiselect(
        "Select plans to compare",
        options=list(selectable["plan_id"]),
        format_func=lambda pid: f"Plan {pid}",
    )
    if selected_ids:
        comp = ranked[ranked["plan_id"].isin(selected_ids)].copy()
        st.dataframe(
            comp[
                [
                    "plan_id",
                    "plan_category",
                    "predicted_total_co2",
                    "total_cost",
                    "material_strength_score",
                    "construction_time_days",
                    "sustainability_score",
                ]
            ],
            width="stretch",
        )

    # Downloadable report
    st.subheader("6. Downloadable optimization report")
    report_df = ranked[
        [
            "plan_id",
            "plan_category",
            "predicted_total_co2",
            "total_cost",
            "material_strength_score",
            "construction_time_days",
            "sustainability_score",
        ]
    ].copy()
    csv_buf = io.StringIO()
    report_df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download report as CSV",
        data=csv_buf.getvalue(),
        file_name="optimized_construction_plans.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

