"""Reusable Streamlit page renderers for AgriShield AI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .advisory import DISCLAIMER, generate_advisory
from .climate_risk import apply_scenario, calculate_climate_risk
from .config import METADATA_PATH, MODEL_INPUT_FEATURES, RAW_DATA_PATH, REPORTS_DIR, TARGET
from .crop_recommendation import recommend_crops
from .data_processing import load_data
from .explainability import global_importance, local_explanation
from .prediction import load_metadata, load_model, predict_yield

LIGHT_CHART_COLORS = ("#2f7d4b", "#9a6700", "#c9473f")
DARK_CHART_COLORS = ("#69c58a", "#f2c14e", "#ff7b72")


def _chart_colors() -> tuple[str, str, str]:
    """Return colors with sufficient contrast for the active Streamlit theme."""
    return DARK_CHART_COLORS if st.context.theme.type == "dark" else LIGHT_CHART_COLORS


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data(RAW_DATA_PATH)


@st.cache_resource(show_spinner="Loading trained yield model…")
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return load_metadata()


@st.cache_data(show_spinner="Calculating global importance…")
def get_global_importance() -> pd.DataFrame:
    data = get_data()
    model = get_model()
    sample = data.loc[data["year"] >= 2023]
    return global_importance(model, sample[MODEL_INPUT_FEATURES], sample[TARGET])


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">AgriShield AI</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def academic_notice() -> None:
    st.warning(
        "Synthetic-data prototype. Scenario outputs show model sensitivity, not causal climate forecasts or guaranteed agronomic outcomes.",
        icon="⚠️",
    )


def _record_selector(key: str, default_district: str = "Pune", default_crop: str = "Soybean") -> pd.Series:
    data = get_data()
    c1, c2, c3 = st.columns(3)
    districts = sorted(data["district"].unique())
    district_index = districts.index(default_district) if default_district in districts else 0
    district = c1.selectbox("District", districts, index=district_index, key=f"{key}_district")
    district_data = data.loc[data["district"] == district]
    crops = sorted(district_data["crop"].unique())
    crop_index = crops.index(default_crop) if default_crop in crops else 0
    crop = c2.selectbox("Crop", crops, index=crop_index, key=f"{key}_crop")
    filtered = district_data.loc[district_data["crop"] == crop]
    years = sorted(filtered["year"].unique(), reverse=True)
    year = c3.selectbox("Base year", years, key=f"{key}_year")
    records = filtered.loc[filtered["year"] == year].sort_values("record_id")
    return records.iloc[0].copy()


def _risk_figure(risk: dict) -> go.Figure:
    green, gold, red = _chart_colors()
    labels = ["Drought", "Heat", "Flood", "Overall"]
    values = [risk["drought_score"], risk["heat_score"], risk["flood_score"], risk["overall_score"]]
    colors = [green if v <= 33 else gold if v <= 66 else red for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors, text=[f"{v:.1f}" for v in values], textposition="auto"))
    fig.update_layout(xaxis=dict(range=[0, 100], title="Risk score (0–100)"), yaxis_title="", height=320, margin=dict(l=10, r=20, t=20, b=30))
    return fig


def _local_chart(record: pd.Series) -> pd.DataFrame:
    model = get_model()
    data = get_data()
    explanation = local_explanation(
        model,
        pd.DataFrame([record])[MODEL_INPUT_FEATURES],
        data.loc[data["year"] <= 2022, MODEL_INPUT_FEATURES],
        top_n=8,
    )
    explanation["Direction"] = np.where(explanation["Contribution"] >= 0, "Raises yield", "Lowers yield")
    explanation["Feature"] = explanation["Feature"].map(_humanize_feature)
    return explanation


def _humanize_feature(name: str) -> str:
    category_prefixes = {
        "crop_": "Crop: ",
        "district_": "District: ",
        "season_": "Season: ",
        "agro_climatic_region_": "Region: ",
    }
    for prefix, label in category_prefixes.items():
        if name.startswith(prefix):
            return label + name.removeprefix(prefix).replace("_", " ")
    friendly = {
        "annual_rainfall_mm": "Annual rainfall",
        "rainfall_anomaly_pct": "Rainfall anomaly",
        "avg_temperature_c": "Average temperature",
        "max_temperature_c": "Maximum temperature",
        "temperature_anomaly_c": "Temperature anomaly",
        "soil_moisture_pct": "Soil moisture",
        "irrigation_pct": "Irrigation coverage",
        "consecutive_dry_days": "Consecutive dry days",
        "heavy_rain_days": "Heavy-rain days",
        "hot_days": "Hot days",
        "soil_fertility_index": "Soil fertility index",
        "rainfall_requirement_ratio": "Crop-relative rainfall",
        "growing_condition_score": "Growing-condition score",
    }
    return friendly.get(name, name.replace("_", " ").title())


def _show_local_explanation(record: pd.Series) -> None:
    green, _, red = _chart_colors()
    with st.spinner("Calculating the local explanation…"):
        explanation = _local_chart(record)
    fig = px.bar(
        explanation,
        x="Contribution",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={"Raises yield": green, "Lowers yield": red},
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=15, b=20), legend_title="")
    st.plotly_chart(fig, width="stretch")
    positive = explanation.loc[explanation["Contribution"] > 0].nlargest(3, "Contribution")
    negative = explanation.loc[explanation["Contribution"] < 0].nsmallest(3, "Contribution")
    pcol, ncol = st.columns(2)
    pcol.markdown("**Top positive factors**")
    pcol.markdown("\n".join(f"- {row.Feature}" for row in positive.itertuples()) or "- No positive factor among the top contributions")
    ncol.markdown("**Top negative factors**")
    ncol.markdown("\n".join(f"- {row.Feature}" for row in negative.itertuples()) or "- No negative factor among the top contributions")
    st.caption("Contributions are calculated on the encoded feature space. One-hot category names may appear explicitly.")


def render_dashboard() -> None:
    hero("Data for Fields. Decisions for Tomorrow.", "Yield prediction, transparent climate-risk analytics, and scenario planning for Maharashtra agriculture.")
    academic_notice()
    st.subheader("Presentation scenario")
    record = _record_selector("dashboard")
    model = get_model()
    prediction = predict_yield(model, record)
    risk = calculate_climate_risk(record)

    cards = st.columns(4)
    cards[0].metric("Predicted yield", f"{prediction:.2f} t/ha", help="Model estimate for selected record")
    cards[1].metric("Overall climate risk", f"{risk['overall_score']:.1f}/100", str(risk["overall_category"]), delta_color="off")
    cards[2].metric("Annual rainfall", f"{record['annual_rainfall_mm']:,.0f} mm", f"{record['rainfall_anomaly_pct']:+.1f}% anomaly", delta_color="off")
    cards[3].metric("Average temperature", f"{record['avg_temperature_c']:.1f} °C", f"{record['temperature_anomaly_c']:+.1f} °C anomaly", delta_color="off")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### Climate-risk breakdown")
        st.plotly_chart(_risk_figure(risk), width="stretch")
    with right:
        st.markdown("### Historical yield trend")
        data = get_data()
        trend = data.loc[(data["district"] == record["district"]) & (data["crop"] == record["crop"])].groupby("year", as_index=False)[TARGET].mean()
        fig = px.line(trend, x="year", y=TARGET, markers=True, color_discrete_sequence=[_chart_colors()[0]])
        fig.update_layout(yaxis_title="Yield (t/ha)", xaxis_title="Year", height=320, margin=dict(l=10, r=10, t=20, b=30))
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Demo what-if: +2 °C and −20% rainfall")
    scenario = apply_scenario(record, temperature_change_c=2, rainfall_change_pct=-20)
    scenario_prediction = predict_yield(model, scenario)
    scenario_risk = calculate_climate_risk(scenario)
    change = 100 * (scenario_prediction - prediction) / prediction if prediction else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline yield", f"{prediction:.2f} t/ha")
    m2.metric("Scenario yield", f"{scenario_prediction:.2f} t/ha", f"{change:+.1f}%")
    m3.metric("Drought risk", f"{scenario_risk['drought_score']:.1f}/100", scenario_risk["drought_category"])
    m4.metric("Overall risk", f"{scenario_risk['overall_score']:.1f}/100", scenario_risk["overall_category"])

    a, b = st.columns([1, 1])
    with a:
        st.markdown("### Why the baseline prediction looks this way")
        _show_local_explanation(record)
    with b:
        st.markdown("### Crop alternatives")
        recs = recommend_crops(model, scenario, get_data())
        display = recs[["Rank", "Crop", "Predicted Yield (t/ha)", "Climate Risk", "Suitability Score"]].copy()
        st.dataframe(display.style.format({"Predicted Yield (t/ha)": "{:.2f}", "Climate Risk": "{:.1f}", "Suitability Score": "{:.1f}"}), hide_index=True, width="stretch")
        st.markdown("### Model-based advisory")
        st.info(generate_advisory(scenario, scenario_risk, change))
        st.caption(DISCLAIMER)


def render_yield_prediction() -> None:
    hero("Yield Prediction", "Estimate crop yield from conditions that can be known before harvest.")
    academic_notice()
    record = _record_selector("prediction")
    st.markdown("### Adjust the field and climate inputs")
    c1, c2, c3, c4 = st.columns(4)
    record["annual_rainfall_mm"] = c1.number_input("Annual rainfall (mm)", 0.0, 5000.0, float(record["annual_rainfall_mm"]), 10.0)
    record["avg_temperature_c"] = c2.number_input("Average temperature (°C)", 10.0, 45.0, float(record["avg_temperature_c"]), 0.1)
    record["soil_moisture_pct"] = c3.number_input("Soil moisture (%)", 0.0, 100.0, float(record["soil_moisture_pct"]), 1.0)
    record["irrigation_pct"] = c4.number_input("Irrigation coverage (%)", 0.0, 100.0, float(record["irrigation_pct"]), 1.0)
    n1, n2, n3, n4 = st.columns(4)
    record["nitrogen_kg_ha"] = n1.number_input("Nitrogen (kg/ha)", 0.0, 600.0, float(record["nitrogen_kg_ha"]), 5.0)
    record["phosphorus_kg_ha"] = n2.number_input("Phosphorus (kg/ha)", 0.0, 100.0, float(record["phosphorus_kg_ha"]), 1.0)
    record["potassium_kg_ha"] = n3.number_input("Potassium (kg/ha)", 0.0, 750.0, float(record["potassium_kg_ha"]), 5.0)
    record["soil_ph"] = n4.number_input("Soil pH", 3.5, 10.0, float(record["soil_ph"]), 0.1)
    prediction = predict_yield(get_model(), record)
    risk = calculate_climate_risk(record)
    st.success(f"Predicted yield: **{prediction:.3f} tonnes/hectare**")
    st.caption(f"Rule-based overall climate risk: {risk['overall_score']:.1f}/100 — {risk['overall_category']}")
    _show_local_explanation(record)


def render_climate_risk() -> None:
    hero("Climate Risk Analytics", "Transparent drought, heat, flood, and combined sensitivity scores from 0 to 100.")
    record = _record_selector("risk")
    risk = calculate_climate_risk(record)
    st.plotly_chart(_risk_figure(risk), width="stretch")
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Drought", "Heat", "Flood", "Overall"], ["drought", "heat", "flood", "overall"]):
        col.metric(label, f"{risk[f'{key}_score']:.1f}/100", risk[f"{key}_category"])
    st.info("Scores are rule-based analytics built from rainfall anomaly, dry days, temperature, heavy rain, soil moisture, irrigation, and crop requirements. They are not disaster probabilities or ML classifications.")


def render_simulator() -> None:
    hero("What-If Climate Simulator", "Change selected conditions and compare model sensitivity with the recorded baseline.")
    academic_notice()
    base = _record_selector("simulator")
    st.markdown("### Scenario controls")
    c1, c2, c3, c4 = st.columns(4)
    temp = c1.slider("Temperature change (°C)", -3.0, 5.0, 2.0, 0.1)
    rain = c2.slider("Rainfall change (%)", -50, 50, -20, 1)
    moisture = c3.slider("Soil moisture change (points)", -30, 30, 0, 1)
    irrigation = c4.slider("Irrigation change (points)", -50, 50, 0, 1)
    scenario = apply_scenario(base, temp, rain, moisture, irrigation)
    model = get_model()
    base_yield = predict_yield(model, base)
    scenario_yield = predict_yield(model, scenario)
    change = 100 * (scenario_yield - base_yield) / base_yield if base_yield else 0
    base_risk, scenario_risk = calculate_climate_risk(base), calculate_climate_risk(scenario)
    cols = st.columns(4)
    cols[0].metric("Original yield", f"{base_yield:.2f} t/ha")
    cols[1].metric("Scenario yield", f"{scenario_yield:.2f} t/ha", f"{change:+.1f}%")
    cols[2].metric("Original risk", f"{base_risk['overall_score']:.1f}/100", base_risk["overall_category"])
    cols[3].metric("Scenario risk", f"{scenario_risk['overall_score']:.1f}/100", scenario_risk["overall_category"])
    a, b = st.columns(2)
    with a:
        comparison = pd.DataFrame({
            "Risk": ["Drought", "Heat", "Flood", "Overall"] * 2,
            "Score": [base_risk[f"{k}_score"] for k in ["drought", "heat", "flood", "overall"]] + [scenario_risk[f"{k}_score"] for k in ["drought", "heat", "flood", "overall"]],
            "Scenario": ["Baseline"] * 4 + ["What-if"] * 4,
        })
        baseline_color = "#a8b7ad" if st.context.theme.type == "dark" else "#6f8476"
        fig = px.bar(comparison, x="Risk", y="Score", color="Scenario", barmode="group", range_y=[0, 100], color_discrete_map={"Baseline": baseline_color, "What-if": _chart_colors()[2]})
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("### Scenario explanation")
        st.info(generate_advisory(scenario, scenario_risk, change))
        st.caption(DISCLAIMER)
        st.markdown("**Changed modeled inputs**")
        st.dataframe(pd.DataFrame({"Condition": ["Temperature", "Rainfall", "Soil moisture", "Irrigation"], "Change": [f"{temp:+.1f} °C", f"{rain:+d}%", f"{moisture:+d} points", f"{irrigation:+d} points"]}), hide_index=True, width="stretch")


def render_crop_recommendation() -> None:
    hero("Crop Recommendation", "Rank crops by crop-relative modeled performance and climate-risk resilience.")
    academic_notice()
    record = _record_selector("recommend")
    ranking = recommend_crops(get_model(), record, get_data())
    st.dataframe(
        ranking.style.format({"Predicted Yield (t/ha)": "{:.3f}", "Crop-relative Performance": "{:.1f}", "Climate Risk": "{:.1f}", "Suitability Score": "{:.1f}"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Suitability formula:** 72% crop-specific historical percentile of predicted yield + 28% climate resilience (`100 − risk`). This avoids favoring high-tonnage crops such as sugarcane purely because of their natural yield scale.")


def render_district_analysis() -> None:
    hero("District Analysis", "Compare agricultural performance and supplied historical climate-risk scores across Maharashtra.")
    data = get_data()
    default = [d for d in ["Pune", "Nagpur", "Nashik"] if d in set(data["district"])]
    districts = st.multiselect("Districts to compare", sorted(data["district"].unique()), default=default, max_selections=6)
    if not districts:
        st.warning("Select at least one district.")
        return
    subset = data.loc[data["district"].isin(districts)]
    summary = subset.groupby("district").agg(
        Historical_Yield=(TARGET, "mean"),
        Rainfall=("annual_rainfall_mm", "mean"),
        Temperature=("avg_temperature_c", "mean"),
        Drought_Risk=("drought_risk_score", "mean"),
        Heat_Risk=("heat_risk_score", "mean"),
        Flood_Risk=("flood_risk_score", "mean"),
        Overall_Risk=("overall_climate_risk_score", "mean"),
    ).reset_index()
    main_crops = subset.groupby("district")["crop"].apply(lambda s: ", ".join(s.value_counts().head(3).index)).rename("Main Crops")
    summary = summary.merge(main_crops, on="district")
    st.dataframe(summary.style.format({c: "{:.1f}" for c in summary.select_dtypes("number").columns}), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        fig = px.bar(summary, x="district", y="Historical_Yield", color="Overall_Risk", color_continuous_scale="RdYlGn_r")
        fig.update_layout(yaxis_title="Mean yield (t/ha)", xaxis_title="District")
        st.plotly_chart(fig, width="stretch")
    with b:
        long = summary.melt(id_vars="district", value_vars=["Drought_Risk", "Heat_Risk", "Flood_Risk"], var_name="Risk", value_name="Score")
        fig = px.bar(long, x="district", y="Score", color="Risk", barmode="group")
        st.plotly_chart(fig, width="stretch")
    st.info("A boundary map is intentionally not fabricated because no verified Maharashtra district GeoJSON was supplied. This comparison view can be replaced by a choropleth after adding an authoritative boundary file with district-name validation.")


def render_historical_trends() -> None:
    hero("Historical Trends", "Filter 25 years of synthetic crop, weather, and risk observations.")
    data = get_data()
    c1, c2, c3 = st.columns(3)
    crops = c1.multiselect("Crops", sorted(data["crop"].unique()), default=[])
    districts = c2.multiselect("Districts", sorted(data["district"].unique()), default=[])
    years = c3.slider("Year range", int(data["year"].min()), int(data["year"].max()), (int(data["year"].min()), int(data["year"].max())))
    filtered = data.loc[data["year"].between(*years)]
    if crops:
        filtered = filtered.loc[filtered["crop"].isin(crops)]
    if districts:
        filtered = filtered.loc[filtered["district"].isin(districts)]
    yearly = filtered.groupby("year", as_index=False).agg(Yield=(TARGET, "mean"), Rainfall=("annual_rainfall_mm", "mean"), Temperature=("avg_temperature_c", "mean"), Risk=("overall_climate_risk_score", "mean"))
    selected = st.selectbox("Trend metric", ["Yield", "Rainfall", "Temperature", "Risk"])
    fig = px.line(yearly, x="year", y=selected, markers=True, color_discrete_sequence=[_chart_colors()[0]])
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        xvar = st.selectbox("Yield relationship", ["annual_rainfall_mm", "avg_temperature_c", "overall_climate_risk_score"])
        fig = px.scatter(filtered, x=xvar, y=TARGET, color="crop", opacity=0.48, trendline=None)
        st.plotly_chart(fig, width="stretch")
    with b:
        crop_summary = filtered.groupby("crop", as_index=False)[TARGET].mean().sort_values(TARGET)
        fig = px.bar(crop_summary, x=TARGET, y="crop", orientation="h", color=TARGET, color_continuous_scale="Greens")
        st.plotly_chart(fig, width="stretch")


def render_explainability() -> None:
    hero("Explainability", "Inspect global permutation importance and local SHAP contributions.")
    academic_notice()
    st.markdown("### Global feature importance")
    importance = get_global_importance().sort_values("Importance")
    fig = px.bar(importance, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Greens")
    st.plotly_chart(fig, width="stretch")
    st.caption("Permutation importance measures test-sample MAE degradation when a raw input is shuffled. Correlated variables can share importance.")
    st.markdown("### Local explanation")
    record = _record_selector("explain")
    prediction = predict_yield(get_model(), record)
    st.metric("Predicted yield", f"{prediction:.3f} t/ha")
    _show_local_explanation(record)


def render_advisory() -> None:
    hero("Agricultural Advisory", "Condition-based guidance grounded in the selected record and transparent risk rules.")
    academic_notice()
    record = _record_selector("advisory")
    risk = calculate_climate_risk(record)
    prediction = predict_yield(get_model(), record)
    st.metric("Modeled yield", f"{prediction:.3f} t/ha")
    st.info(generate_advisory(record, risk))
    st.warning(DISCLAIMER)
    st.markdown("The advisory changes only when its underlying conditions cross documented thresholds. It does not call an external language model and does not invent weather, soil, or farm-management facts.")


def render_about() -> None:
    hero("About AgriShield AI", "An academically honest, portfolio-ready decision-support prototype.")
    metadata = get_metadata()
    st.markdown("""
### What this project demonstrates

AgriShield AI combines a time-aware yield regression pipeline, a separate transparent climate-risk engine, transformed-feature SHAP explanations, crop-normalized recommendations, and interactive scenario testing.

### Leakage controls

- `production_tonnes` is excluded because it is calculated from yield and cultivated area after harvest.
- Supplied climate-risk scores and labels are excluded from yield prediction and used only for historical analytics.
- Preprocessing and feature engineering live inside the fitted pipeline, so imputation, scaling, and encoding do not learn from held-out years.

### Limitations

The supplied data is synthetic. It does not represent verified farm measurements. Real deployment would require validation with official crop production, meteorological, soil, remote-sensing, and farm-management data. The simulator measures model sensitivity, not causal climate impact. Local agronomists must validate any operational recommendation.
""")
    if metadata:
        st.markdown("### Saved model facts")
        st.json(metadata, expanded=False)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.markdown("### Model evaluation")
        st.dataframe(comparison.style.format({c: "{:.4f}" for c in comparison.select_dtypes("number").columns}), hide_index=True, width="stretch")
        st.caption("MAPE excludes target values below 0.10 t/ha because percentage error becomes unstable near zero.")
