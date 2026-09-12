"""Reproducible dataset audit used by the project report and notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, TARGET


def analyze_dataset(data: pd.DataFrame) -> dict:
    numeric = data.select_dtypes(include=np.number)
    target_corr = numeric.corr(numeric_only=True)[TARGET].sort_values(key=abs, ascending=False)
    identity_error = (data["production_tonnes"] - data[TARGET] * data["cultivated_area_ha"]).abs()
    summary = {
        "shape": list(data.shape),
        "year_range": [int(data["year"].min()), int(data["year"].max())],
        "districts": int(data["district"].nunique()),
        "crops": sorted(data["crop"].unique().tolist()),
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_record_ids": int(data["record_id"].duplicated().sum()),
        "numeric_columns": numeric.columns.tolist(),
        "categorical_columns": data.select_dtypes(exclude=np.number).columns.tolist(),
        "target_summary": data[TARGET].describe().to_dict(),
        "production_identity": {
            "median_absolute_rounding_error_tonnes": float(identity_error.median()),
            "max_absolute_rounding_error_tonnes": float(identity_error.max()),
            "conclusion": "production_tonnes is a post-harvest arithmetic derivative and target leakage",
        },
        "top_absolute_target_correlations": target_corr.head(12).to_dict(),
        "risk_category_counts": {
            c: data[c].value_counts().to_dict()
            for c in ["drought_risk", "heat_risk", "flood_risk", "overall_climate_risk"]
        },
        "quality_flags": [
            "Yield has a floor at 0.08 t/ha and a strongly crop-dependent distribution.",
            "Cultivated area and production are highly right-skewed; robust preprocessing is appropriate.",
            "Overall risk has no High records under the supplied labels, creating class imbalance.",
            "Cross-crop correlations can be confounded by crop scale; group-wise views are essential.",
        ],
    }
    return summary


def write_eda_reports(data: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_dataset(data)
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    data.groupby("crop")[TARGET].agg(["count", "mean", "std", "min", "median", "max"]).to_csv(
        output_dir / "crop_yield_summary.csv"
    )
    data.groupby("district").agg(
        mean_yield=(TARGET, "mean"),
        mean_rainfall=("annual_rainfall_mm", "mean"),
        mean_temperature=("avg_temperature_c", "mean"),
        mean_overall_risk=("overall_climate_risk_score", "mean"),
    ).sort_values("mean_yield", ascending=False).to_csv(output_dir / "district_summary.csv")
    data.groupby("year")[[TARGET, "annual_rainfall_mm", "avg_temperature_c", "overall_climate_risk_score"]].mean().to_csv(
        output_dir / "annual_trends.csv"
    )
    return summary
