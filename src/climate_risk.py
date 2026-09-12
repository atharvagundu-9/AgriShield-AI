"""Transparent climate-risk scoring for baseline and scenario analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CROP_REQUIREMENTS


def risk_category(score: float) -> str:
    if score <= 33:
        return "Low"
    if score <= 66:
        return "Medium"
    return "High"


def _bounded(value: float) -> float:
    return float(np.clip(value, 0, 100))


def calculate_climate_risk(row: pd.Series | dict) -> dict[str, float | str]:
    """Return rule-based drought, heat, flood, and combined risk scores.

    These scores are transparent sensitivity indices. They are not ML predictions
    and do not estimate the probability of a real-world disaster.
    """
    r = row if isinstance(row, dict) else row.to_dict()
    crop = str(r.get("crop", "Soybean"))
    needs = CROP_REQUIREMENTS.get(crop, {"rainfall": 800.0, "heat_threshold": 34.0, "irrigation": 50.0})
    rain_ratio = float(r["annual_rainfall_mm"]) / needs["rainfall"]
    deficit = max(0.0, -float(r["rainfall_anomaly_pct"]))
    drought = _bounded(
        0.28 * min(deficit / 50, 1) * 100
        + 0.27 * min(float(r["consecutive_dry_days"]) / 60, 1) * 100
        + 0.22 * max(0, 1 - float(r["soil_moisture_pct"]) / 55) * 100
        + 0.13 * max(0, 1 - float(r["irrigation_pct"]) / max(needs["irrigation"], 1)) * 100
        + 0.10 * max(0, 1 - rain_ratio) * 100
    )
    heat_excess = max(0.0, float(r["max_temperature_c"]) - needs["heat_threshold"])
    heat = _bounded(
        0.40 * min(heat_excess / 8, 1) * 100
        + 0.30 * min(float(r["hot_days"]) / 60, 1) * 100
        + 0.20 * min(max(0, float(r["temperature_anomaly_c"])) / 4, 1) * 100
        + 0.10 * max(0, 1 - float(r["soil_moisture_pct"]) / 55) * 100
    )
    surplus = max(0.0, float(r["rainfall_anomaly_pct"]))
    flood = _bounded(
        0.35 * min(surplus / 50, 1) * 100
        + 0.35 * min(float(r["heavy_rain_days"]) / 20, 1) * 100
        + 0.20 * max(0, min((rain_ratio - 1.15) / 1.35, 1)) * 100
        + 0.10 * max(0, min((float(r["soil_moisture_pct"]) - 60) / 15, 1)) * 100
    )
    overall = _bounded(0.42 * drought + 0.33 * heat + 0.25 * flood)
    return {
        "drought_score": round(drought, 1),
        "drought_category": risk_category(drought),
        "heat_score": round(heat, 1),
        "heat_category": risk_category(heat),
        "flood_score": round(flood, 1),
        "flood_category": risk_category(flood),
        "overall_score": round(overall, 1),
        "overall_category": risk_category(overall),
    }


def apply_scenario(
    row: pd.Series,
    temperature_change_c: float = 0.0,
    rainfall_change_pct: float = 0.0,
    soil_moisture_change: float = 0.0,
    irrigation_change: float = 0.0,
) -> pd.Series:
    scenario = row.copy()
    rain_factor = 1 + rainfall_change_pct / 100
    scenario["annual_rainfall_mm"] = max(0, row["annual_rainfall_mm"] * rain_factor)
    scenario["rainfall_anomaly_pct"] = np.clip(row["rainfall_anomaly_pct"] + rainfall_change_pct, -100, 150)
    for field in ("avg_temperature_c", "max_temperature_c", "min_temperature_c"):
        scenario[field] = row[field] + temperature_change_c
    scenario["temperature_anomaly_c"] = row["temperature_anomaly_c"] + temperature_change_c
    scenario["hot_days"] = max(0, round(row["hot_days"] + max(0, temperature_change_c) * 5))
    scenario["heavy_rain_days"] = max(0, round(row["heavy_rain_days"] * max(0, rain_factor)))
    scenario["consecutive_dry_days"] = max(0, round(row["consecutive_dry_days"] * max(0, 1 - rainfall_change_pct / 100)))
    scenario["soil_moisture_pct"] = float(np.clip(row["soil_moisture_pct"] + soil_moisture_change, 0, 100))
    scenario["irrigation_pct"] = float(np.clip(row["irrigation_pct"] + irrigation_change, 0, 100))
    return scenario
