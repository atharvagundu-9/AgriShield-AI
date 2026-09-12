"""Scientifically motivated, deterministic feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import CROP_REQUIREMENTS


def _clip01(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.clip(values, 0.0, 1.0)


def add_agronomic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pre-harvest features derived only from available input conditions.

    The formulas are transparent indices, not causal or agronomic guarantees. They
    are deliberately bounded so unusual input values do not dominate a model.
    """
    data = frame.copy()
    crop = data["crop"].astype(str)
    rain_need = crop.map({k: v["rainfall"] for k, v in CROP_REQUIREMENTS.items()}).fillna(800.0)
    heat_limit = crop.map({k: v["heat_threshold"] for k, v in CROP_REQUIREMENTS.items()}).fillna(34.0)
    irrigation_need = crop.map({k: v["irrigation"] for k, v in CROP_REQUIREMENTS.items()}).fillna(50.0)

    data["rainfall_requirement_ratio"] = (data["annual_rainfall_mm"] / rain_need).clip(0, 3)
    data["rainfall_deficit_index"] = _clip01((-data["rainfall_anomaly_pct"]) / 50.0)
    heat_excess = (data["max_temperature_c"] - heat_limit).clip(lower=0)
    data["heat_stress_index"] = _clip01(0.55 * heat_excess / 8.0 + 0.45 * data["hot_days"] / 70.0)
    data["drought_stress_index"] = _clip01(
        0.35 * data["rainfall_deficit_index"]
        + 0.30 * data["consecutive_dry_days"] / 70.0
        + 0.25 * (1 - data["soil_moisture_pct"] / 75.0)
        + 0.10 * (1 - data["irrigation_pct"] / 100.0)
    )
    npk = (
        _clip01(data["nitrogen_kg_ha"] / 400.0)
        + _clip01(data["phosphorus_kg_ha"] / 50.0)
        + _clip01(data["potassium_kg_ha"] / 500.0)
    ) / 3.0
    ph_score = _clip01(1 - np.abs(data["soil_ph"] - 6.8) / 2.0)
    data["soil_fertility_index"] = _clip01(0.75 * npk + 0.25 * ph_score)
    data["irrigation_adequacy"] = (data["irrigation_pct"] / irrigation_need).clip(0, 2)
    data["extreme_weather_days"] = (
        data["hot_days"] + data["consecutive_dry_days"] + 1.5 * data["heavy_rain_days"]
    )
    data["rainfall_soil_interaction"] = (
        data["rainfall_requirement_ratio"] * data["soil_moisture_pct"] / 75.0
    ).clip(0, 3)
    data["temperature_rainfall_interaction"] = (
        data["temperature_anomaly_c"] * data["rainfall_anomaly_pct"] / 100.0
    ).clip(-3, 3)
    rain_balance = _clip01(1 - np.abs(data["rainfall_requirement_ratio"] - 1.0) / 1.5)
    data["growing_condition_score"] = 100 * _clip01(
        0.25 * rain_balance
        + 0.20 * (data["soil_moisture_pct"] / 75.0)
        + 0.20 * data["soil_fertility_index"]
        + 0.20 * _clip01(data["irrigation_adequacy"])
        + 0.15 * (1 - data["heat_stress_index"])
    )
    return data


class AgronomicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible wrapper around :func:`add_agronomic_features`."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AgronomicFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AgronomicFeatureEngineer expects a pandas DataFrame.")
        return add_agronomic_features(X)
