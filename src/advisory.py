"""Plain-language advisory derived directly from modeled conditions."""

from __future__ import annotations

import pandas as pd


DISCLAIMER = "Decision-support output — not a substitute for local agricultural expert advice."


def generate_advisory(record: pd.Series | dict, risk: dict, yield_change_pct: float | None = None) -> str:
    r = record if isinstance(record, dict) else record.to_dict()
    crop = r.get("crop", "The selected crop")
    factors: list[str] = []
    actions: list[str] = []
    if float(r["rainfall_anomaly_pct"]) < -15:
        factors.append("below-normal rainfall")
        actions.append("review irrigation scheduling and moisture-conservation options")
    if float(r["soil_moisture_pct"]) < 35:
        factors.append("low soil moisture")
        actions.append("verify field moisture before making irrigation decisions")
    if float(r["temperature_anomaly_c"]) > 1.2 or float(r["hot_days"]) > 35:
        factors.append("elevated heat exposure")
        actions.append("consider locally validated heat-stress mitigation")
    if float(r["heavy_rain_days"]) > 14 or float(r["rainfall_anomaly_pct"]) > 30:
        factors.append("heavy-rain exposure")
        actions.append("check drainage and waterlogging risk")
    if float(r["irrigation_pct"]) < 30:
        factors.append("limited irrigation coverage")
    if not factors:
        factors.append("broadly balanced modeled growing conditions")
        actions.append("continue monitoring rainfall, heat, and soil moisture")

    movement = ""
    if yield_change_pct is not None:
        direction = "decrease" if yield_change_pct < 0 else "increase"
        movement = f" The scenario produces a modeled yield {direction} of {abs(yield_change_pct):.1f}%."
    factor_text = ", ".join(factors[:-1]) + (" and " if len(factors) > 1 else "") + factors[-1]
    action_text = "; ".join(dict.fromkeys(actions))
    return (
        f"{crop} has {str(risk['drought_category']).lower()} drought risk, "
        f"{str(risk['heat_category']).lower()} heat risk, and {str(risk['flood_category']).lower()} flood risk. "
        f"The main modeled signals are {factor_text}.{movement} A practical next step is to {action_text}."
    )
