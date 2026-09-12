"""Crop ranking normalized against each crop's historical yield distribution."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from .climate_risk import calculate_climate_risk
from .prediction import predict_yield


def recommend_crops(
    model: Pipeline,
    base_record: pd.Series,
    history: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for crop in sorted(history["crop"].dropna().unique()):
        candidate = base_record.copy()
        candidate["crop"] = crop
        crop_history = history.loc[history["crop"] == crop]
        if not crop_history.empty:
            candidate["season"] = crop_history["season"].mode().iloc[0]
        predicted = predict_yield(model, candidate)
        risk = calculate_climate_risk(candidate)
        historical_yield = crop_history["yield_tonnes_per_ha"]
        percentile = 100 * float((historical_yield <= predicted).mean()) if len(historical_yield) else 50.0
        suitability = 0.72 * percentile + 0.28 * (100 - float(risk["overall_score"]))
        rows.append(
            {
                "Crop": crop,
                "Predicted Yield (t/ha)": predicted,
                "Crop-relative Performance": percentile,
                "Climate Risk": float(risk["overall_score"]),
                "Risk Category": risk["overall_category"],
                "Suitability Score": suitability,
            }
        )
    result = pd.DataFrame(rows).sort_values("Suitability Score", ascending=False).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result
