"""Model loading and prediction helpers shared by Streamlit pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import METADATA_PATH, MODEL_INPUT_FEATURES, MODEL_PATH


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Trained model is missing. Run: python -m src.train")
    return joblib.load(path)


def load_metadata(path: str | Path = METADATA_PATH) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def predict_yield(model: Pipeline, record: pd.Series | dict[str, Any]) -> float:
    row = pd.DataFrame([dict(record)])[MODEL_INPUT_FEATURES]
    return max(0.0, float(model.predict(row)[0]))
