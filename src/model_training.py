"""Train, compare, evaluate, and persist yield-regression pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from .data_processing import build_preprocessor, model_frame, time_aware_split
from .feature_engineering import AgronomicFeatureEngineer


def _models() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=260,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=260,
            learning_rate=0.045,
            max_depth=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBRegressor

        candidates["XGBoost"] = XGBRegressor(
            n_estimators=450,
            learning_rate=0.045,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except ImportError:
        pass
    return candidates


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("features", AgronomicFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    truth = np.asarray(y_true, dtype=float)
    meaningful = truth >= 0.10
    mape = np.mean(np.abs((truth[meaningful] - prediction[meaningful]) / truth[meaningful])) * 100
    return {
        "MAE": float(mean_absolute_error(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "R2": float(r2_score(truth, prediction)),
        "MAPE_pct": float(mape),
    }


def train_and_compare(data: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    """Use 2001–2020 for training, 2021–2022 validation, and 2023–2025 test."""
    split = time_aware_split(data)
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, estimator in _models().items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(split.X_train, split.y_train)
        fitted[name] = pipeline
        validation = regression_metrics(split.y_validation, pipeline.predict(split.X_validation))
        test = regression_metrics(split.y_test, pipeline.predict(split.X_test))
        rows.append(
            {
                "Model": name,
                **{f"Validation {key}": value for key, value in validation.items()},
                **{f"Test {key}": value for key, value in test.items()},
            }
        )

    comparison = pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)
    # Select on validation performance so the newest years remain an honest final test.
    best_name = str(comparison.sort_values("Validation RMSE").iloc[0]["Model"])

    # Refit the chosen algorithm on all pre-test observations after comparison.
    pretest_mask = data["year"] <= 2022
    X_all, y_all = model_frame(data)
    final_pipeline = make_pipeline(_models()[best_name])
    final_pipeline.fit(X_all.loc[pretest_mask], y_all.loc[pretest_mask])

    # A random split is reported only as a diagnostic comparison, never used for selection.
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=data["crop"]
    )
    random_pipeline = make_pipeline(_models()[best_name])
    random_pipeline.fit(X_train_r, y_train_r)
    random_metrics = regression_metrics(y_test_r, random_pipeline.predict(X_test_r))

    metadata = {
        "selected_model": best_name,
        "target": TARGET,
        "selection_note": "Selected by 2021–2022 validation RMSE; 2023–2025 remained an untouched final test.",
        "time_split": {
            "train": "2001–2020",
            "validation": "2021–2022",
            "test": "2023–2025",
            "train_rows": len(split.X_train),
            "validation_rows": len(split.X_validation),
            "test_rows": len(split.X_test),
        },
        "random_split_diagnostic": random_metrics,
        "test_metrics": comparison.loc[comparison["Model"] == best_name].iloc[0].drop(labels=["Model"]).to_dict(),
        "leakage_exclusions": [
            "production_tonnes (post-harvest arithmetic derivative of yield × area)",
            "all supplied climate-risk scores and labels (derived analytics kept separate)",
            "record_id and constant state",
        ],
    }
    return final_pipeline, comparison, metadata


def save_artifacts(
    pipeline: Pipeline,
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
    model_path: Path = MODEL_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    joblib.dump(Pipeline(pipeline.steps[:-1]), PREPROCESSOR_PATH, compress=3)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_training(data: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    pipeline, comparison, metadata = train_and_compare(data)
    save_artifacts(pipeline, comparison, metadata)
    return pipeline, comparison, metadata
