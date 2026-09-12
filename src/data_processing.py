"""Dataset loading, validation, preprocessing, and time-aware splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .config import (
    CATEGORICAL_FEATURES,
    MODEL_INPUT_FEATURES,
    MODEL_NUMERIC_FEATURES,
    TARGET,
)


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    required = set(MODEL_INPUT_FEATURES + [TARGET])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return data


def model_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data[MODEL_INPUT_FEATURES].copy(), data[TARGET].copy()


def time_aware_split(
    data: pd.DataFrame,
    validation_years: tuple[int, ...] = (2021, 2022),
    test_years: tuple[int, ...] = (2023, 2024, 2025),
) -> DatasetSplits:
    train_mask = ~data["year"].isin(validation_years + test_years)
    validation_mask = data["year"].isin(validation_years)
    test_mask = data["year"].isin(test_years)
    if not (train_mask.any() and validation_mask.any() and test_mask.any()):
        raise ValueError("Time split produced an empty partition. Check year coverage.")
    X, y = model_frame(data)
    return DatasetSplits(
        X.loc[train_mask], y.loc[train_mask],
        X.loc[validation_mask], y.loc[validation_mask],
        X.loc[test_mask], y.loc[test_mask],
    )


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, MODEL_NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
