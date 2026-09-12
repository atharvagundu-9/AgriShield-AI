"""Technically correct explanations on the model's transformed feature space."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def transformed_data(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    engineered = model.named_steps["features"].transform(X)
    preprocessor = model.named_steps["preprocessor"]
    matrix = preprocessor.transform(engineered)
    names = preprocessor.get_feature_names_out()
    return np.asarray(matrix), np.asarray(names)


def local_explanation(
    model: Pipeline,
    row: pd.DataFrame,
    background: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    """Calculate SHAP values after preprocessing; fall back to signed contributions."""
    sample_matrix, names = transformed_data(model, row)
    background_matrix, _ = transformed_data(model, background.sample(min(100, len(background)), random_state=42))
    estimator = model.named_steps["model"]
    try:
        import shap

        explainer = shap.Explainer(estimator, background_matrix)
        values = np.asarray(explainer(sample_matrix).values)[0]
    except Exception:
        if hasattr(estimator, "feature_importances_"):
            center = np.nanmedian(background_matrix, axis=0)
            values = (sample_matrix[0] - center) * np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            values = sample_matrix[0] * np.ravel(estimator.coef_)
        else:
            values = np.zeros(len(names))
    result = pd.DataFrame({"Feature": names, "Contribution": values})
    result["Absolute"] = result["Contribution"].abs()
    return result.nlargest(top_n, "Absolute").drop(columns="Absolute").sort_values("Contribution")


def global_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> pd.DataFrame:
    sample_n = min(600, len(X))
    sample = X.sample(sample_n, random_state=42)
    target = y.loc[sample.index]
    result = permutation_importance(
        model, sample, target, scoring="neg_mean_absolute_error", n_repeats=3, random_state=42, n_jobs=-1
    )
    return (
        pd.DataFrame({"Feature": X.columns, "Importance": result.importances_mean})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )
