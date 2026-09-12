import unittest

import numpy as np
import pandas as pd

from src.climate_risk import apply_scenario, calculate_climate_risk
from src.config import ENGINEERED_NUMERIC_FEATURES, MODEL_INPUT_FEATURES, RAW_DATA_PATH
from src.crop_recommendation import recommend_crops
from src.data_processing import load_data, time_aware_split
from src.feature_engineering import add_agronomic_features
from src.prediction import load_model, predict_yield


class AgriShieldCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data(RAW_DATA_PATH)
        cls.model = load_model()
        cls.row = cls.data.iloc[0].copy()

    def test_time_split_has_no_year_overlap(self):
        split = time_aware_split(self.data)
        self.assertLessEqual(split.X_train["year"].max(), 2020)
        self.assertEqual(set(split.X_validation["year"].unique()), {2021, 2022})
        self.assertEqual(set(split.X_test["year"].unique()), {2023, 2024, 2025})

    def test_engineered_features_are_finite(self):
        result = add_agronomic_features(self.data.head(20)[MODEL_INPUT_FEATURES])
        self.assertTrue(set(ENGINEERED_NUMERIC_FEATURES).issubset(result.columns))
        self.assertTrue(np.isfinite(result[ENGINEERED_NUMERIC_FEATURES].to_numpy()).all())

    def test_risk_scores_are_bounded(self):
        risk = calculate_climate_risk(self.row)
        for name in ("drought", "heat", "flood", "overall"):
            self.assertGreaterEqual(risk[f"{name}_score"], 0)
            self.assertLessEqual(risk[f"{name}_score"], 100)

    def test_scenario_changes_inputs_without_mutating_baseline(self):
        baseline_rain = float(self.row["annual_rainfall_mm"])
        changed = apply_scenario(self.row, temperature_change_c=2, rainfall_change_pct=-20)
        self.assertAlmostEqual(changed["annual_rainfall_mm"], baseline_rain * 0.8)
        self.assertEqual(float(self.row["annual_rainfall_mm"]), baseline_rain)

    def test_saved_model_predicts_nonnegative_value(self):
        prediction = predict_yield(self.model, self.row)
        self.assertTrue(np.isfinite(prediction))
        self.assertGreaterEqual(prediction, 0)

    def test_crop_ranking_is_complete_and_sorted(self):
        result = recommend_crops(self.model, self.row, self.data)
        self.assertEqual(len(result), self.data["crop"].nunique())
        self.assertEqual(result["Rank"].tolist(), list(range(1, len(result) + 1)))
        self.assertTrue(result["Suitability Score"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
