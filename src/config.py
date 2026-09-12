"""Project-wide paths, schema, and reproducibility settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "maharashtra_agri_yield_climate_risk_10000.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "agriculture_prepared.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "yield_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessing.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
TARGET = "yield_tonnes_per_ha"

# These fields are identifiers, constants, post-harvest values, or supplied risk outputs.
# They remain available to analytics but are excluded from yield-model inputs.
EXCLUDED_MODEL_COLUMNS = [
    "record_id",
    "state",
    "production_tonnes",
    "drought_risk_score",
    "drought_risk",
    "heat_risk_score",
    "heat_risk",
    "flood_risk_score",
    "flood_risk",
    "overall_climate_risk_score",
    "overall_climate_risk",
]

CATEGORICAL_FEATURES = ["district", "agro_climatic_region", "crop", "season"]
BASE_NUMERIC_FEATURES = [
    "year",
    "annual_rainfall_mm",
    "rainfall_anomaly_pct",
    "avg_temperature_c",
    "max_temperature_c",
    "min_temperature_c",
    "temperature_anomaly_c",
    "humidity_pct",
    "soil_moisture_pct",
    "soil_ph",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
    "irrigation_pct",
    "hot_days",
    "consecutive_dry_days",
    "heavy_rain_days",
    "cultivated_area_ha",
]

ENGINEERED_NUMERIC_FEATURES = [
    "rainfall_requirement_ratio",
    "rainfall_deficit_index",
    "heat_stress_index",
    "drought_stress_index",
    "soil_fertility_index",
    "irrigation_adequacy",
    "extreme_weather_days",
    "rainfall_soil_interaction",
    "temperature_rainfall_interaction",
    "growing_condition_score",
]

MODEL_INPUT_FEATURES = CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
MODEL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

CROP_REQUIREMENTS = {
    "Cotton": {"rainfall": 800.0, "heat_threshold": 35.0, "irrigation": 45.0},
    "Maize": {"rainfall": 650.0, "heat_threshold": 34.0, "irrigation": 45.0},
    "Rice": {"rainfall": 1200.0, "heat_threshold": 35.0, "irrigation": 65.0},
    "Soybean": {"rainfall": 750.0, "heat_threshold": 34.0, "irrigation": 40.0},
    "Sugarcane": {"rainfall": 1500.0, "heat_threshold": 36.0, "irrigation": 80.0},
    "Wheat": {"rainfall": 500.0, "heat_threshold": 30.0, "irrigation": 55.0},
}
