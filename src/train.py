"""Command-line entry point for analysis and model training."""

from .config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from .data_processing import load_data
from .eda import write_eda_reports
from .feature_engineering import add_agronomic_features
from .model_training import run_training


def main() -> None:
    data = load_data(RAW_DATA_PATH)
    write_eda_reports(data)
    prepared = add_agronomic_features(data)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PROCESSED_DATA_PATH, index=False)
    _, comparison, metadata = run_training(data)
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {metadata['selected_model']}")


if __name__ == "__main__":
    main()
