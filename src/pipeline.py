import argparse
from pathlib import Path

from preprocess import preprocess_data
from train import run_training_pipeline


def parse_args():
    """Parse command-line options for the end-to-end pipeline."""
    parser = argparse.ArgumentParser(description="Run preprocessing and training end to end.")
    parser.add_argument("--raw-file", default="data/raw/houses_data_eng.csv")
    parser.add_argument("--cleaned-file", default="data/processed/cleaned_data.csv")
    parser.add_argument("--experiment-name", default="Morocco_House_Price_Prediction")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    """Preprocess raw data, train models, and persist the best artifact."""
    args = parse_args()

    raw_path = Path(args.raw_file)
    cleaned_path = Path(args.cleaned_file)

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    print(f"Raw file: {raw_path}")
    print(f"Cleaned file: {cleaned_path}")
    preprocess_data(str(raw_path), str(cleaned_path))

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    summary = run_training_pipeline(
        cleaned_file=str(cleaned_path),
        experiment_name=args.experiment_name,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best model: {summary['best_model_name']}")
    print(f"Model file: {summary['best_model_path']}")
    print(f"Comparison file: {summary['comparison_path']}")
    print(f"Metadata file: models/model_metadata.json")


if __name__ == "__main__":
    main()
