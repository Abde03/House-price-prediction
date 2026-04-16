import argparse
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import encode_categorical_columns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"


MODEL_FEATURES = [
    "price",
    "rooms",
    "bathrooms",
    "area",
    "elevator",
    "floor",
    "terrace",
    "parking",
    "type",
    "city",
    "neighborhood",
]


@dataclass(frozen=True)
class CandidateModel:
    name: str
    builder: Callable[[], object]
    params: dict
    plot_prefix: str


def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Load cleaned dataset."""
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame):
    """Split dataframe into features X and target y."""
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a reduced set of columns to limit model complexity."""
    selected = [col for col in MODEL_FEATURES if col in df.columns]
    return df[selected].copy()


def split_without_feature_overlap(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split data so identical feature rows never appear in both train and test."""
    groups = pd.util.hash_pandas_object(X, index=False)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model with basic default params."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_extra_trees(X_train, y_train):
    """Train Extra Trees model with robust defaults for small tabular data."""
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_with_group_cv(model_builder, X: pd.DataFrame, y: pd.Series, n_splits: int = 10) -> dict:
    """Compute robust metrics across multiple group-aware splits."""
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    groups = pd.util.hash_pandas_object(X, index=False)

    maes, rmses, r2s = [], [], []

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        model = model_builder()
        model.fit(X_train_cv, y_train_cv)
        preds = model.predict(X_test_cv)
        m = evaluate_model(y_test_cv, preds)

        maes.append(m["mae"])
        rmses.append(m["rmse"])
        r2s.append(m["r2"])

    return {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_r2_mean": float(np.mean(r2s)),
        "cv_r2_std": float(np.std(r2s)),
    }


def log_model_to_mlflow(model_name: str, model, metrics: dict, params: dict):
    """Log model, parameters, and metrics to MLflow."""
    with mlflow.start_run(run_name=model_name):
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, artifact_path="model")


def save_model(model, output_path: str):
    """Save trained model locally."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_file)


def save_feature_schema(feature_columns, output_path: str = "models/feature_schema.json"):
    """Persist feature order for inference apps."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": list(feature_columns)}, f, ensure_ascii=False, indent=2)


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, output_path: str = "results/actual_vs_predicted.png"):
    """Plot actual vs predicted values."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Price', fontsize=12)
    ax.set_ylabel('Predicted Price', fontsize=12)
    ax.set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_feature_importance(model, feature_names, model_name: str, output_path: str = "results/feature_importance.png"):
    """Plot feature importance for tree-based models."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(indices)), importances[indices], alpha=0.7, edgecolor='k')
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title(f'{model_name}: Top 15 Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    return None


def plot_residuals(y_true, y_pred, model_name: str, output_path: str = "results/residuals.png"):
    """Plot residuals analysis."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Price', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title(f'{model_name}: Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Residuals distribution
    axes[1].hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'{model_name}: Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def build_model_candidates() -> list[CandidateModel]:
    """Return the models that are compared during training."""
    return [
        CandidateModel(
            name="LinearRegression",
            builder=lambda: LinearRegression(),
            params={"model": "LinearRegression"},
            plot_prefix="lr",
        ),
        CandidateModel(
            name="RandomForestRegressor (Robust)",
            builder=lambda: RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            params={
                "model": "RandomForestRegressor",
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
            plot_prefix="rf_robust",
        ),
        CandidateModel(
            name="ExtraTreesRegressor (Robust)",
            builder=lambda: ExtraTreesRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            params={
                "model": "ExtraTreesRegressor",
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
            plot_prefix="et_robust",
        ),
    ]


def log_candidate_run(candidate: CandidateModel, model, holdout_metrics: dict, cv_metrics: dict, X_train, y_test, preds, feature_names):
    """Log a single model experiment to MLflow and save artifacts."""
    with mlflow.start_run(run_name=candidate.name):
        mlflow.set_tag("pipeline_stage", "model_training")
        mlflow.set_tag("model_name", candidate.name)

        for param_name, param_value in candidate.params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("use_neighborhood", True)

        for metric_name, metric_value in holdout_metrics.items():
            mlflow.log_metric(f"holdout_{metric_name}", metric_value)

        for metric_name, metric_value in cv_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, artifact_path="model")

        actual_vs_pred_path = plot_actual_vs_predicted(
            y_test,
            preds,
            candidate.name,
            f"results/{candidate.plot_prefix}_actual_vs_pred.png",
        )
        mlflow.log_artifact(actual_vs_pred_path)

        residuals_path = plot_residuals(
            y_test,
            preds,
            candidate.name,
            f"results/{candidate.plot_prefix}_residuals.png",
        )
        mlflow.log_artifact(residuals_path)

        importance_path = plot_feature_importance(
            model,
            feature_names,
            candidate.name,
            f"results/{candidate.plot_prefix}_importance.png",
        )
        if importance_path:
            mlflow.log_artifact(importance_path)


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Convert training results into a sortable comparison table."""
    comparison_rows = []

    for result in results:
        row = {"model_name": result["model_name"]}
        row.update({f"holdout_{metric_name}": float(metric_value) for metric_name, metric_value in result["holdout_metrics"].items()})
        row.update({metric_name: float(metric_value) for metric_name, metric_value in result["cv_metrics"].items()})
        row.update(result["params"])
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    return comparison_df


def save_training_metadata(metadata: dict, output_path: Path = MODEL_METADATA_PATH):
    """Persist training metadata for the Streamlit app and future automation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def run_training_pipeline(
    cleaned_file: str = "data/processed/cleaned_data.csv",
    experiment_name: str = "Morocco_House_Price_Prediction",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Run preprocessing assumptions, compare models, and persist the best one."""
    mlflow.set_experiment(experiment_name)

    df = load_cleaned_data(cleaned_file)
    df = select_model_features(df)
    df_encoded = encode_categorical_columns(df)

    duplicate_rows = int(df_encoded.duplicated().sum())
    print(f"Duplicate rows in encoded data: {duplicate_rows}")
    print("Keeping duplicates but using group-aware splits to prevent leakage.")

    X, y = split_features_target(df_encoded)
    X_train, X_test, y_train, y_test = split_without_feature_overlap(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    feature_names = list(X_train.columns)
    save_feature_schema(feature_names, str(FEATURE_SCHEMA_PATH))

    results: list[dict] = []

    for candidate in build_model_candidates():
        print("\n" + "=" * 60)
        print(f"Training {candidate.name}...")
        print("=" * 60)

        model = candidate.builder()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        holdout_metrics = evaluate_model(y_test, preds)
        cv_metrics = evaluate_with_group_cv(candidate.builder, X, y)

        print(
            f"{candidate.name} - Holdout MAE: {holdout_metrics['mae']:.2f}, "
            f"RMSE: {holdout_metrics['rmse']:.2f}, R²: {holdout_metrics['r2']:.4f}"
        )
        print(
            f"{candidate.name} - CV mean R²: {cv_metrics['cv_r2_mean']:.4f} "
            f"± {cv_metrics['cv_r2_std']:.4f}"
        )

        log_candidate_run(candidate, model, holdout_metrics, cv_metrics, X_train, y_test, preds, feature_names)

        results.append(
            {
                "model_name": candidate.name,
                "model": model,
                "params": candidate.params,
                "holdout_metrics": holdout_metrics,
                "cv_metrics": cv_metrics,
            }
        )

    comparison_df = build_comparison_table(results)
    if comparison_df.empty:
        raise RuntimeError("No models were trained.")

    best_result = max(results, key=lambda item: item["cv_metrics"]["cv_r2_mean"])
    best_model = best_result["model"]
    best_model_name = best_result["model_name"]
    best_metrics = best_result["holdout_metrics"]
    best_cv_metrics = best_result["cv_metrics"]

    save_model(best_model, str(BEST_MODEL_PATH))
    comparison_path = Path(MODEL_COMPARISON_PATH)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)

    metadata = {
        "experiment_name": experiment_name,
        "cleaned_file": cleaned_file,
        "best_model_name": best_model_name,
        "best_model_path": str(BEST_MODEL_PATH),
        "best_metrics": {key: float(value) for key, value in best_metrics.items()},
        "best_cv_metrics": {key: float(value) for key, value in best_cv_metrics.items()},
        "feature_columns": feature_names,
        "feature_schema_path": str(FEATURE_SCHEMA_PATH),
        "comparison_path": str(comparison_path),
        "model_names": [result["model_name"] for result in results],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_training_metadata(metadata)

    with mlflow.start_run(run_name="TrainingSummary"):
        mlflow.set_tag("pipeline_stage", "summary")
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_param("candidate_model_count", len(results))
        mlflow.log_metric("best_holdout_r2", float(best_metrics["r2"]))
        mlflow.log_metric("best_cv_r2_mean", float(best_cv_metrics["cv_r2_mean"]))
        mlflow.log_metric("best_cv_r2_std", float(best_cv_metrics["cv_r2_std"]))
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(MODEL_METADATA_PATH))

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for _, row in comparison_df.iterrows():
        print(
            f"{row['model_name']:30s} | Holdout R²: {row['holdout_r2']:>8.4f} | "
            f"CV R²(mean±std): {row['cv_r2_mean']:.4f} +/- {row['cv_r2_std']:.4f}"
        )

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 60)
    print(f"MAE  : {best_metrics['mae']:.2f}")
    print(f"RMSE : {best_metrics['rmse']:.2f}")
    print(f"R²   : {best_metrics['r2']:.4f}")
    print(f"CV R² mean: {best_cv_metrics['cv_r2_mean']:.4f} (+/- {best_cv_metrics['cv_r2_std']:.4f})")
    print(f"\nFeatures used ({X_train.shape[1]}): {feature_names}")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Comparison saved to: {comparison_path}")
    print(f"Metadata saved to: {MODEL_METADATA_PATH}")

    return metadata


def parse_args():
    """Parse command-line options for automation."""
    parser = argparse.ArgumentParser(description="Train and compare house price models.")
    parser.add_argument("--cleaned-file", default="data/processed/cleaned_data.csv")
    parser.add_argument("--experiment-name", default="Morocco_House_Price_Prediction")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main(cleaned_file: str = "data/processed/cleaned_data.csv", experiment_name: str = "Morocco_House_Price_Prediction", test_size: float = 0.2, random_state: int = 42):
    """Train models and persist the best one."""
    return run_training_pipeline(
        cleaned_file=cleaned_file,
        experiment_name=experiment_name,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        cleaned_file=args.cleaned_file,
        experiment_name=args.experiment_name,
        test_size=args.test_size,
        random_state=args.random_state,
    )