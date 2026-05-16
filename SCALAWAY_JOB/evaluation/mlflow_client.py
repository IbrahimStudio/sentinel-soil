from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from .config import EvaluationConfig


def setup(cfg: EvaluationConfig) -> str:
    """Configure MLflow env, create experiment if needed. Returns experiment_id."""
    if cfg.s3_endpoint_url:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", cfg.s3_endpoint_url)

    # Map Scaleway credentials to the AWS env vars that boto3/MLflow expect.
    # Existing AWS_* vars are not overwritten (explicit beats implicit).
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        key = os.environ.get("SCALEWAY_ACCESS_KEY")
        if key:
            os.environ["AWS_ACCESS_KEY_ID"] = key
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        secret = os.environ.get("SCALEWAY_SECRET_KEY")
        if secret:
            os.environ["AWS_SECRET_ACCESS_KEY"] = secret

    mlflow.set_tracking_uri(cfg.tracking_uri)

    exp = mlflow.get_experiment_by_name(cfg.experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            cfg.experiment_name,
            artifact_location=cfg.artifact_root,
        )
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(cfg.experiment_name)
    return exp_id


def log_training_run(
    *,
    target: str,
    model_name: str,
    model_config: str,
    results: dict[str, Any],
    run_batch: str,
    feature_set: str,
    random_state: int,
    block_size_m: int,
    model: Any,
    fold_csv: Path | None = None,
    importance_csv: Path | None = None,
    shap_bar_png: Path | None = None,
    shap_dot_png: Path | None = None,
) -> str:
    """Log one (target × model × config × cv_type) training run. Returns MLflow run_id."""
    run_name = f"{target}__{model_name}__{model_config}__{results['cv_type']}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "target": target,
            "model": model_name,
            "config": model_config,
            "cv_type": results["cv_type"],
            "run_batch": run_batch,
            "feature_set": feature_set,
        })

        params: dict[str, Any] = {
            "n_splits": results["n_splits"],
            "n_samples": results["n_samples"],
            "n_features": results["n_features"],
            "random_state": random_state,
            "block_size_m": block_size_m,
        }
        for attr in (
            "n_estimators", "learning_rate", "min_samples_leaf",
            "max_features", "max_iter", "alpha", "l1_ratio",
        ):
            v = getattr(model, attr, None)
            if v is not None:
                params[attr] = v
        mlflow.log_params(params)

        mlflow.log_metrics({
            "cv_r2_mean": results["cv_r2_mean"],
            "cv_r2_std": results["cv_r2_std"],
            "cv_rmse_mean": results["cv_rmse_mean"],
            "cv_rmse_std": results["cv_rmse_std"],
            "cv_mae_mean": results["cv_mae_mean"],
            "cv_mae_std": results["cv_mae_std"],
        })

        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"  [MLflow] model artifact skipped: {e}")

        for path in (fold_csv, importance_csv, shap_bar_png, shap_dot_png):
            if path and path.exists():
                mlflow.log_artifact(str(path))

        return run.info.run_id
