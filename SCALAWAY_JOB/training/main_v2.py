"""
Job 3v2 – Training entry point for the Process API pipeline.

Trains all four soil texture targets (Clay, Silt, Sand, Coarse) in a single run
using the same model architecture as soil_texture_pipeline_v2.py, then exports
clean model artifacts ready for the inference worker.

Model artifacts layout (local):
    $MODEL_DIR/
    ├── clay.pkl, silt.pkl, sand.pkl, coarse.pkl   ← best model per target
    ├── features.json                               ← feature contract + filter hash
    ├── VERSION                                     ← "v2.0-processapi"
    └── training_report.json                        ← R2/RMSE/MAE per target

S3 layout (when --model-s3-prefix is set):
    s3://$BUCKET/$PREFIX$TIMESTAMP/clay.pkl  ...   ← versioned copy
    s3://$BUCKET/${PREFIX}latest/clay.pkl    ...   ← always overwritten

Environment variables:
    SCALEWAY_S3_ENDPOINT, SCALEWAY_S3_BUCKET, SCALEWAY_ACCESS_KEY, SCALEWAY_SECRET_KEY
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.config import Config as BotoConfig

from soil_texture_pipeline_v2 import (
    TEXTURE_FRACTIONS_DEFAULT,
    add_spatial_blocks,
    build_model_suite,
    ensure_dir,
    evaluate_texture_model,
    load_dataset,
    save_json,
)

log = logging.getLogger(__name__)

VERSION = "v2.0-processapi"

TARGETS = list(TEXTURE_FRACTIONS_DEFAULT)   # ["Clay", "Silt", "Sand", "Coarse"]

# Feature contract — must match process_api_extractor.py FEATURE_NAMES exactly.
FEATURE_NAMES = [
    "B02_median", "B03_median", "B04_median", "B05_median",
    "B06_median", "B07_median", "B08_median", "B8A_median",
    "B11_median", "B12_median",
    "NDVI", "NDMI",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process API training — all targets, clean model export")
    p.add_argument("--features",      default=os.getenv("FEATURES_PATH", "/data/features_v2.parquet"),
                   help="Feature parquet produced by feature-store (process_api mode)")
    p.add_argument("--model-dir",     default=os.getenv("MODEL_DIR", "/models"),
                   help="Directory for clean model artifacts (clay.pkl etc.)")
    p.add_argument("--filter-config", default="/data/filter_config.json",
                   help="Path to filter_config.json (hash embedded in features.json)")
    p.add_argument("--reports-dir",   default=os.getenv("REPORTS_DIR", "/reports"),
                   help="Directory for full CV reports (fold scores, metrics CSV, plots)")
    p.add_argument("--n-splits",       type=int, default=5)
    p.add_argument("--random-state",   type=int, default=42)
    p.add_argument("--model-s3-prefix", default=os.getenv("MODEL_S3_PREFIX", ""),
                   help="S3 prefix to upload model artifacts after training "
                        "(e.g. 'models/process_api/'). Empty string = skip upload.")
    return p.parse_args()


def _filter_hash(filter_config_path: str) -> str:
    with open(filter_config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

_ARTIFACT_FILES = [
    "clay.pkl", "silt.pkl", "sand.pkl", "coarse.pkl",
    "features.json", "VERSION", "training_report.json",
]

_CONTENT_TYPES = {
    ".pkl":  "application/octet-stream",
    ".json": "application/json",
    "":      "text/plain",   # VERSION has no extension
}


def _upload_models_to_s3(model_dir: Path, prefix: str, timestamp: str) -> None:
    """
    Upload model artifacts to S3 under two paths:
      {prefix}{timestamp}/  — immutable versioned copy
      {prefix}latest/       — always overwritten with the most recent run
    """
    endpoint  = os.environ.get("SCALEWAY_S3_ENDPOINT", "")
    bucket    = os.environ.get("SCALEWAY_S3_BUCKET", "")
    access    = os.environ.get("SCALEWAY_ACCESS_KEY", "")
    secret    = os.environ.get("SCALEWAY_SECRET_KEY", "")
    region    = os.environ.get("SCALEWAY_S3_REGION", "fr-par")

    if not all([endpoint, bucket, access, secret]):
        log.warning("S3 credentials incomplete — skipping model upload.")
        return

    if not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=BotoConfig(retries={"max_attempts": 6, "mode": "standard"}),
    )

    destinations = [f"{prefix}{timestamp}/", f"{prefix}latest/"]

    for dest in destinations:
        for filename in _ARTIFACT_FILES:
            local = model_dir / filename
            if not local.exists():
                log.debug("Skipping %s (not found locally).", filename)
                continue
            ext = local.suffix
            content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")
            key = f"{dest}{filename}"
            s3.upload_file(
                str(local), bucket, key,
                ExtraArgs={"ContentType": content_type},
            )
            log.debug("Uploaded → s3://%s/%s", bucket, key)

        log.info("Models uploaded → s3://%s/%s", bucket, dest)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    model_dir   = Path(args.model_dir)
    reports_dir = Path(args.reports_dir)
    ensure_dir(model_dir)
    ensure_dir(reports_dir)

    # --- Filter config hash ---
    f_hash = _filter_hash(args.filter_config)
    log.info("filter_config sha256=%s...", f_hash[:12])

    # --- Load features ---
    df = load_dataset(args.features)
    log.info("Loaded %d rows from %s", len(df), args.features)

    # Validate feature columns are present
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        sys.exit(f"Feature columns missing from parquet: {missing}. "
                 "Ensure feature-store ran in process_api mode.")

    # --- Spatial blocks for spatial CV ---
    group_col: Optional[str] = None
    try:
        df = add_spatial_blocks(df, lat_col="TH_LAT", lon_col="TH_LONG")
        group_col = "block_id"
        log.info("Spatial blocks created (20 km grid).")
    except Exception as exc:
        log.warning("Spatial blocks failed (%s). Spatial CV skipped.", exc)

    # --- Train ---
    suite = build_model_suite(args.random_state)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = reports_dir / f"run_{run_ts}"
    ensure_dir(run_dir)

    all_cv_rows: list[dict[str, Any]] = []
    best_pipes: dict[str, Any] = {}

    for target in TARGETS:
        if target not in df.columns or df[target].isna().all():
            log.warning("Skipping %s — no labels in parquet.", target)
            continue

        log.info("=== %s ===", target)
        target_rows: list[dict] = []
        candidate_pipes: dict = {}

        cv_modes = ["random"] + (["spatial"] if group_col else [])

        for ms in suite:
            for cv_mode in cv_modes:
                log.info("  model=%s config=%s cv=%s", ms.name, ms.config, cv_mode)
                try:
                    pipe, results, fold_df, X_used, _ = evaluate_texture_model(
                        df, target, ms.model,
                        drop_cols=[c for c in df.columns if c not in FEATURE_NAMES + [target]
                                   and c not in TARGETS],
                        group_col=group_col,
                        cv_mode=cv_mode,
                        n_splits=args.n_splits,
                        random_state=args.random_state,
                        scale_numeric=ms.scale_numeric,
                    )
                    results["model"]   = ms.name
                    results["config"]  = ms.config
                    results["cv_mode"] = cv_mode
                    target_rows.append(results)

                    fold_df.to_csv(
                        run_dir / f"fold_scores__{target}__{ms.name}__{ms.config}__{results['cv_type']}.csv",
                        index=False,
                    )
                    save_json(
                        run_dir / f"metrics__{target}__{ms.name}__{ms.config}__{results['cv_type']}.json",
                        results,
                    )
                    if cv_mode == "random":
                        candidate_pipes[(ms.name, ms.config)] = pipe

                except Exception as exc:
                    log.error("  FAILED %s/%s/%s: %s", target, ms.name, ms.config, exc)

        all_cv_rows.extend(target_rows)

        # Select best model by Random CV R2
        random_rows = [r for r in target_rows if r.get("cv_type") == "RandomKFold"]
        if not random_rows:
            log.warning("No successful runs for %s.", target)
            continue

        best_row = max(random_rows, key=lambda r: r["cv_r2_mean"])
        best_key = (best_row["model"], best_row["config"])
        best_pipe = candidate_pipes.get(best_key)

        if best_pipe is not None:
            pkl_path = model_dir / f"{target.lower()}.pkl"
            joblib.dump(best_pipe, pkl_path)
            best_pipes[target] = best_pipe
            log.info("  Best: %s/%s  R2=%.3f → %s",
                     best_row["model"], best_row["config"], best_row["cv_r2_mean"], pkl_path)

    # --- CV summary ---
    if all_cv_rows:
        summary_df = pd.DataFrame(all_cv_rows).sort_values(
            ["target", "cv_type", "cv_r2_mean"], ascending=[True, True, False]
        )
        summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)

    # --- Model artifacts ---
    features_json = {
        "feature_names":      FEATURE_NAMES,
        "n_features":         len(FEATURE_NAMES),
        "source":             "process_api",
        "spatial_aggregation":"center_pixel",
        "temporal_aggregation":"median",
        "filter_config_hash": f_hash,
    }
    save_json(model_dir / "features.json", features_json)
    (model_dir / "VERSION").write_text(VERSION, encoding="utf-8")

    training_report: dict[str, Any] = {
        "version":            VERSION,
        "timestamp":          datetime.now().isoformat(),
        "filter_config_hash": f_hash,
        "n_rows":             int(len(df)),
        "n_splits":           args.n_splits,
        "targets":            {},
    }
    for target in TARGETS:
        rows = [r for r in all_cv_rows if r.get("target") == target and r.get("cv_type") == "RandomKFold"]
        if rows:
            best = max(rows, key=lambda r: r["cv_r2_mean"])
            training_report["targets"][target] = {
                "best_model":   best["model"],
                "best_config":  best["config"],
                "cv_r2_mean":   best["cv_r2_mean"],
                "cv_r2_std":    best["cv_r2_std"],
                "cv_rmse_mean": best["cv_rmse_mean"],
                "cv_mae_mean":  best["cv_mae_mean"],
            }
    save_json(model_dir / "training_report.json", training_report)

    log.info("Artifacts written to %s/", model_dir)
    log.info("  %s", ", ".join(f"{t.lower()}.pkl" for t in TARGETS if t in best_pipes))
    log.info("  features.json  filter_config_hash=%s...", f_hash[:12])
    log.info("  VERSION=%s", VERSION)

    # --- Checkpoint test ---
    _checkpoint(model_dir, best_pipes)

    # --- Upload to S3 ---
    if args.model_s3_prefix:
        log.info("Uploading model artifacts to S3 (prefix=%s)...", args.model_s3_prefix)
        _upload_models_to_s3(model_dir, args.model_s3_prefix, run_ts)
    else:
        log.info("--model-s3-prefix not set, skipping S3 upload.")


def _checkpoint(model_dir: Path, best_pipes: dict) -> None:
    features_json = json.loads((model_dir / "features.json").read_text())
    assert len(features_json["feature_names"]) == features_json["n_features"]

    X_dummy = pd.DataFrame(
        np.random.rand(5, len(FEATURE_NAMES)),
        columns=FEATURE_NAMES,
    )

    all_ok = True
    for target in TARGETS:
        pkl = model_dir / f"{target.lower()}.pkl"
        if not pkl.exists():
            log.warning("Checkpoint: %s.pkl not found.", target.lower())
            all_ok = False
            continue
        model = joblib.load(pkl)
        preds = model.predict(X_dummy)
        assert preds.shape == (5,), f"{target}: shape {preds.shape}"
        assert preds.min() >= -20 and preds.max() <= 120, \
            f"{target}: predictions out of plausible range"

    if len(best_pipes) == 4:
        total = sum(p.predict(X_dummy)[0] for p in best_pipes.values())
        log.info("Checkpoint: target sum for random sample = %.1f (expect ~100)", total)

    if all_ok:
        log.info("Training checkpoint: PASSED")
    else:
        log.warning("Training checkpoint: some models missing.")


if __name__ == "__main__":
    main()
