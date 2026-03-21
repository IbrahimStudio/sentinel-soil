"""
Job 3 – Training entry point.

Loads the feature table produced by Job 2, trains soil texture regressors,
and writes metrics + model artefacts to the reports directory.

Environment variables:
    FEATURES_PATH    Path to feature parquet/xlsx (default: /data/features.parquet)
    TARGET           Soil property to predict: Clay | Silt | Sand | Coarse (default: Clay)
    REPORTS_DIR      Output directory for reports (default: reports/)
    PIPELINE_VERSION v1 | v2 (default: v2)
"""
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train soil texture regressors")
    p.add_argument("--features", default=os.getenv("FEATURES_PATH", "/data/features.parquet"))
    p.add_argument("--target", default=os.getenv("TARGET", "Clay"),
                   choices=["Clay", "Silt", "Sand", "Coarse"])
    p.add_argument("--reports-dir", default=os.getenv("REPORTS_DIR", "reports"))
    p.add_argument("--pipeline", default=os.getenv("PIPELINE_VERSION", "v2"),
                   choices=["v1", "v2"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.pipeline == "v2":
        from soil_texture_pipeline_v2 import run_pipeline
    else:
        from soil_texture_pipeline import run_pipeline

    print(f"Training pipeline={args.pipeline} target={args.target}")
    print(f"  features:    {args.features}")
    print(f"  reports dir: {args.reports_dir}")

    run_pipeline(
        features_path=args.features,
        target=args.target,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
