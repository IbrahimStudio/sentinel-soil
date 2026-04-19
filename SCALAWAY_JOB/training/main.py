"""
Job 3 – Training entry point.

Loads the feature table produced by Job 2, trains soil texture regressors,
and writes metrics + model artefacts to the reports directory.

Environment variables:
    FEATURES_PATH    Path to feature parquet/xlsx (default: /data/features.parquet)
    TARGET           Space-separated targets or "all" (default: all four fractions)
    REPORTS_DIR      Output directory for reports (default: reports/)
    PIPELINE_VERSION v1 | v2 (default: v2)
"""
from __future__ import annotations

import argparse
import os
from typing import List

ALL_TARGETS: List[str] = ["Clay", "Silt", "Sand", "Coarse"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train soil texture regressors")
    p.add_argument("--features", default=os.getenv("FEATURES_PATH", "/data/features.parquet"))
    p.add_argument(
        "--target",
        nargs="+",
        default=os.getenv("TARGET", "all").split(),
        choices=[*ALL_TARGETS, "all"],
        metavar="TARGET",
        help="One or more of Clay Silt Sand Coarse, or 'all' (default: all)",
    )
    p.add_argument("--reports-dir", default=os.getenv("REPORTS_DIR", "reports"))
    p.add_argument("--pipeline", default=os.getenv("PIPELINE_VERSION", "v2"),
                   choices=["v1", "v2"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    targets = ALL_TARGETS if "all" in args.target else args.target

    if args.pipeline == "v2":
        from soil_texture_pipeline_v2 import run_pipeline
    else:
        from soil_texture_pipeline import run_pipeline

    print(f"Training pipeline={args.pipeline} targets={targets}")
    print(f"  features:    {args.features}")
    print(f"  reports dir: {args.reports_dir}")

    for target in targets:
        print(f"\n{'='*60}")
        print(f"  Target: {target}")
        print(f"{'='*60}")
        run_pipeline(
            features_path=args.features,
            target=target,
            reports_dir=args.reports_dir,
        )


if __name__ == "__main__":
    main()
