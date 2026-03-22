from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict
import os

from sh_pipeline.models import parse_job
from sh_pipeline.paths import build_job_paths
from sh_pipeline.storage import storage_from_env
from sh_pipeline.logging_utils import setup_logger

from domain.extract_time_series import extract_one
from domain.bare_soil_features import compute_baresoil_features_from_ts_root


def run_one_job(payload: Dict[str, Any], *, config_path: str = "configs/dev.yaml") -> Dict[str, Any]:
    job = parse_job(payload)
    paths = build_job_paths(job.job_id)

    # Ensure local dirs exist
    paths.local_root.mkdir(parents=True, exist_ok=True)
    paths.local_logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir=paths.local_logs_dir, name=f"job_{job.job_id}")

    # Pass logger so upload actions get logged too
    storage = storage_from_env(logger=logger)

    logger.info(
        f"Starting job_id={job.job_id} point_id={job.point_id} "
        f"({job.lat},{job.lon}) survey={job.survey_date} "
        f"range=({getattr(job,'start_date',None)},{getattr(job,'end_date',None)}) "
        f"window={job.window.w}x{job.window.h} "
        f"ndvi<{job.ndvi_threshold} min_obs={job.min_obs}"
    )


    try:
        # Clean local folders if rerun (leave logs)
        for sub in [paths.local_intermediate_root, paths.local_features_root]:
            if sub.exists():
                shutil.rmtree(sub, ignore_errors=True)
            sub.mkdir(parents=True, exist_ok=True)

        # 1) EXTRACT locally -> writes under paths.local_intermediate_root/<point_id>/grid.../survey...
        ts_root = extract_one(
            lat=job.lat,
            lon=job.lon,
            survey_date=job.survey_date,
            window_days=job.window_days,
            start_date=getattr(job, "start_date", None),
            end_date=getattr(job, "end_date", None),
            pixel_window_w=job.window.w,
            pixel_window_h=job.window.h,
            res_m=job.res_m,
            config_path=config_path,
            point_id=job.point_id,
            max_cloud_coverage=job.max_cloud_coverage,
            mosaicking_order=job.mosaicking_order,
            out_root=paths.local_intermediate_root,
            logger=logger,
        )


        # Upload intermediate
        # NOTE: uploading ts_root (not the whole local_intermediate_root) keeps remote keys tidy
        storage.upload_tree(ts_root, paths.obj_intermediate_prefix)

        # 2) FEATURES locally
        feat_root, seed_summary_df = compute_baresoil_features_from_ts_root(
            ts_root=ts_root,
            seed_id=job.point_id,
            lat=job.lat,
            lon=job.lon,
            ndvi_threshold=job.ndvi_threshold,
            min_obs=job.min_obs,
            feat_out_root=paths.local_features_root,
            logger=logger,
        )

        # Upload features
        storage.upload_tree(feat_root, paths.obj_features_prefix)

        # Upload logs
        storage.upload_tree(paths.local_logs_dir, paths.obj_logs_prefix, include_globs=["*.log"])

        # Manifest (SUCCESS)
        manifest = {
            "job_id": job.job_id,
            "point_id": job.point_id,
            "status": "SUCCESS",
            "ndvi_threshold": job.ndvi_threshold,
            "min_obs": job.min_obs,
            "window_w": job.window.w,
            "window_h": job.window.h,
            "window_days": job.window_days,
            "res_m": job.res_m,
            "max_cloud_coverage": job.max_cloud_coverage,
            "mosaicking_order": job.mosaicking_order,
            "intermediate_prefix": paths.obj_intermediate_prefix,
            "features_prefix": paths.obj_features_prefix,
            "logs_prefix": paths.obj_logs_prefix,
        }
        storage.put_text(
            paths.obj_manifest_key,
            json.dumps(manifest, indent=2),
            content_type="application/json",
        )

        logger.info("Job completed successfully")
        return manifest

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Job failed")

        # Always upload logs on failure
        try:
            storage.upload_tree(paths.local_logs_dir, paths.obj_logs_prefix, include_globs=["*.log"])
        except Exception:
            logger.exception("Failed uploading logs (non-fatal)")

        # Optional but recommended: also upload whatever intermediate exists for debugging
        # (keep it light; you can restrict globs if needed)
        try:
            if paths.local_intermediate_root.exists():
                storage.upload_tree(paths.local_intermediate_root, paths.obj_intermediate_prefix)
        except Exception:
            logger.exception("Failed uploading intermediate debug artifacts (non-fatal)")

        manifest = {
            "job_id": job.job_id,
            "point_id": job.point_id,
            "status": "FAILED",
            "error": str(e),
            "traceback": tb[-8000:],
            "logs_prefix": paths.obj_logs_prefix,
            "intermediate_prefix": paths.obj_intermediate_prefix,
            "start_date": getattr(job, "start_date", None),
            "end_date": getattr(job, "end_date", None),
        }
        storage.put_text(
            paths.obj_manifest_key,
            json.dumps(manifest, indent=2),
            content_type="application/json",
        )
        return manifest
    finally:
        # Always attempt cleanup of bulky local artifacts
        keep = os.getenv("KEEP_LOCAL_ARTIFACTS", "0") == "1"
        keep_on_fail = os.getenv("KEEP_LOCAL_ON_FAIL", "1") == "1"

        # If job failed and you want to keep local artifacts, skip cleanup
        failed = False
        try:
            # you can detect failure based on manifest if you prefer;
            # simplest is to set failed=True in the except block
            pass
        except Exception:
            pass

        if keep:
            logger.info("KEEP_LOCAL_ARTIFACTS=1 -> skipping cleanup")
            return

        if failed and keep_on_fail:
            logger.info("Job failed and KEEP_LOCAL_ON_FAIL=1 -> skipping cleanup")
            return

        for sub in [paths.local_intermediate_root, paths.local_features_root]:
            try:
                if sub.exists():
                    shutil.rmtree(sub, ignore_errors=True)
                    logger.info(f"Cleaned local dir: {sub}")
            except Exception:
                logger.exception(f"Failed cleaning local dir: {sub}")


def main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Scaleway soil-sentinel job worker")
    parser.add_argument("--payload", required=True, help="Path to payload JSON file")
    parser.add_argument("--config", default="configs/dev.yaml", help="Config path")
    args = parser.parse_args()

    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)

    manifest = run_one_job(payload, config_path=args.config)
    print(json.dumps(manifest, indent=2))

    if manifest.get("status") != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    main()
