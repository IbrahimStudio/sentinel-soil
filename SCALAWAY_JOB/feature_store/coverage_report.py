"""
coverage_report.py — Post-ingestion data quality assessment.

Reads raw .npz rasters from S3 (written by ingestion/process_api), applies the
pixel-level filters from filter_config.json, and produces:

  {output_dir}/
    summary.csv          — per-point metrics (n_fetched, n_valid, status, ...)
    summary.json         — aggregate stats + quality alerts
    hist_dates.png       — fetched vs valid dates per point
    ecdf_valid_dates.png — ECDF of valid dates with min_obs threshold line
    scl_breakdown.png    — SCL class distribution across all pixels
    ndvi_distribution.png— NDVI histogram before/after filtering
    band_ranges.png      — reflectance box plots per band (sanity: expect [0,1])
    monthly_heatmap.png  — acquisition frequency by month
    map_coverage.png     — scatter map coloured by valid dates

Usage (from SCALAWAY_JOB/):
    python feature_store/coverage_report.py \\
        --xlsx gabri_filters.xlsx \\
        --filter-config DSM_WEBAPP/filter_config.json \\
        --raster-prefix raw_rasters/ \\
        --output-dir training/reports/coverage
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from tqdm import tqdm

# Reuse filter logic from the feature-store extractor
from process_api_extractor import _apply_pixel_filter, _B04, _B08, _SCL

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCL_LABELS = {
    0: "No Data", 1: "Saturated", 2: "Dark Features", 3: "Cloud Shadow",
    4: "Vegetation", 5: "Not Vegetated", 6: "Water", 7: "Unclassified",
    8: "Cloud Med", 9: "Cloud High", 10: "Thin Cirrus", 11: "Snow/Ice",
}
SCL_COLORS = {
    0: "#000000", 1: "#ff0000", 2: "#2f2f2f", 3: "#643200",
    4: "#00a000", 5: "#ffe65a", 6: "#0000ff", 7: "#808080",
    8: "#c0c0c0", 9: "#ffffff", 10: "#64c8ff", 11: "#ff96ff",
}

BAND_NAMES_10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _make_s3():
    cfg = BotoConfig(retries={"max_attempts": 3, "mode": "standard"},
                     connect_timeout=15, read_timeout=120)
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SCALEWAY_S3_ENDPOINT"],
        region_name=os.environ.get("SCALEWAY_S3_REGION", "fr-par"),
        aws_access_key_id=os.environ["SCALEWAY_ACCESS_KEY"],
        aws_secret_access_key=os.environ["SCALEWAY_SECRET_KEY"],
        config=cfg,
    )


def _try_load_npz(s3, bucket: str, key: str):
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return dict(np.load(io.BytesIO(body), allow_pickle=False))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def _try_load_meta(s3, bucket: str, key: str):
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body.decode())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


# ---------------------------------------------------------------------------
# Per-point processing
# ---------------------------------------------------------------------------

def _process_point(
    point_id: str,
    npz: dict,
    meta: dict,
    filter_config: dict,
) -> dict:
    """Compute coverage metrics for one point from its raw rasters."""
    pf = filter_config["pixel_filter"]
    min_obs = filter_config["temporal_aggregation"].get("min_valid_observations_per_pixel", 3)

    rasters = npz["rasters"]           # (N, 9, 9, 11)
    dates   = list(npz["dates"])       # ["YYYY-MM-DD", ...]
    n_dates = rasters.shape[0]

    mask = _apply_pixel_filter(
        rasters,
        pf["scl_keep_classes"],
        pf.get("ndvi_min", -0.1),
        pf["ndvi_max"],
        pf["nbr2_max"],
    )   # (N, 9, 9) bool

    cp_row, cp_col = meta.get("center_pixel", [4, 4])
    center_valid = mask[:, cp_row, cp_col]   # (N,) bool
    n_valid = int(center_valid.sum())

    # Bare-soil pixel fraction per date (fraction of 9×9 patch passing filter)
    bare_pct_per_date = mask.reshape(n_dates, -1).mean(axis=1)  # (N,)

    # SCL class counts across ALL pixels and dates
    scl_all = rasters[..., _SCL].astype(np.int16).ravel()
    scl_counts = {cls: int((scl_all == cls).sum()) for cls in range(12)}

    # NDVI at all [4,5] SCL pixels (pre-NDVI filter)
    scl_mask_all = (rasters[..., _SCL].astype(np.int16) == 4) | \
                   (rasters[..., _SCL].astype(np.int16) == 5)
    b08 = rasters[..., 6].astype(np.float32)
    b04 = rasters[..., 2].astype(np.float32)
    denom = b08 + b04
    ndvi_all = np.where(np.abs(denom) < 1e-9, 0.0, (b08 - b04) / denom)
    ndvi_scl_pixels  = ndvi_all[scl_mask_all].ravel()
    ndvi_valid_pixels = ndvi_all[mask].ravel()

    # Band reflectance values at valid center-pixel dates
    if n_valid > 0:
        band_vals = rasters[center_valid, cp_row, cp_col, :10].astype(np.float64)
    else:
        band_vals = np.empty((0, 10))

    # Acquisition months
    months = [int(d[5:7]) for d in dates if len(d) >= 7]

    return {
        "point_id":           point_id,
        "lat":                meta.get("lucas_lat", float("nan")),
        "lon":                meta.get("lucas_lon", float("nan")),
        "n_dates_fetched":    n_dates,
        "n_valid_dates":      n_valid,
        "bare_pct_mean":      float(bare_pct_per_date.mean()) if n_dates > 0 else 0.0,
        "status":             "ok" if n_valid >= min_obs else "insufficient",
        # Accumulated arrays (collected globally, not stored per-point)
        "_scl_counts":        scl_counts,
        "_ndvi_scl":          ndvi_scl_pixels,
        "_ndvi_valid":        ndvi_valid_pixels,
        "_band_vals":         band_vals,
        "_months":            months,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_hist_dates(rows: list[dict], min_obs: int, out: Path) -> None:
    fetched = [r["n_dates_fetched"] for r in rows]
    valid   = [r["n_valid_dates"]   for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = range(0, max(fetched + [1]) + 2)
    ax.hist(fetched, bins=bins, alpha=0.6, label="Fetched dates", color="#4c8cbf")
    ax.hist(valid,   bins=bins, alpha=0.7, label="Valid bare-soil dates", color="#e07b39")
    ax.axvline(min_obs, color="red", linestyle="--", linewidth=1.2,
               label=f"min_obs = {min_obs}")
    ax.set_xlabel("Dates per point")
    ax.set_ylabel("Number of LUCAS points")
    ax.set_title("Acquisition coverage: fetched vs valid bare-soil dates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_ecdf_valid(rows: list[dict], min_obs: int, out: Path) -> None:
    valid = sorted(r["n_valid_dates"] for r in rows)
    n = len(valid)
    y = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(valid, y, where="post", color="#e07b39", linewidth=1.8)
    ax.axvline(min_obs, color="red", linestyle="--", linewidth=1.2,
               label=f"min_obs = {min_obs}")
    # Annotate dropout fraction
    pct_below = 100 * sum(1 for v in valid if v < min_obs) / n
    ax.axhline(pct_below / 100, color="grey", linestyle=":", linewidth=1)
    ax.text(min_obs + 0.3, pct_below / 100 + 0.01,
            f"{pct_below:.1f}% will be dropped", fontsize=8, color="grey")
    ax.set_xlabel("Valid bare-soil dates (center pixel)")
    ax.set_ylabel("Cumulative fraction of points")
    ax.set_title("ECDF — valid dates per LUCAS point")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_scl_breakdown(rows: list[dict], out: Path) -> None:
    total_counts: dict[int, int] = defaultdict(int)
    for r in rows:
        for cls, cnt in r["_scl_counts"].items():
            total_counts[cls] += cnt

    total = sum(total_counts.values()) or 1
    classes = sorted(total_counts)
    fractions = [total_counts[c] / total for c in classes]
    labels    = [SCL_LABELS.get(c, str(c)) for c in classes]
    colors    = [SCL_COLORS.get(c, "#aaaaaa") for c in classes]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, fractions, color=colors, edgecolor="black", linewidth=0.5)
    for bar, frac in zip(bars, fractions):
        if frac > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{frac:.1%}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Fraction of all pixels")
    ax.set_title("SCL class distribution across all fetched rasters")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_ndvi_distribution(rows: list[dict], ndvi_min: float, ndvi_max: float, out: Path) -> None:
    ndvi_scl   = np.concatenate([r["_ndvi_scl"]   for r in rows if len(r["_ndvi_scl"])   > 0], axis=0) \
                 if any(len(r["_ndvi_scl"]) > 0 for r in rows) else np.array([])
    ndvi_valid = np.concatenate([r["_ndvi_valid"] for r in rows if len(r["_ndvi_valid"]) > 0], axis=0) \
                 if any(len(r["_ndvi_valid"]) > 0 for r in rows) else np.array([])

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-0.5, 0.8, 80)
    if len(ndvi_scl) > 0:
        ax.hist(ndvi_scl,   bins=bins, alpha=0.5, color="#4c8cbf", label="SCL [4,5] pixels (pre-filter)")
    if len(ndvi_valid) > 0:
        ax.hist(ndvi_valid, bins=bins, alpha=0.7, color="#e07b39", label="After all filters")
    ax.axvline(ndvi_min, color="red",    linestyle="--", linewidth=1.2, label=f"ndvi_min={ndvi_min}")
    ax.axvline(ndvi_max, color="purple", linestyle="--", linewidth=1.2, label=f"ndvi_max={ndvi_max}")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Pixel count")
    ax.set_title("NDVI distribution: SCL-filtered vs fully-filtered pixels")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_band_ranges(rows: list[dict], out: Path) -> None:
    all_vals = np.concatenate([r["_band_vals"] for r in rows if len(r["_band_vals"]) > 0], axis=0) \
               if any(len(r["_band_vals"]) > 0 for r in rows) else np.empty((0, 10))

    fig, ax = plt.subplots(figsize=(10, 4))
    if len(all_vals) > 0:
        ax.boxplot(
            [all_vals[:, i] for i in range(10)],
            labels=BAND_NAMES_10,
            patch_artist=True,
            boxprops=dict(facecolor="#a8d0e6"),
            medianprops=dict(color="navy", linewidth=1.5),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            whis=(1, 99),
        )
    ax.axhline(0.0, color="red",    linestyle="--", linewidth=1, label="Expected min (0)")
    ax.axhline(1.0, color="orange", linestyle="--", linewidth=1, label="Expected max (1)")
    ax.set_ylabel("Reflectance (BOA)")
    ax.set_title("Band reflectance ranges at valid bare-soil center pixels")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_monthly_heatmap(rows: list[dict], out: Path) -> None:
    month_counts = defaultdict(int)
    for r in rows:
        for m in r["_months"]:
            month_counts[m] += 1

    months = list(range(1, 13))
    counts = [month_counts[m] for m in months]
    labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(labels, counts, color="#5ba85a", edgecolor="black", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Number of acquisitions")
    ax.set_title("Acquisition frequency by calendar month")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_map_coverage(rows: list[dict], min_obs: int, out: Path) -> None:
    lats  = [r["lat"]           for r in rows]
    lons  = [r["lon"]           for r in rows]
    valid = [r["n_valid_dates"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(lons, lats, c=valid, cmap="RdYlGn", s=8,
                    vmin=0, vmax=max(valid + [min_obs * 2]),
                    linewidths=0, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Valid bare-soil dates")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic coverage: valid bare-soil dates per LUCAS point")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

_REPORT_CONTENT_TYPES = {
    ".csv":  "text/csv",
    ".json": "application/json",
    ".png":  "image/png",
}


def _upload_report_to_s3(out_dir: Path, timestamp: str, s3_prefix: str = "coverage_reports/") -> None:
    """Upload all report files to S3 under a versioned and a latest prefix."""
    bucket   = os.environ.get("SCALEWAY_S3_BUCKET", "")
    endpoint = os.environ.get("SCALEWAY_S3_ENDPOINT", "")
    access   = os.environ.get("SCALEWAY_ACCESS_KEY", "")
    secret   = os.environ.get("SCALEWAY_SECRET_KEY", "")

    if not all([bucket, endpoint, access, secret]):
        log.warning("S3 credentials incomplete — skipping coverage report upload.")
        return

    if not s3_prefix.endswith("/"):
        s3_prefix += "/"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("SCALEWAY_S3_REGION", "fr-par"),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=BotoConfig(retries={"max_attempts": 6, "mode": "standard"}),
    )

    files = list(out_dir.iterdir())
    destinations = [f"{s3_prefix}{timestamp}/", f"{s3_prefix}latest/"]

    for dest in destinations:
        for f in files:
            if not f.is_file():
                continue
            content_type = _REPORT_CONTENT_TYPES.get(f.suffix, "application/octet-stream")
            key = f"{dest}{f.name}"
            s3.upload_file(str(f), bucket, key, ExtraArgs={"ContentType": content_type})
        log.info("Coverage report uploaded → s3://%s/%s", bucket, dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-ingestion coverage quality report")
    p.add_argument("--xlsx",           required=True)
    p.add_argument("--filter-config",  required=True)
    p.add_argument("--raster-prefix",  default="raw_rasters/")
    p.add_argument("--output-dir",     default="/reports/coverage")
    p.add_argument("--limit",          type=int, default=-1,
                   help="Process only first N points (-1 = all)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.filter_config) as f:
        filter_config = json.load(f)

    pf      = filter_config["pixel_filter"]
    min_obs = filter_config["temporal_aggregation"].get("min_valid_observations_per_pixel", 3)
    ndvi_min = pf.get("ndvi_min", -0.1)
    ndvi_max = pf["ndvi_max"]

    lucas_df = pd.read_excel(args.xlsx)
    if args.limit > 0:
        lucas_df = lucas_df.head(args.limit)
    log.info("Loaded %d LUCAS points.", len(lucas_df))

    bucket = os.environ["SCALEWAY_S3_BUCKET"]
    prefix = args.raster_prefix if args.raster_prefix.endswith("/") else args.raster_prefix + "/"
    s3 = _make_s3()

    rows_ok:       list[dict] = []
    rows_missing:  list[dict] = []

    for _, src in tqdm(lucas_df.iterrows(), total=len(lucas_df), desc="Analysing points"):
        pid = str(src["POINT_ID"])
        npz  = _try_load_npz( s3, bucket, f"{prefix}{pid}.npz")
        meta = _try_load_meta(s3, bucket, f"{prefix}{pid}_meta.json")

        if npz is None or meta is None:
            rows_missing.append({
                "point_id": pid,
                "lat": float(src.get("TH_LAT", float("nan"))),
                "lon": float(src.get("TH_LONG", float("nan"))),
                "n_dates_fetched": 0,
                "n_valid_dates": 0,
                "bare_pct_mean": 0.0,
                "status": "not_fetched",
                "_scl_counts": {}, "_ndvi_scl": np.array([]),
                "_ndvi_valid": np.array([]), "_band_vals": np.empty((0, 10)), "_months": [],
            })
            continue

        rows_ok.append(_process_point(pid, npz, meta, filter_config))

    all_rows = rows_ok + rows_missing
    log.info("Processed %d points (%d fetched, %d missing).",
             len(all_rows), len(rows_ok), len(rows_missing))

    # --- Summary CSV ---
    summary_cols = ["point_id", "lat", "lon", "n_dates_fetched", "n_valid_dates",
                    "bare_pct_mean", "status"]
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_rows])
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    log.info("Written summary.csv (%d rows)", len(summary_df))

    # --- Aggregate stats + alerts ---
    n_total       = len(all_rows)
    n_not_fetched = sum(1 for r in all_rows if r["status"] == "not_fetched")
    n_insufficient = sum(1 for r in all_rows if r["status"] == "insufficient")
    n_ok          = sum(1 for r in all_rows if r["status"] == "ok")
    valid_counts  = [r["n_valid_dates"] for r in rows_ok]

    alerts: list[str] = []
    fetch_rate    = (n_total - n_not_fetched) / n_total if n_total else 0
    survival_rate = n_ok / n_total if n_total else 0

    if fetch_rate < 0.9:
        alerts.append(f"LOW FETCH RATE: only {fetch_rate:.1%} of points have rasters on S3 — check ingestion logs")
    if survival_rate < 0.5:
        alerts.append(f"HIGH DROPOUT: only {survival_rate:.1%} of points have >= {min_obs} valid dates — consider relaxing filters or extending time_window_days")
    if valid_counts:
        p25 = int(np.percentile(valid_counts, 25))
        if p25 < min_obs:
            alerts.append(f"P25 valid dates = {p25} < min_obs ({min_obs}) — most points are near the dropout threshold")

    # Band range check from valid pixels
    all_band_vals = np.concatenate([r["_band_vals"] for r in rows_ok if len(r["_band_vals"]) > 0], axis=0) \
                    if any(len(r["_band_vals"]) > 0 for r in rows_ok) else np.empty((0, 10))
    if len(all_band_vals) > 0:
        band_max = float(all_band_vals.max())
        band_min = float(all_band_vals.min())
        if band_max > 2.0:
            alerts.append(f"REFLECTANCE OUT OF RANGE: max band value = {band_max:.1f} (expected <= 1.0) — check evalscript scaling")
        if band_min < -0.5:
            alerts.append(f"REFLECTANCE OUT OF RANGE: min band value = {band_min:.3f} — possible nodata or scaling issue")

    summary_json = {
        "n_total":          n_total,
        "n_fetched":        n_total - n_not_fetched,
        "n_not_fetched":    n_not_fetched,
        "n_ok":             n_ok,
        "n_insufficient":   n_insufficient,
        "fetch_rate":       round(fetch_rate, 4),
        "survival_rate":    round(survival_rate, 4),
        "valid_dates": {
            "median": float(np.median(valid_counts)) if valid_counts else 0,
            "p25":    float(np.percentile(valid_counts, 25)) if valid_counts else 0,
            "p75":    float(np.percentile(valid_counts, 75)) if valid_counts else 0,
            "max":    float(max(valid_counts)) if valid_counts else 0,
        },
        "filter_config": {
            "min_obs": min_obs,
            "ndvi_min": ndvi_min,
            "ndvi_max": ndvi_max,
            "nbr2_max": pf["nbr2_max"],
            "scl_keep": pf["scl_keep_classes"],
        },
        "alerts": alerts,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    if alerts:
        for a in alerts:
            log.warning("ALERT: %s", a)
    else:
        log.info("No quality alerts raised.")

    # --- Plots ---
    plot_rows = [r for r in all_rows if r["status"] != "not_fetched"]

    log.info("Generating plots...")
    _plot_hist_dates(plot_rows, min_obs, out_dir / "hist_dates.png")
    _plot_ecdf_valid(plot_rows, min_obs, out_dir / "ecdf_valid_dates.png")
    _plot_scl_breakdown(plot_rows, out_dir / "scl_breakdown.png")
    _plot_ndvi_distribution(plot_rows, ndvi_min, ndvi_max, out_dir / "ndvi_distribution.png")
    _plot_band_ranges(plot_rows, out_dir / "band_ranges.png")
    _plot_monthly_heatmap(plot_rows, out_dir / "monthly_heatmap.png")
    _plot_map_coverage(all_rows, min_obs, out_dir / "map_coverage.png")
    log.info("Plots written to %s/", out_dir)

    # --- Final summary log ---
    log.info("Coverage report complete:")
    log.info("  Total points:      %d", n_total)
    log.info("  Fetched:           %d (%.1f%%)", n_total - n_not_fetched, 100 * fetch_rate)
    log.info("  Surviving filter:  %d (%.1f%%)", n_ok, 100 * survival_rate)
    log.info("  Median valid dates: %.0f  (P25=%.0f, P75=%.0f)",
             summary_json["valid_dates"]["median"],
             summary_json["valid_dates"]["p25"],
             summary_json["valid_dates"]["p75"])
    if alerts:
        log.warning("%d quality alert(s) — see summary.json", len(alerts))

    # --- Upload to S3 ---
    _upload_report_to_s3(out_dir, run_ts)


if __name__ == "__main__":
    main()
