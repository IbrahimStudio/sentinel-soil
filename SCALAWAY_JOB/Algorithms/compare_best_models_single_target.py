#!/usr/bin/env python3
"""
Compare best model for ONE target across 3 scenarios (A/B/C),
passing directly the path to summary_metrics.json files.

Example:
python compare_best_models_single_target.py \
  --json-a /path/A/summary_metrics.json \
  --json-b /path/B/summary_metrics.json \
  --json-c /path/C/summary_metrics.json \
  --target Clay \
  --metric cv_r2_mean \
  --out ./Clay_best_r2.png
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


# --------------------------
# IO
# --------------------------
def load_metrics_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# Selection logic
# --------------------------
def best_for_target(
    rows: List[Dict[str, Any]],
    *,
    target: str,
    metric: str,
    higher_is_better: bool,
    cv_mode_filter: Optional[str] = None,
) -> Dict[str, Any]:

    candidates = [
        r for r in rows
        if r.get("target") == target
        and metric in r
        and r[metric] is not None
        and (cv_mode_filter is None or r.get("cv_mode") == cv_mode_filter)
    ]

    if not candidates:
        raise ValueError(
            f"No candidates found for target='{target}', metric='{metric}', cv_mode={cv_mode_filter}"
        )

    key_fn = lambda r: float(r[metric])

    return max(candidates, key=key_fn) if higher_is_better else min(candidates, key=key_fn)


# --------------------------
# Plot
# --------------------------
def plot_single_target(
    *,
    target: str,
    metric: str,
    best_a: Dict[str, Any],
    best_b: Dict[str, Any],
    best_c: Dict[str, Any],
    label_a: str,
    label_b: str,
    label_c: str,
    title: str,
    out_png: Optional[Path],
) -> None:

    # ---- Styling ----
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green

    labels = [label_a, label_b, label_c]
    values = [
        float(best_a[metric]),
        float(best_b[metric]),
        float(best_c[metric]),
    ]

    bests = [best_a, best_b, best_c]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(3)

    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9,
        width=0.6,
    )

    # ---- Title & Labels ----
    ax.set_title(title, pad=15)
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)

    # ---- Cleaner grid ----
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # ---- Annotate numeric value on top ----
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # ---- Annotate model/config inside bar ----
    for i, bar in enumerate(bars):
        r = bests[i]

        annotation = "\n".join(
            filter(None, [
                r.get("model", ""),
                r.get("config", ""),
                r.get("cv_type", ""),
            ])
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.55,
            annotation,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                alpha=0.4,
            ),
        )

    plt.tight_layout()

    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=300)
        print(f"[OK] Saved -> {out_png}")
    else:
        plt.show()


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json-a", required=True, type=str)
    parser.add_argument("--json-b", required=True, type=str)
    parser.add_argument("--json-c", required=True, type=str)

    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--metric", default="cv_r2_mean", type=str)

    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--cv-mode", default=None, type=str)

    parser.add_argument("--label-a", default="Scenario A (SCL+NDVI)")
    parser.add_argument("--label-b", default="Scenario B (SCL+CHIM)")
    parser.add_argument("--label-c", default="Scenario C (SCL CLEAN)")

    parser.add_argument("--title", default=None)
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    rows_a = load_metrics_json(Path(args.json_a))
    rows_b = load_metrics_json(Path(args.json_b))
    rows_c = load_metrics_json(Path(args.json_c))

    higher_is_better = not args.lower_is_better

    best_a = best_for_target(rows_a, target=args.target, metric=args.metric,
                             higher_is_better=higher_is_better, cv_mode_filter=args.cv_mode)
    best_b = best_for_target(rows_b, target=args.target, metric=args.metric,
                             higher_is_better=higher_is_better, cv_mode_filter=args.cv_mode)
    best_c = best_for_target(rows_c, target=args.target, metric=args.metric,
                             higher_is_better=higher_is_better, cv_mode_filter=args.cv_mode)

    title = args.title or f"{args.target} — best by {args.metric}"

    out_path = Path(args.out) if args.out else None

    plot_single_target(
        target=args.target,
        metric=args.metric,
        best_a=best_a,
        best_b=best_b,
        best_c=best_c,
        label_a=args.label_a,
        label_b=args.label_b,
        label_c=args.label_c,
        title=title,
        out_png=out_path,
    )


if __name__ == "__main__":
    main()