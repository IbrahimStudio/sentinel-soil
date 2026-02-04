#!/usr/bin/env python3
"""
Refactored Batch Statistics from XLSX

Batch processing of statistics API requests from Excel files using the new statistics module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from statistics.batch.xlsx_processor import create_xlsx_processor
from statistics.batch.workers import create_worker

def _parse_sheet_arg(sheet_arg: str) -> Any:
    """Parse sheet argument (can be name or index)"""
    if sheet_arg.isdigit():
        return int(sheet_arg)
    return sheet_arg

def _progress_callback(current: int, total: int, point_id: str, status: str) -> None:
    """Progress callback for worker execution"""
    print(f"[{current}/{total}] {status} point_id={point_id}")

def main() -> None:
    """Main entry point for batch processing from XLSX"""
    ap = argparse.ArgumentParser(description="Batch Statistics API from XLSX")
    ap.add_argument("--xlsx", type=str, required=True, help="Path to input Excel file")
    ap.add_argument("--sheet", type=str, default="0", help='Sheet name or index (e.g., "0" or "Sheet1")')
    ap.add_argument("--limit", type=int, default=-1, help="Process only first N rows (-1 for all)")
    ap.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    ap.add_argument("--evalscript_path", type=str, required=True, help="Path to evalscript file")
    ap.add_argument("--window_days", type=int, default=15, help="+/- days around survey date")
    ap.add_argument("--bbox_size_m", type=float, default=30.0, help="Bounding box size in meters")
    ap.add_argument("--interval", type=str, default="P1D", help="Aggregation interval")
    ap.add_argument("--res", type=int, default=20, help="Spatial resolution")
    ap.add_argument("--coverage_threshold", type=float, default=0.8, help="Minimum coverage threshold")
    ap.add_argument("--out_dir", type=str, default="out_batch_json", help="Output directory")
    args = ap.parse_args()

    try:
        # Read evalscript
        evalscript = Path(args.evalscript_path).read_text(encoding="utf-8")

        # Parse sheet argument
        sheet = _parse_sheet_arg(args.sheet)

        # Create XLSX processor
        xlsx_processor = create_xlsx_processor(
            xlsx_path=Path(args.xlsx),
            sheet=sheet,
            window_days=args.window_days,
            bbox_size_m=args.bbox_size_m,
            limit=args.limit
        )

        # Process Excel file to create jobs
        jobs = xlsx_processor.process()
        print(f"Prepared {len(jobs)} jobs from {args.xlsx}. workers={args.workers}")

        # Create worker
        worker = create_worker(
            evalscript=evalscript,
            interval=args.interval,
            resolution=args.res,
            coverage_threshold=args.coverage_threshold,
            max_workers=args.workers
        )

        # Execute jobs
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = worker.execute_jobs(
            jobs=jobs,
            out_dir=out_dir,
            progress_callback=_progress_callback
        )

        # Print summary
        success_count = sum(1 for r in results if r.status == "SUCCESS")
        failure_count = len(results) - success_count

        print(f"\nProcessing complete:")
        print(f"Success: {success_count}/{len(results)}")
        print(f"Failed: {failure_count}/{len(results)}")
        print(f"Raw per-point API JSON: {out_dir / 'raw_response'}")
        print(f"Daily parsed JSONL:     {out_dir / 'daily_parsed.jsonl'}")
        print(f"Daily kept JSONL:       {out_dir / 'daily_kept.jsonl'}")
        print(f"Aggregated JSONL:       {out_dir / 'aggregated_one_row.jsonl'}")
        print(f"Errors JSONL:           {out_dir / 'errors.jsonl'}")

    except Exception as e:
        print(f"Error in batch processing: {e}")
        raise

if __name__ == "__main__":
    main()