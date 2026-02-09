#!/usr/bin/env python3
"""
Worker Execution Framework for Statistics API

Handles parallel execution of statistics API jobs with proper error handling.
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..client import StatisticsApiClient, create_client_from_env
from ..models import JobSpec, JobResult, DailyStatsRecord, AggregatedStatsRecord
from ..processing.parsers import parse_daily_records, aggregate_records

@dataclass
class WorkerConfig:
    """Configuration for worker execution"""
    evalscript: str
    interval: str = "P1D"
    resolution: int = 10
    coverage_threshold: float = 0.8
    max_workers: int = 3

class StatisticsWorker:
    """
    Executes statistics API jobs with parallel processing
    """

    def __init__(self, config: WorkerConfig):
        self.config = config

    def _execute_single_job(self, job: JobSpec) -> JobResult:
        """
        Execute a single statistics API job

        Args:
            job: Job specification to execute

        Returns:
            JobResult with execution results
        """
        try:
            # Create API client
            client = create_client_from_env()

            # Execute API request
            response = client.request_statistics(
                bbox=job.bbox.to_list(),
                start_date=job.start_date,
                end_date=job.end_date,
                interval=self.config.interval,
                evalscript=self.config.evalscript,
                res=self.config.resolution,
                mosaicking_order="leastCC"
            )

            # Parse daily records
            daily_rows = parse_daily_records(
                response,
                lat=job.lat,
                lon=job.lon,
                bbox=job.bbox.to_list(),
                start_date=job.start_date,
                end_date=job.end_date,
                interval=self.config.interval
            )

            # Aggregate records
            kept_rows, aggregated = aggregate_records(
                daily_rows,
                coverage_threshold=self.config.coverage_threshold
            )

            return JobResult(
                status="SUCCESS",
                job_id=job.job_id,
                point_id=job.point_id,
                daily_rows=daily_rows,
                kept_rows=kept_rows,
                aggregated=aggregated
            )

        except Exception as e:
            return JobResult(
                status="FAILED",
                job_id=job.job_id,
                point_id=job.point_id,
                error=str(e)
            )

    def execute_jobs(
        self,
        jobs: List[JobSpec],
        *,
        out_dir: Path,
        progress_callback: Optional[callable] = None
    ) -> List[JobResult]:
        """
        Execute multiple jobs with parallel processing

        Args:
            jobs: List of job specifications to execute
            out_dir: Output directory for results
            progress_callback: Optional callback for progress updates

        Returns:
            List of JobResult objects
        """
        results: List[JobResult] = []

        # Prepare output directories
        out_raw_dir = out_dir / "raw_response"
        out_raw_dir.mkdir(parents=True, exist_ok=True)

        out_daily_parsed = out_dir / "daily_parsed.jsonl"
        out_daily_kept = out_dir / "daily_kept.jsonl"
        out_agg = out_dir / "aggregated_one_row.jsonl"
        out_err = out_dir / "errors.jsonl"

        # Clear old output files
        for p in [out_daily_parsed, out_daily_kept, out_agg, out_err]:
            if p.exists():
                p.unlink()

        # Execute jobs
        if self.config.max_workers <= 1:
            # Sequential execution
            for i, job in enumerate(jobs, start=1):
                if progress_callback:
                    progress_callback(i, len(jobs), job.point_id, "STARTED")

                result = self._execute_single_job(job)
                results.append(result)

                # Write results to files
                self._write_job_results(
                    result,
                    out_raw_dir,
                    out_daily_parsed,
                    out_daily_kept,
                    out_agg,
                    out_err
                )

                if progress_callback:
                    progress_callback(i, len(jobs), job.point_id, result.status)
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_job = {
                    executor.submit(self._execute_single_job, job): job
                    for job in jobs
                }

                completed = 0
                for future in as_completed(future_to_job):
                    completed += 1
                    job = future_to_job[future]

                    try:
                        result = future.result()
                    except Exception as e:
                        result = JobResult(
                            status="FAILED",
                            job_id=job.job_id,
                            point_id=job.point_id,
                            error=str(e)
                        )

                    results.append(result)

                    # Write results to files
                    self._write_job_results(
                        result,
                        out_raw_dir,
                        out_daily_parsed,
                        out_daily_kept,
                        out_agg,
                        out_err
                    )

                    if progress_callback:
                        progress_callback(completed, len(jobs), job.point_id, result.status)

        return results

    def _write_job_results(
        self,
        result: JobResult,
        out_raw_dir: Path,
        out_daily_parsed: Path,
        out_daily_kept: Path,
        out_agg: Path,
        out_err: Path
    ) -> None:
        """
        Write job results to output files

        Args:
            result: Job result to write
            out_raw_dir: Directory for raw API responses
            out_daily_parsed: File for daily parsed records
            out_daily_kept: File for kept daily records
            out_agg: File for aggregated records
            out_err: File for errors
        """
        if result.status != "SUCCESS":
            # Write error
            with open(out_err, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "status": result.status,
                    "job_id": result.job_id,
                    "point_id": result.point_id,
                    "error": result.error
                }, ensure_ascii=False) + "\n")
            return

        # Write daily parsed records
        for row in result.daily_rows:
            with open(out_daily_parsed, "a", encoding="utf-8") as f:
                f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

        # Write kept daily records
        for row in result.kept_rows:
            with open(out_daily_kept, "a", encoding="utf-8") as f:
                f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

        # Write aggregated record
        if result.aggregated:
            with open(out_agg, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.aggregated.__dict__, ensure_ascii=False) + "\n")

def create_worker(
    evalscript: str,
    *,
    interval: str = "P1D",
    resolution: int = 20,
    coverage_threshold: float = 0.8,
    max_workers: int = 3
) -> StatisticsWorker:
    """
    Factory function to create statistics worker

    Args:
        evalscript: Evalscript for feature computation
        interval: Aggregation interval
        resolution: Spatial resolution
        coverage_threshold: Minimum coverage threshold
        max_workers: Maximum number of parallel workers

    Returns:
        Configured StatisticsWorker instance
    """
    config = WorkerConfig(
        evalscript=evalscript,
        interval=interval,
        resolution=resolution,
        coverage_threshold=coverage_threshold,
        max_workers=max_workers
    )

    return StatisticsWorker(config)