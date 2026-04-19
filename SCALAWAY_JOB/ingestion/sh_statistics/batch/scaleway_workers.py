#!/usr/bin/env python3
"""
Scaleway Storage-Aware Worker Execution Framework for Statistics API

Handles parallel execution of statistics API jobs with proper error handling
and storage of results to Scalaway object storage.
"""

from __future__ import annotations

import json
import uuid
import tempfile
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging

from ..client import StatisticsApiClient, create_client_from_env
from ..models import JobSpec, JobResult, DailyStatsRecord, AggregatedStatsRecord
from ..processing.parsers import parse_daily_records, aggregate_records
from sh_pipeline.storage import storage_from_env

# Per-worker-process singletons — set by _worker_process_init
_proc_sh_client: Optional[StatisticsApiClient] = None
_proc_evalscript: str = ""
_proc_interval: str = "P1D"
_proc_resolution: int = 10


def _worker_process_init(evalscript: str, interval: str, resolution: int) -> None:
    """Runs once per worker process; creates shared clients so jobs don't repeat the OAuth handshake."""
    global _proc_sh_client, _proc_evalscript, _proc_interval, _proc_resolution
    _proc_sh_client = create_client_from_env()
    _proc_evalscript = evalscript
    _proc_interval = interval
    _proc_resolution = resolution


def _execute_job_in_worker(job: JobSpec) -> JobResult:
    """Module-level job runner — uses the per-process client set by _worker_process_init."""
    try:
        client = _proc_sh_client
        size_m = _proc_resolution * 3
        evalscript_type = "only_scl" if "only_scl" in _proc_evalscript.lower() else "features"

        chunks = _year_chunks(job.start_date, job.end_date)
        all_daily_rows = []

        for chunk_start, chunk_end in chunks:
            response = client.request_statistics_meter_based(
                lat=job.lat,
                lon=job.lon,
                size_m=size_m,
                resolution_m=_proc_resolution,
                start_date=chunk_start,
                end_date=chunk_end,
                interval=_proc_interval,
                evalscript=_proc_evalscript,
                mosaicking_order="leastCC"
            )

            logging.info("chunk %s–%s: %d intervals", chunk_start, chunk_end,
                         len(response.get("data", [])))

            chunk_rows = parse_daily_records(
                response,
                lat=job.lat,
                lon=job.lon,
                bbox=job.bbox.to_list(),
                start_date=chunk_start,
                end_date=chunk_end,
                interval=_proc_interval,
                evalscript_type=evalscript_type
            )
            all_daily_rows.extend(chunk_rows)

        for row in all_daily_rows:
            row.query_start_date = job.start_date
            row.query_end_date = job.end_date

        kept_rows, aggregated = aggregate_records(
            all_daily_rows,
            coverage_threshold=job.coverage_threshold
        )

        return JobResult(
            status="SUCCESS",
            job_id=job.job_id,
            point_id=job.point_id,
            daily_rows=all_daily_rows,
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

@dataclass
class ScalewayWorkerConfig:
    """Configuration for Scaleway worker execution"""
    evalscript: str
    interval: str = "P1D"
    resolution: int = 10
    coverage_threshold: float = 0.8
    max_workers: int = 3
    storage_prefix: str = "batch_results"

def _year_chunks(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Split a date range into calendar-year-sized chunks.

    A 10-year daily request (~3 650 intervals) can exceed the SH Statistics API
    server timeout. Splitting into annual sub-requests keeps each call under
    ~365 intervals and is safe to reassemble.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    chunks: List[Tuple[str, str]] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = date(chunk_start.year, 12, 31)
        chunk_end = min(chunk_end, end)
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        chunk_start = date(chunk_start.year + 1, 1, 1)
    return chunks


class ScalewayStatisticsWorker:
    """
    Executes statistics API jobs with parallel processing and stores results
    to Scalaway object storage
    """

    def __init__(self, config: ScalewayWorkerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Storage client is created lazily in the main process only
        self._storage_client = None

    def _get_storage_client(self):
        """Lazy initialization of storage client (main process only)"""
        if self._storage_client is None:
            self._storage_client = storage_from_env(logger=self.logger)
        return self._storage_client


    def _upload_job_results(self, result: JobResult) -> None:
        """
        Upload job results to Scalaway object storage

        Args:
            result: Job result to upload
        """
        storage_client = self._get_storage_client()

        if result.status != "SUCCESS":
            # Upload error information
            error_data = {
                "status": result.status,
                "job_id": result.job_id,
                "point_id": result.point_id,
                "error": result.error,
                "timestamp": json.dumps({"timestamp": "TODO"})
            }
            error_key = f"{self.config.storage_prefix}/errors/{result.point_id}.json"
            storage_client.put_text(error_key, json.dumps(error_data, ensure_ascii=False))
            return

        # Upload aggregated results
        if result.aggregated:
            agg_key = f"{self.config.storage_prefix}/aggregated/{result.point_id}.json"
            storage_client.put_text(agg_key, json.dumps(result.aggregated.__dict__, ensure_ascii=False))

        # Upload daily parsed records
        daily_parsed_key = f"{self.config.storage_prefix}/daily_parsed/{result.point_id}.jsonl"
        daily_parsed_lines = []
        for row in result.daily_rows:
            # Convert to dict and ensure proper JSON serialization
            row_dict = row.__dict__.copy()
            # Convert any numpy types or special objects to native Python types
            if 'p50' in row_dict and isinstance(row_dict['p50'], dict):
                row_dict['p50'] = {k: float(v) if v is not None and hasattr(v, '__float__') else v for k, v in row_dict['p50'].items()}
            daily_parsed_lines.append(json.dumps(row_dict, ensure_ascii=False))
        storage_client.put_text(daily_parsed_key, "\n".join(daily_parsed_lines))

        # Upload kept daily records
        daily_kept_key = f"{self.config.storage_prefix}/daily_kept/{result.point_id}.jsonl"
        daily_kept_lines = []
        for row in result.kept_rows:
            # Convert to dict and ensure proper JSON serialization
            row_dict = row.__dict__.copy()
            # Convert any numpy types or special objects to native Python types
            if 'p50' in row_dict and isinstance(row_dict['p50'], dict):
                row_dict['p50'] = {k: float(v) if v is not None and hasattr(v, '__float__') else v for k, v in row_dict['p50'].items()}
            daily_kept_lines.append(json.dumps(row_dict, ensure_ascii=False))
        storage_client.put_text(daily_kept_key, "\n".join(daily_kept_lines))

    def execute_jobs(
        self,
        jobs: List[JobSpec],
        *,
        progress_callback: Optional[Callable] = None
    ) -> List[JobResult]:
        """
        Execute multiple jobs with parallel processing and upload to storage

        Args:
            jobs: List of job specifications to execute
            progress_callback: Optional callback for progress updates

        Returns:
            List of JobResult objects
        """
        results: List[JobResult] = []

        # Execute jobs
        if self.config.max_workers <= 1:
            _worker_process_init(self.config.evalscript, self.config.interval, self.config.resolution)
            for i, job in enumerate(jobs, start=1):
                if progress_callback:
                    progress_callback(i, len(jobs), job.point_id, "STARTED")

                result = _execute_job_in_worker(job)
                results.append(result)

                # Upload results to storage
                self._upload_job_results(result)

                if progress_callback:
                    progress_callback(i, len(jobs), job.point_id, result.status)
        else:
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                initializer=_worker_process_init,
                initargs=(self.config.evalscript, self.config.interval, self.config.resolution),
            ) as executor:
                future_to_job = {
                    executor.submit(_execute_job_in_worker, job): job
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

                    # Upload results to storage (main process only)
                    self._upload_job_results(result)

                    if progress_callback:
                        progress_callback(completed, len(jobs), job.point_id, result.status)

        return results

def create_scaleway_worker(
    evalscript: str,
    *,
    interval: str = "P1D",
    resolution: int = 20,
    coverage_threshold: float = 0.8,
    max_workers: int = 3,
    storage_prefix: str = "batch_results"
) -> ScalewayStatisticsWorker:
    """
    Factory function to create Scaleway statistics worker

    Args:
        evalscript: Evalscript for feature computation
        interval: Aggregation interval
        resolution: Spatial resolution
        coverage_threshold: Minimum coverage threshold
        max_workers: Maximum number of parallel workers
        storage_prefix: Prefix for storage keys

    Returns:
        Configured ScalewayStatisticsWorker instance
    """
    config = ScalewayWorkerConfig(
        evalscript=evalscript,
        interval=interval,
        resolution=resolution,
        coverage_threshold=coverage_threshold,
        max_workers=max_workers,
        storage_prefix=storage_prefix
    )

    return ScalewayStatisticsWorker(config)