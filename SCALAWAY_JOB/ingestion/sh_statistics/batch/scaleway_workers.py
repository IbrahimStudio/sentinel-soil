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
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging

from ..client import StatisticsApiClient, create_client_from_env
from ..models import JobSpec, JobResult, DailyStatsRecord, AggregatedStatsRecord
from ..processing.parsers import parse_daily_records, aggregate_records
from sh_pipeline.storage import storage_from_env

@dataclass
class ScalewayWorkerConfig:
    """Configuration for Scaleway worker execution"""
    evalscript: str
    interval: str = "P1D"
    resolution: int = 10
    coverage_threshold: float = 0.8
    max_workers: int = 3
    storage_prefix: str = "batch_results"

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

    @staticmethod
    def _execute_single_job_static(job: JobSpec, evalscript: str, interval: str, resolution: int) -> JobResult:
        """
        Static version of job execution that can be pickled for ProcessPoolExecutor

        Args:
            job: Job specification to execute
            evalscript: Evalscript for feature computation
            interval: Aggregation interval
            resolution: Spatial resolution in meters

        Returns:
            JobResult with execution results
        """
        try:
            # Create API client
            client = create_client_from_env()

            # Execute API request using meter-based implementation
            size_m = resolution * 3  # 3x3 pixels minimum

            response = client.request_statistics_meter_based(
                lat=job.lat,
                lon=job.lon,
                size_m=size_m,
                resolution_m=resolution,
                start_date=job.start_date,
                end_date=job.end_date,
                interval=interval,
                evalscript=evalscript,
                mosaicking_order="leastCC"
            )

            logging.info(response)

            # Parse daily records
            # Determine evalscript type based on the evalscript content
            evalscript_type = "only_scl" if "only_scl" in evalscript.lower() else "features"

            daily_rows = parse_daily_records(
                response,
                lat=job.lat,
                lon=job.lon,
                bbox=job.bbox.to_list(),
                start_date=job.start_date,
                end_date=job.end_date,
                interval=interval,
                evalscript_type=evalscript_type
            )

            # Aggregate records using job-specific coverage threshold
            kept_rows, aggregated = aggregate_records(
                daily_rows,
                coverage_threshold=job.coverage_threshold
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

    def _execute_single_job(self, job: JobSpec) -> JobResult:
        """
        Execute a single statistics API job using meter-based requests

        Args:
            job: Job specification to execute

        Returns:
            JobResult with execution results
        """
        try:
            # Create API client
            client = create_client_from_env()

            # Execute API request using meter-based implementation
            # Note: We need to extract bbox size from the job's bbox
            # For now, we'll use a fixed size_m that matches the resolution
            # In a production system, we would calculate the actual size from the bbox
            size_m = self.config.resolution * 3  # 3x3 pixels minimum

            response = client.request_statistics_meter_based(
                lat=job.lat,
                lon=job.lon,
                size_m=size_m,
                resolution_m=self.config.resolution,
                start_date=job.start_date,
                end_date=job.end_date,
                interval=self.config.interval,
                evalscript=self.config.evalscript,
                mosaicking_order="leastCC"
            )

            print(response)

            # Parse daily records
            # Determine evalscript type based on the evalscript content
            evalscript_type = "only_scl" if "only_scl" in self.config.evalscript.lower() else "features"

            daily_rows = parse_daily_records(
                response,
                lat=job.lat,
                lon=job.lon,
                bbox=job.bbox.to_list(),
                start_date=job.start_date,
                end_date=job.end_date,
                interval=self.config.interval,
                evalscript_type=evalscript_type
            )

            # Aggregate records using job-specific coverage threshold
            kept_rows, aggregated = aggregate_records(
                daily_rows,
                coverage_threshold=job.coverage_threshold
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
            # Sequential execution
            for i, job in enumerate(jobs, start=1):
                if progress_callback:
                    progress_callback(i, len(jobs), job.point_id, "STARTED")

                result = self._execute_single_job(job)
                results.append(result)

                # Upload results to storage
                self._upload_job_results(result)

                if progress_callback:
                    progress_callback(i, len(jobs), job.point_id, result.status)
        else:
            # Parallel execution using static method to avoid pickling issues
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_job = {
                    executor.submit(
                        self._execute_single_job_static,
                        job,
                        self.config.evalscript,
                        self.config.interval,
                        self.config.resolution
                    ): job
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