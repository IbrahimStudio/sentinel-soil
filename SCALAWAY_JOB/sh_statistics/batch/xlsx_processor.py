#!/usr/bin/env python3
"""
XLSX Batch Processor for Statistics API

Handles reading input data from Excel files and managing batch processing jobs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..models import JobSpec, bbox_around_point_m

# Required columns in input XLSX
REQUIRED_COLS = ["POINT_ID", "TH_LAT", "TH_LONG", "SURVEY_DATE"]

def _normalize_point_id(x: Any) -> str:
    """
    Normalize point ID from various input formats

    Args:
        x: Raw point ID value

    Returns:
        Cleaned point ID string
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

@dataclass
class XlsxBatchConfig:
    """Configuration for XLSX batch processing"""
    xlsx_path: Path
    sheet: Any  # Sheet name or index
    start_date: str = "2015-01-01"  # Fixed start date for all points
    end_date: str = "2018-12-31"    # Fixed end date for all points
    bbox_size_m: float = 30.0
    limit: int = -1  # -1 for no limit
    # Filtering thresholds
    ndvi_threshold: float = 0.2
    mndwi_threshold: float = 0.0
    sun_zenith_threshold: float = 70.0
    coverage_threshold: float = 0.8
    scl_exclude_classes: List[int] = None  # SCL classes to exclude

    def __post_init__(self):
        if self.scl_exclude_classes is None:
            # Default: exclude clouds, shadows, water, snow/ice
            self.scl_exclude_classes = [3, 6, 8, 9, 10, 11]

class XlsxBatchProcessor:
    """
    Processes Excel files to create statistics API jobs
    """

    def __init__(self, config: XlsxBatchConfig):
        self.config = config

    def _validate_input_file(self) -> pd.DataFrame:
        """Read and validate input Excel file"""
        if not self.config.xlsx_path.exists():
            raise FileNotFoundError(f"XLSX file not found: {self.config.xlsx_path}")

        # Read Excel file
        df = pd.read_excel(self.config.xlsx_path, sheet_name=self.config.sheet)

        # Check required columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in XLSX: {missing}")

        return df

    def _create_jobs_from_dataframe(self, df: pd.DataFrame) -> List[JobSpec]:
        """Create job specifications from DataFrame rows"""
        jobs: List[JobSpec] = []

        # Apply limit if specified
        if self.config.limit > 0:
            df = df.head(self.config.limit)

        for _, row in df.iterrows():
            try:
                point_id = _normalize_point_id(row["POINT_ID"])
                if not point_id:
                    continue  # Skip empty point IDs

                lat = float(row["TH_LAT"])
                lon = float(row["TH_LONG"])


                start_date = self.config.start_date
                end_date = self.config.end_date

                # Create bounding box
                bbox = bbox_around_point_m(lat, lon, self.config.bbox_size_m)

                # Create job specification with filtering thresholds
                job_id = uuid.uuid4().hex[:10]
                job = JobSpec(
                    job_id=job_id,
                    point_id=point_id,
                    lat=lat,
                    lon=lon,
                    start_date=start_date,
                    end_date=end_date,
                    bbox=bbox,
                    ndvi_threshold=self.config.ndvi_threshold,
                    mndwi_threshold=self.config.mndwi_threshold,
                    sun_zenith_threshold=self.config.sun_zenith_threshold,
                    coverage_threshold=self.config.coverage_threshold,
                    scl_exclude_classes=self.config.scl_exclude_classes
                )

                jobs.append(job)

            except Exception as e:
                # Skip rows with errors (will be logged separately)
                continue

        return jobs

    def process(self) -> List[JobSpec]:
        """
        Process the Excel file and return job specifications

        Returns:
            List of JobSpec objects ready for execution

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If required columns are missing
        """
        df = self._validate_input_file()
        df = df.copy()

        # Clean data
        df["POINT_ID"] = df["POINT_ID"].apply(_normalize_point_id)
        df = df[df["POINT_ID"] != ""]  # Remove empty point IDs
        df = df.dropna(subset=["TH_LAT", "TH_LONG", "SURVEY_DATE"])  # Remove rows with missing required data

        # Create jobs
        jobs = self._create_jobs_from_dataframe(df)

        return jobs

def create_xlsx_processor(
    xlsx_path: Path,
    sheet: Any,
    *,
    start_date: str = "2015-01-01",
    end_date: str = "2018-12-31",
    bbox_size_m: float = 30.0,
    limit: int = -1,
    ndvi_threshold: float = 0.2,
    mndwi_threshold: float = 0.0,
    sun_zenith_threshold: float = 70.0,
    coverage_threshold: float = 0.8,
    scl_exclude_classes: List[int] = None
) -> XlsxBatchProcessor:
    """
    Factory function to create XLSX batch processor

    Args:
        xlsx_path: Path to Excel file
        sheet: Sheet name or index
        start_date: Fixed start date for all points (YYYY-MM-DD)
        end_date: Fixed end date for all points (YYYY-MM-DD)
        bbox_size_m: Size of bounding box in meters
        limit: Maximum number of rows to process (-1 for all)
        ndvi_threshold: NDVI threshold for filtering
        mndwi_threshold: MNDWI threshold for filtering
        sun_zenith_threshold: Sun zenith angle threshold for filtering
        coverage_threshold: Minimum coverage threshold
        scl_exclude_classes: SCL classes to exclude from analysis

    Returns:
        Configured XlsxBatchProcessor instance
    """
    config = XlsxBatchConfig(
        xlsx_path=xlsx_path,
        sheet=sheet,
        start_date=start_date,
        end_date=end_date,
        bbox_size_m=bbox_size_m,
        limit=limit,
        ndvi_threshold=ndvi_threshold,
        mndwi_threshold=mndwi_threshold,
        sun_zenith_threshold=sun_zenith_threshold,
        coverage_threshold=coverage_threshold,
        scl_exclude_classes=scl_exclude_classes
    )

    return XlsxBatchProcessor(config)
