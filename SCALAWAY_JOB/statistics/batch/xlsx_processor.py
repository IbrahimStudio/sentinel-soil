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
    window_days: int = 15
    bbox_size_m: float = 30.0
    limit: int = -1  # -1 for no limit

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

                # Parse survey date and create time window
                survey_dt = pd.to_datetime(row["SURVEY_DATE"], errors="coerce")
                if pd.isna(survey_dt):
                    continue  # Skip invalid dates

                center = pd.to_datetime(survey_dt).normalize()
                start_date = (center - pd.Timedelta(days=self.config.window_days)).strftime("%Y-%m-%d")
                end_date = (center + pd.Timedelta(days=self.config.window_days)).strftime("%Y-%m-%d")

                # Create bounding box
                bbox = bbox_around_point_m(lat, lon, self.config.bbox_size_m)

                # Create job specification
                job_id = uuid.uuid4().hex[:10]
                job = JobSpec(
                    job_id=job_id,
                    point_id=point_id,
                    lat=lat,
                    lon=lon,
                    start_date=start_date,
                    end_date=end_date,
                    bbox=bbox
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
    window_days: int = 15,
    bbox_size_m: float = 30.0,
    limit: int = -1
) -> XlsxBatchProcessor:
    """
    Factory function to create XLSX batch processor

    Args:
        xlsx_path: Path to Excel file
        sheet: Sheet name or index
        window_days: +/- days around survey date
        bbox_size_m: Size of bounding box in meters
        limit: Maximum number of rows to process (-1 for all)

    Returns:
        Configured XlsxBatchProcessor instance
    """
    config = XlsxBatchConfig(
        xlsx_path=xlsx_path,
        sheet=sheet,
        window_days=window_days,
        bbox_size_m=bbox_size_m,
        limit=limit
    )

    return XlsxBatchProcessor(config)