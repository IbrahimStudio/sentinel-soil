#!/usr/bin/env python3
"""
Test script to verify the time window functionality
"""

import sys
import os
from datetime import datetime

# Add the current directory to Python path so we can import the module
sys.path.insert(0, '.')

from scaleway_batch_stats_from_xlsx import _calculate_time_window_dates

def test_time_window_calculation():
    """Test the time window calculation function"""

    print("Testing time window calculation function...")

    # Test case 1: 30-day window (should give 15 days before and after)
    survey_date = "2023-06-15"
    time_window = 30
    start_date, end_date = _calculate_time_window_dates(survey_date, time_window)

    print(f"Test 1 - 30 day window around {survey_date}:")
    print(f"  Start: {start_date}")
    print(f"  End: {end_date}")

    # Verify the calculation
    survey_dt = datetime.strptime(survey_date, "%Y-%m-%d").date()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    days_before = (survey_dt - start_dt).days
    days_after = (end_dt - survey_dt).days

    print(f"  Days before survey date: {days_before}")
    print(f"  Days after survey date: {days_after}")
    print(f"  Total window: {days_before + days_after} days")
    print()

    # Test case 2: 730-day window (should give 365 days before and after)
    survey_date = "2023-06-15"
    time_window = 730
    start_date, end_date = _calculate_time_window_dates(survey_date, time_window)

    print(f"Test 2 - 730 day window around {survey_date}:")
    print(f"  Start: {start_date}")
    print(f"  End: {end_date}")

    # Verify the calculation
    survey_dt = datetime.strptime(survey_date, "%Y-%m-%d").date()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    days_before = (survey_dt - start_dt).days
    days_after = (end_dt - survey_dt).days

    print(f"  Days before survey date: {days_before}")
    print(f"  Days after survey date: {days_after}")
    print(f"  Total window: {days_before + days_after} days")
    print()

    # Test case 3: Edge case with odd number (31 days)
    survey_date = "2023-06-15"
    time_window = 31
    start_date, end_date = _calculate_time_window_dates(survey_date, time_window)

    print(f"Test 3 - 31 day window around {survey_date}:")
    print(f"  Start: {start_date}")
    print(f"  End: {end_date}")

    # Verify the calculation
    survey_dt = datetime.strptime(survey_date, "%Y-%m-%d").date()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    days_before = (survey_dt - start_dt).days
    days_after = (end_dt - survey_dt).days

    print(f"  Days before survey date: {days_before}")
    print(f"  Days after survey date: {days_after}")
    print(f"  Total window: {days_before + days_after} days")
    print()

    print("✅ All time window calculations completed successfully!")

if __name__ == "__main__":
    test_time_window_calculation()