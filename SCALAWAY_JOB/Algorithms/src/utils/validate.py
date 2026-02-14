import logging
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class DataValidator:
    """Validator for dataset quality and completeness."""

    def __init__(self):
        """Initialize the validator."""
        pass

    def validate_dataset(self, dataset_df: pd.DataFrame, required_columns: Dict[str, str]) -> Dict[str, Any]:
        """Validate dataset structure and content.

        Args:
            dataset_df: Dataset to validate
            required_columns: Dictionary of required column patterns and descriptions

        Returns:
            Dictionary with validation results
        """
        validation = {
            'row_count': len(dataset_df),
            'columns': list(dataset_df.columns),
            'missing_values': {},
            'required_columns_present': {},
            'numeric_columns': [],
            'warnings': [],
            'errors': []
        }

        # Check row count
        if validation['row_count'] == 0:
            validation['errors'].append("Dataset is empty")
            return validation

        # Check for missing values
        missing_values = dataset_df.isnull().sum()
        validation['missing_values'] = missing_values[missing_values > 0].to_dict()

        if validation['missing_values']:
            logger.warning(f"Found missing values in columns: {list(validation['missing_values'].keys())}")

        # Check required columns
        for col_pattern, description in required_columns.items():
            found = any(col_pattern in col for col in dataset_df.columns)
            validation['required_columns_present'][col_pattern] = found
            if not found:
                validation['warnings'].append(f"Missing required column pattern: {col_pattern} ({description})")

        # Check numeric columns
        numeric_cols = dataset_df.select_dtypes(include=['number']).columns
        validation['numeric_columns'] = list(numeric_cols)

        if len(numeric_cols) < 5:
            validation['warnings'].append(f"Only {len(numeric_cols)} numeric columns found")

        # Check for constant columns
        constant_cols = dataset_df.columns[dataset_df.nunique() <= 1]
        if len(constant_cols) > 0:
            validation['warnings'].append(f"Found constant columns: {list(constant_cols)}")

        # Log validation summary
        logger.info(f"Dataset validation: {validation['row_count']} rows, {len(validation['columns'])} columns")
        if validation['errors']:
            logger.error(f"Validation errors: {validation['errors']}")
        if validation['warnings']:
            logger.warning(f"Validation warnings: {validation['warnings']}")

        return validation

    def validate_targets(self, dataset_df: pd.DataFrame, target_columns: List[str]) -> Dict[str, Any]:
        """Validate target columns.

        Args:
            dataset_df: Dataset to validate
            target_columns: List of target column names

        Returns:
            Dictionary with target validation results
        """
        target_validation = {
            'targets_present': {},
            'target_statistics': {},
            'warnings': [],
            'errors': []
        }

        for target in target_columns:
            present = target in dataset_df.columns
            target_validation['targets_present'][target] = present

            if present:
                target_series = dataset_df[target]
                stats = {
                    'dtype': str(target_series.dtype),
                    'not_null_count': target_series.notna().sum(),
                    'null_count': target_series.isna().sum(),
                    'min': target_series.min() if pd.api.types.is_numeric_dtype(target_series) else None,
                    'max': target_series.max() if pd.api.types.is_numeric_dtype(target_series) else None,
                    'mean': target_series.mean() if pd.api.types.is_numeric_dtype(target_series) else None,
                    'std': target_series.std() if pd.api.types.is_numeric_dtype(target_series) else None
                }
                target_validation['target_statistics'][target] = stats

                # Check for issues
                if stats['null_count'] > 0:
                    target_validation['warnings'].append(f"Target {target} has {stats['null_count']} null values")

                if pd.api.types.is_numeric_dtype(target_series):
                    if stats['std'] == 0:
                        target_validation['errors'].append(f"Target {target} has zero variance")
                    if stats['not_null_count'] < 10:
                        target_validation['errors'].append(f"Target {target} has only {stats['not_null_count']} valid samples")
            else:
                target_validation['errors'].append(f"Target column {target} not found in dataset")

        return target_validation

def get_data_validator() -> DataValidator:
    """Get a configured data validator instance."""
    return DataValidator()