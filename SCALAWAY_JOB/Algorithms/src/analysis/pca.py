import logging
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class PCAAnalyzer:
    """PCA analysis for exploratory data analysis."""

    def __init__(self, n_components: int = 2, config_path: str = 'config.yaml'):
        """Initialize the PCA analyzer.

        Args:
            n_components: Number of PCA components to compute
            config_path: Path to configuration file
        """
        self.n_components = n_components
        self.config_path = config_path
        self.pca = None
        self.scaler = None
        self.imputer = None
        self.pipeline = None
        self.feature_names = None
        self.explained_variance = None
        self.components = None
        self.exclude_columns = self._load_exclude_columns()

    def _load_exclude_columns(self) -> List[str]:
        """Load the list of columns to exclude from PCA from configuration.

        Returns:
            List of column names to exclude from PCA analysis
        """
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}

            exclude_columns = config.get('pca', {}).get('exclude_columns', [])
            logger.info(f"Loaded PCA exclude columns from config: {exclude_columns}")
            return exclude_columns
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using default exclusions")
            return [
                'point_id', 'lat', 'lon', 'n_days_total', 'n_days_kept',
                'kept_ratio', 'coverage_median_kept', 'coverage_min_kept',
                'clay', 'silt', 'sand', 'coarse',
                'Clay', 'Silt', 'Sand', 'Coarse'
            ]
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return [
                'point_id', 'lat', 'lon', 'n_days_total', 'n_days_kept',
                'kept_ratio', 'coverage_median_kept', 'coverage_min_kept',
                'clay', 'silt', 'sand', 'coarse',
                'Clay', 'Silt', 'Sand', 'Coarse'
            ]

    def prepare_features(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for PCA analysis.

        Args:
            dataset_df: Input dataset

        Returns:
            DataFrame with selected and prepared features
        """
        # Use the configured list of columns to exclude
        exclude_columns = self.exclude_columns

        # Select numeric feature columns
        feature_cols = []
        for col in dataset_df.columns:
            if col not in exclude_columns:
                if pd.api.types.is_numeric_dtype(dataset_df[col]):
                    feature_cols.append(col)

        if not feature_cols:
            raise ValueError("No numeric feature columns found for PCA")

        self.feature_names = feature_cols
        features_df = dataset_df[feature_cols]

        logger.info(f"Selected {len(feature_cols)} features for PCA: {feature_cols}")
        return features_df

    def run_pca(self, features_df: pd.DataFrame) -> None:
        """Run PCA analysis on the feature matrix.

        Args:
            features_df: DataFrame with features to analyze
        """
        # Create preprocessing pipeline
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.n_components))
        ])

        # Fit the pipeline
        self.pipeline.fit(features_df)
        self.imputer = self.pipeline.named_steps['imputer']
        self.scaler = self.pipeline.named_steps['scaler']
        self.pca = self.pipeline.named_steps['pca']

        # Store results
        self.explained_variance = self.pca.explained_variance_ratio_
        self.components = self.pca.components_

        logger.info(f"PCA completed. Explained variance: {self.explained_variance}")

    def get_pca_results(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Get PCA-transformed data.

        Args:
            features_df: Original features DataFrame

        Returns:
            DataFrame with PCA components
        """
        if self.pipeline is None:
            raise ValueError("PCA has not been run yet. Call run_pca() first.")

        pca_results = self.pipeline.transform(features_df)
        pca_df = pd.DataFrame(
            pca_results,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=features_df.index
        )

        return pca_df

    def plot_explained_variance(self, output_dir: Path) -> None:
        """Plot explained variance by PCA components.

        Args:
            output_dir: Directory to save the plot
        """
        if self.explained_variance is None:
            raise ValueError("PCA has not been run yet. Call run_pca() first.")

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(self.explained_variance) + 1), self.explained_variance, alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        plt.xticks(range(1, len(self.explained_variance) + 1))

        # Add cumulative variance line
        cumulative_variance = np.cumsum(self.explained_variance)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-', marker='o')
        plt.ylabel('Explained Variance Ratio / Cumulative')

        output_path = output_dir / 'pca_explained_variance.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved explained variance plot to: {output_path}")

    def plot_pca_scatter(self, pca_df: pd.DataFrame, dataset_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot PCA scatter plot colored by target variable if available.

        Args:
            pca_df: DataFrame with PCA results
            dataset_df: Original dataset with target variables
            output_dir: Directory to save the plot
        """
        plt.figure(figsize=(12, 8))

        # Try to find a target column to color by
        target_col = None
        for col in ['clay', 'silt', 'sand']:
            if col in dataset_df.columns:
                target_col = col
                break

        if target_col and pd.api.types.is_numeric_dtype(dataset_df[target_col]):
            # Create scatter plot colored by target
            scatter = plt.scatter(
                pca_df['PC1'], pca_df['PC2'],
                c=dataset_df[target_col],
                cmap='viridis',
                alpha=0.7
            )
            plt.colorbar(scatter, label=target_col)
            title = f'PCA Scatter Plot (Colored by {target_col})'
        else:
            # Simple scatter plot
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
            title = 'PCA Scatter Plot'

        plt.xlabel(f'PC1 ({self.explained_variance[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({self.explained_variance[1]*100:.1f}%)')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        output_path = output_dir / 'pca_scatter_plot.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved PCA scatter plot to: {output_path}")

    def save_pca_results(self, pca_df: pd.DataFrame, output_dir: Path) -> None:
        """Save PCA results to CSV files.

        Args:
            pca_df: DataFrame with PCA results
            output_dir: Directory to save outputs
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PCA components - handle case where some features were dropped during imputation
        try:
            components_df = pd.DataFrame(
                self.components,
                columns=self.feature_names,
                index=[f'PC{i+1}' for i in range(self.n_components)]
            )
        except ValueError:
            # If shape mismatch, use only the features that were actually used
            num_used_features = self.components.shape[1]
            used_feature_names = self.feature_names[:num_used_features]
            components_df = pd.DataFrame(
                self.components,
                columns=used_feature_names,
                index=[f'PC{i+1}' for i in range(self.n_components)]
            )

        components_path = output_dir / 'pca_components.csv'
        components_df.to_csv(components_path)
        logger.info(f"Saved PCA components to: {components_path}")

        # Save explained variance
        variance_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(self.n_components)],
            'explained_variance': self.explained_variance,
            'cumulative_variance': np.cumsum(self.explained_variance)
        })
        variance_path = output_dir / 'pca_explained_variance.csv'
        variance_df.to_csv(variance_path, index=False)
        logger.info(f"Saved PCA explained variance to: {variance_path}")

        # Save PCA results
        pca_results_path = output_dir / 'pca_results.csv'
        pca_df.to_csv(pca_results_path)
        logger.info(f"Saved PCA results to: {pca_results_path}")

    def create_pca_report(self, output_dir: Path) -> Dict[str, Any]:
        """Create a report of PCA analysis.

        Args:
            output_dir: Directory to save the report

        Returns:
            Dictionary with PCA analysis results
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_components': self.n_components,
            'explained_variance': self.explained_variance.tolist(),
            'cumulative_variance': np.cumsum(self.explained_variance).tolist(),
            'feature_names': self.feature_names,
            'total_variance_explained': np.sum(self.explained_variance)
        }

        report_path = output_dir / 'pca_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved PCA report to: {report_path}")
        return report

def get_pca_analyzer(n_components: int = 2) -> PCAAnalyzer:
    """Get a configured PCA analyzer instance."""
    return PCAAnalyzer(n_components)