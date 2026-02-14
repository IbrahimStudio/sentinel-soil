import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import yaml

# Set up logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trainer for soil texture prediction models."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the model trainer.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.targets = self.config.get('targets', ['clay', 'silt', 'sand'])
        self.models_config = self.config.get('models', {})
        self.random_state = 42

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Parsed configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {config_path}")
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def prepare_data(self, dataset_df: pd.DataFrame, use_pca: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for modeling.

        Args:
            dataset_df: Input dataset
            use_pca: Whether to use PCA features instead of raw features

        Returns:
            Tuple of (features DataFrame, target column names)
        """
        # Select features
        if use_pca:
            # Use PCA features if available
            pca_cols = [col for col in dataset_df.columns if col.startswith('PC')]
            if not pca_cols:
                raise ValueError("No PCA features found in dataset. Run PCA first.")
            features_df = dataset_df[pca_cols]
        else:
            # Use raw Sentinel-2 features
            exclude_patterns = [
                'point_id', 'lat', 'lon', 'n_days_total', 'n_days_kept',
                'kept_ratio', 'coverage_median_kept', 'coverage_min_kept',
                'clay', 'silt', 'sand', 'PC'  # Exclude PCA features and targets
            ]

            feature_cols = []
            for col in dataset_df.columns:
                if not any(pattern in col for pattern in exclude_patterns):
                    if pd.api.types.is_numeric_dtype(dataset_df[col]):
                        feature_cols.append(col)

            if not feature_cols:
                raise ValueError("No numeric feature columns found for modeling")

            features_df = dataset_df[feature_cols]

        # Validate targets
        available_targets = [target for target in self.targets if target in dataset_df.columns]
        if not available_targets:
            raise ValueError(f"No target columns found in dataset. Expected: {self.targets}")

        logger.info(f"Prepared data: {len(feature_cols)} features, {len(available_targets)} targets")
        return features_df, available_targets

    def train_test_split_data(self, features_df: pd.DataFrame, target_series: pd.Series) -> Tuple:
        """Split data into train and test sets.

        Args:
            features_df: Features DataFrame
            target_series: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            features_df, target_series,
            test_size=0.2,
            random_state=self.random_state
        )

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   model_name: str, target_name: str) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training targets
            model_name: Name of model to train ('random_forest' or 'gradient_boosting')
            target_name: Name of target variable

        Returns:
            Dictionary with training results and best model
        """
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not configured")

        model_config = self.models_config[model_name]

        # Get model class and parameter grid
        if model_name == 'random_forest':
            model = RandomForestRegressor(random_state=self.random_state)
            param_grid = {
                'n_estimators': model_config.get('n_estimators', [100, 200]),
                'max_depth': model_config.get('max_depth', [10, 20, None])
            }
        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=self.random_state)
            param_grid = {
                'n_estimators': model_config.get('n_estimators', [100, 200]),
                'learning_rate': model_config.get('learning_rate', [0.05, 0.1])
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Store results
        results = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'feature_names': list(X_train.columns)
        }

        logger.info(f"Trained {model_name} for {target_name}: best params = {grid_search.best_params_}")
        return results

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a trained model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'n_samples': len(y_test)
        }

        return metrics

    def get_feature_importance(self, model, feature_names: List[str], model_name: str) -> pd.DataFrame:
        """Get feature importance for a trained model.

        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model

        Returns:
            DataFrame with feature importance scores
        """
        if model_name == 'random_forest':
            # Use feature importances from RandomForest
            importances = model.feature_importances_
        elif model_name == 'gradient_boosting':
            # Try to get feature importances, fall back to permutation importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                # Use permutation importance as fallback
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=self.random_state)
                importances = result.importances_mean
        else:
            raise ValueError(f"Unknown model: {model_name}")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_feature_importance(self, importance_df: pd.DataFrame, output_dir: Path,
                              model_name: str, target_name: str) -> None:
        """Plot feature importance.

        Args:
            importance_df: DataFrame with feature importance
            output_dir: Directory to save the plot
            model_name: Name of the model
            target_name: Name of the target variable
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name} - {target_name}')
        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'feature_importance_{model_name}_{target_name}.png'
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved feature importance plot to: {output_path}")

    def plot_predictions(self, y_test: pd.Series, y_pred: pd.Series,
                        output_dir: Path, model_name: str, target_name: str) -> None:
        """Plot predictions vs true values.

        Args:
            y_test: True target values
            y_pred: Predicted target values
            output_dir: Directory to save the plot
            model_name: Name of the model
            target_name: Name of the target variable
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs True Values - {model_name} - {target_name}')

        # Add metrics to plot
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}',
                transform=plt.gca().transAxes, verticalalignment='top')

        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'predictions_{model_name}_{target_name}.png'
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved predictions plot to: {output_path}")

    def plot_residuals(self, y_test: pd.Series, y_pred: pd.Series,
                      output_dir: Path, model_name: str, target_name: str) -> None:
        """Plot residuals.

        Args:
            y_test: True target values
            y_pred: Predicted target values
            output_dir: Directory to save the plot
            model_name: Name of the model
            target_name: Name of the target variable
        """
        residuals = y_test - y_pred

        plt.figure(figsize=(12, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residuals Plot - {model_name} - {target_name}')

        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'residuals_{model_name}_{target_name}.png'
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved residuals plot to: {output_path}")

    def train_all_models(self, dataset_df: pd.DataFrame, use_pca: bool = False) -> Dict[str, Any]:
        """Train all models for all targets.

        Args:
            dataset_df: Input dataset
            use_pca: Whether to use PCA features

        Returns:
            Dictionary with all training results
        """
        # Prepare data
        features_df, available_targets = self.prepare_data(dataset_df, use_pca)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'use_pca': use_pca,
            'n_features': len(features_df.columns),
            'feature_names': list(features_df.columns),
            'targets': {},
            'models': list(self.models_config.keys())
        }

        # Train models for each target
        for target in available_targets:
            logger.info(f"Training models for target: {target}")

            # Filter out rows with missing target values
            target_mask = dataset_df[target].notna()
            if not target_mask.any():
                logger.warning(f"No valid data for target {target}, skipping")
                continue

            X = features_df[target_mask]
            y = dataset_df.loc[target_mask, target]

            # Split data
            X_train, X_test, y_train, y_test = self.train_test_split_data(X, y)

            target_results = {
                'n_samples_total': len(y),
                'n_samples_train': len(y_train),
                'n_samples_test': len(y_test),
                'models': {}
            }

            # Train each model type
            for model_name in self.models_config.keys():
                logger.info(f"Training {model_name} for {target}")

                # Train model
                train_results = self.train_model(X_train, y_train, model_name, target)
                model = train_results['model']

                # Evaluate on test set
                test_metrics = self.evaluate_model(model, X_test, y_test)

                # Get feature importance
                importance_df = self.get_feature_importance(
                    model, train_results['feature_names'], model_name
                )

                # Create output directory for this target/model
                target_model_dir = Path(f'outputs/models/{target}/{model_name}')
                target_model_dir.mkdir(parents=True, exist_ok=True)

                # Save model
                model_path = target_model_dir / f'{model_name}_{target}.joblib'
                joblib.dump(model, model_path)

                # Save metrics
                metrics_path = target_model_dir / 'metrics.json'
                metrics_report = {
                    'cv_metrics': {
                        'best_params': train_results['best_params'],
                        'best_score': train_results['best_score']
                    },
                    'test_metrics': test_metrics
                }
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_report, f, indent=2)

                # Create plots
                y_pred = model.predict(X_test)
                self.plot_feature_importance(importance_df, target_model_dir, model_name, target)
                self.plot_predictions(y_test, y_pred, target_model_dir, model_name, target)
                self.plot_residuals(y_test, y_pred, target_model_dir, model_name, target)

                # Store results
                target_results['models'][model_name] = {
                    'best_params': train_results['best_params'],
                    'cv_best_score': train_results['best_score'],
                    'test_metrics': test_metrics,
                    'feature_importance': importance_df.to_dict('records'),
                    'model_path': str(model_path),
                    'metrics_path': str(metrics_path)
                }

            all_results['targets'][target] = target_results

        return all_results

def get_model_trainer(config_path: str = 'config.yaml') -> ModelTrainer:
    """Get a configured model trainer instance."""
    return ModelTrainer(config_path)