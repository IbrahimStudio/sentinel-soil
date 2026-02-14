# Soil Texture Prediction Pipeline

A complete, reproducible pipeline for building feature datasets from Sentinel-2 data, running PCA analysis, and training ML models to predict soil texture properties (clay, silt, sand).

## Project Structure

```
Algorithms/
├── src/
│   ├── io/                  # Data I/O modules
│   ├── data/                # Dataset building
│   ├── analysis/            # PCA and exploratory analysis
│   ├── modeling/            # ML model training
│   └── utils/               # Utilities and validation
├── scripts/                 # CLI entrypoints
├── outputs/                 # Generated artifacts (gitignored)
│   ├── cache/               # Cached intermediate data
│   ├── datasets/            # Final datasets
│   ├── reports/             # Analysis reports
│   └── models/              # Trained models
├── data/                    # Local input data (gitignored)
│   └── gabri_filters.xlsx   # Excel file with target data
├── .env.example             # Environment variable template
├── pyproject.toml           # Python dependencies
└── README.md                # This file
```

## Setup

### 1. Install dependencies

```bash
cd Algorithms
pip install -e .
```

### 2. Set up environment variables

Copy `.env.example` to `.env` and fill in your Scaleway S3 credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `S3_ENDPOINT_URL` - Scaleway S3 endpoint
- `AWS_ACCESS_KEY_ID` - Access key
- `AWS_SECRET_ACCESS_KEY` - Secret key
- `AWS_REGION` - (optional) AWS region
- `S3_BUCKET` - Bucket name (default: soil-sentinel)

### 3. Prepare input data

Place the Excel file `gabri_filters.xlsx` in the `Algorithms/data/` directory.

## Usage

### Run the complete pipeline

```bash
cd Algorithms
python -m scripts.run_all run-all
```

### Run individual steps

```bash
# Download features from Scaleway S3
python -m scripts.run_all download-features

# Build dataset (join Sentinel-2 features with Excel targets)
python -m scripts.run_all build-dataset

# Run PCA analysis
python -m scripts.run_all pca

# Train ML models
python -m scripts.run_all train
```

### Additional options

```bash
# Refresh cache (force re-download from S3)
python -m scripts.run_all download-features --refresh

# Use PCA features for modeling
python -m scripts.run_all train --use-pca
```

## Outputs

All outputs are written to `Algorithms/outputs/`:

- **Cache**: `outputs/cache/features_raw.parquet` - Raw Sentinel-2 features
- **Datasets**: `outputs/datasets/dataset.parquet` - Final joined dataset
- **Reports**: `outputs/reports/join_report.json` - Join validation report
- **PCA**: `outputs/analysis/pca_*` - PCA analysis results and plots
- **Models**: `outputs/models/{target}/{model}.joblib` - Trained models
- **Metrics**: `outputs/models/{target}/metrics.json` - Model performance metrics
- **Plots**: `outputs/models/{target}/plots/` - Feature importance and prediction plots

## Configuration

Create a `config.yaml` file to customize:

```yaml
s3:
  bucket: soil-sentinel
  prefix: soil-sentinel/batch_results_2015_2018_scl_ndvi/aggregated/

targets:
  - clay
  - silt
  - sand

features:
  include: null  # List of features to include, or null for all
  exclude: null  # List of features to exclude

models:
  random_forest:
    n_estimators: [100, 200]
    max_depth: [10, 20, None]
  gradient_boosting:
    n_estimators: [100, 200]
    learning_rate: [0.05, 0.1]
```

## Implementation Details

### Data Pipeline

1. **Feature Extraction**: Downloads JSON files from Scaleway S3, parses Sentinel-2 band/index data
2. **Dataset Building**: Joins Sentinel-2 features with Excel target data on POINT_ID
3. **PCA Analysis**: Exploratory dimensionality reduction with visualization
4. **Model Training**: Separate regressions for each target using RandomForest and GradientBoosting

### Key Features

- **Robust Error Handling**: Comprehensive logging and validation
- **Deterministic Runs**: Fixed random seeds for reproducibility
- **Caching**: Avoids re-downloading data when not needed
- **Modular Design**: Each component can be run independently
- **Configuration**: Flexible through config files and CLI arguments