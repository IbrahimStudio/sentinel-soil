import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor



def pca_regression(
    df: pd.DataFrame,
    target_column: str,
    *,
    id_columns=("POINT_ID",),
    drop_columns=(),
    n_components: float | int = 0.95,
    regressor: RegressorMixin,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    PCA on predictors + arbitrary regressor to predict target_column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_column : str
        Name of target variable.
    id_columns : tuple[str]
        Columns to drop if present (IDs).
    drop_columns : tuple[str]
        Any additional columns to drop from predictors.
    n_components : float|int
        PCA components: float keeps variance (e.g. 0.95), int keeps #components.
    regressor : sklearn regressor instance
        Any sklearn regressor (LinearRegression, RandomForestRegressor, etc.)
    test_size : float
        Train/test split.
    random_state : int
        Reproducibility.

    Returns
    -------
    pipeline : sklearn Pipeline
    pca : sklearn PCA
    results : dict
        Metrics + PCA variance info.
    """

    if target_column not in df.columns:
        raise ValueError(f"target_column='{target_column}' not found in df columns.")

    data = df.copy()

    # Drop ID columns if present
    for c in id_columns:
        if c in data.columns:
            data = data.drop(columns=c)

    # Separate X/y
    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Drop any user-specified columns from X
    if drop_columns:
        X = X.drop(columns=list(drop_columns), errors="ignore")

    # Basic numeric-only guard (optional but usually helpful in this context)
    X = X.select_dtypes(include=[np.number])
    y = pd.to_numeric(y, errors="coerce")

    # Drop rows with missing target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Simple missing handling for predictors: drop rows with any NaNs
    # (If you prefer imputation, I can swap this for SimpleImputer.)
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("regressor", regressor),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    pca = pipeline.named_steps["pca"]

    results = {
        "target": target_column,
        "R2": float(r2),
        "RMSE": rmse,
        "n_components_used": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_count_before_pca": int(X.shape[1]),
    }

    return pipeline, pca, results

from sklearn.ensemble import RandomForestRegressor


def pca_random_forest(
    df: pd.DataFrame,
    target_column: str,
    *,
    id_columns=("POINT_ID",),
    drop_columns=(),
    n_components: float | int = 0.95,
    test_size: float = 0.2,
    random_state: int = 42,
    # RF params (sane defaults; tune later)
    n_estimators: int = 500,
    max_depth=None,
    min_samples_leaf: int = 1,
    max_features: str | float | int = "sqrt",
    n_jobs: int = -1,
):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
    )

    return pca_regression(
        df,
        target_column=target_column,
        id_columns=id_columns,
        drop_columns=drop_columns,
        n_components=n_components,
        regressor=rf,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == '__main__':
    feature_ds_path = "texture_silt_scl_ndvi_features.xlsx"
    df = pd.read_excel(feature_ds_path)
    

    # pipe, pca, res = pca_regression(
    #     df,
    #     target_column="Silt",
    #     n_components=0.95,
    #     regressor=RandomForestRegressor()
    # )

    # print(res)

    pipe, pca, res = pca_random_forest(
        df,
        target_column="Silt",
        n_components=0.95,
        n_estimators=800,
        min_samples_leaf=2
    )
    print(res)

