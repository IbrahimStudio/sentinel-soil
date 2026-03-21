import pandas as pd
import matplotlib.pyplot as plt

def build_predictors_corr_from_target_corr(
    corr_csv_path: str,
    out_corr_csv_path: str = "corr__predictors_only.csv",
    out_png_path: str = "corr__predictors_only.png",
    target_col: str | None = None,     # e.g. "Sand" / "Clay" / "Silt" / "Coarse"
    figsize=(10, 9),
    dpi=200,
):
    """
    Given a correlation-matrix CSV like corr__Sand__predictors.csv (square matrix with
    predictors + target), remove the target row/col (if present) and save:
      - predictors-only correlation matrix CSV
      - heatmap PNG

    Assumptions:
      - First column is the row index (names)
      - Column headers match row index names
    """
    # Read correlation matrix
    corr = pd.read_csv(corr_csv_path, index_col=0)

    # Try to auto-detect target if not provided:
    # - If there's exactly one column that is NOT in the known predictors set you can pass it,
    #   but here we do a safe approach: if target_col None, do nothing unless it matches last col name
    if target_col is None:
        # Heuristic: if matrix includes a target, it's often the last column/row (as in your example)
        # We'll pick it only if it appears both as a column and index AND its name doesn't start with "p50_"
        candidate = corr.columns[-1]
        if candidate in corr.index and not str(candidate).startswith("p50_") and candidate not in ("Elev", "lat", "lon"):
            target_col = candidate

    # Drop target row/col if present
    if target_col is not None and target_col in corr.columns and target_col in corr.index:
        corr_pred = corr.drop(index=target_col, columns=target_col)
    else:
        corr_pred = corr  # already predictors-only

    # Save predictors-only correlation matrix
    corr_pred.to_csv(out_corr_csv_path)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_pred.values, vmin=-1, vmax=1)  # no custom colors; default colormap

    ax.set_title("Correlation matrix (Pearson) - Predictors only")

    # Ticks
    labels = corr_pred.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=dpi)
    plt.close(fig)

    return corr_pred


if __name__ == "__main__":
    # Example usage:
    corr_pred = build_predictors_corr_from_target_corr(
        corr_csv_path=r"Algorithms\reports\sand_run_20260218_131359\correlations\corr__Sand__predictors.csv",
        out_corr_csv_path=r"Algorithms\reports\corr__predictors_only.csv",
        out_png_path=r"Algorithms\reports\corr__predictors_only.png",
        target_col="Sand",  # you can omit this if you want auto-detect
    )
    print("Saved predictors-only correlation to:", "corr__predictors_only.csv")
    print("Saved predictors-only heatmap to:", "corr__predictors_only.png")