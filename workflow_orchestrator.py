from bare_soil_aggregate_pixels import *
from extract_time_series import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def run_pipeline(
    *,
    seeds_table_path: str | Path,
    out_dir: Path = Path("./data/outputs"),
    N: int = 2,
    window_days: int = 30,
    res_m: float = 10.0,
    ndvi_threshold: float = 0.2,
    min_obs: int = 2,
    config_path: str = "configs/dev.yaml",
):
    logger = setup_logger(log_dir=Path("logs"), name="sentinel_soil_pipeline")

    # 1) load seeds (from our previously developed loader)
    seeds = load_seeds_from_table(seeds_table_path)
    logger.info(f"Loaded {len(seeds)} seeds from {seeds_table_path}")

    # also load the original df so we can enrich it
    p = Path(seeds_table_path)
    df = pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p, dtype=str)

    # 2) per seed: extract -> compute features -> collect seed_summary rows
    summaries: List[pd.DataFrame] = []
    failures: List[tuple[str, str]] = []

    for s in seeds:
        try:
            # 2a) extraction (your existing function)
            ts_root = extract_one(
                lat=s.lat,
                lon=s.lon,
                N=N,
                survey_date=s.survey_date,
                window_days=window_days,
                res_m=res_m,
                config_path=config_path,
                seed_id=s.seed_id,
            )

            # 2b) features
            _, seed_summary_df = compute_baresoil_features_from_ts_root(
                ts_root=ts_root,
                seed_id=s.seed_id,
                lat=s.lat,
                lon=s.lon,
                ndvi_threshold=ndvi_threshold,
                min_obs=min_obs,
                base_dir=Path("./data"),
                logger=logger,
            )
            summaries.append(seed_summary_df)

        except Exception as e:
            logger.exception(f"Seed {s.seed_id} failed")
            failures.append((s.seed_id, str(e)))

    if not summaries:
        raise RuntimeError("No successful seeds. Check logs.")

    summary_all = pd.concat(summaries, ignore_index=True)

    # 3) enrich original seed df
    # Ensure join key type consistency
    df["POINT_ID"] = df["POINT_ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    summary_all["seed_id"] = summary_all["seed_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    print(df)
    print(seed_summary_df)
    print(summary_all)


    enriched = df.merge(summary_all, left_on="POINT_ID", right_on="seed_id", how="left")

    print(enriched)

    # 4) save outputs (xlsx/csv/parquet)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_xlsx = out_dir / "seeds_enriched.xlsx"
    enriched_parquet = out_dir / "seeds_enriched.parquet"
    enriched_csv = out_dir / "seeds_enriched.csv"
    failures_csv = out_dir / "failures.csv"

    enriched.to_excel(enriched_xlsx, index=False)
    enriched.to_parquet(enriched_parquet, index=False)
    enriched.to_csv(enriched_csv, index=False)

    if failures:
        pd.DataFrame(failures, columns=["seed_id", "error"]).to_csv(failures_csv, index=False)

    logger.info(f"✅ Enriched table written: {enriched_xlsx}")
    logger.info(f"✅ Global parquet written: {enriched_parquet}")
    if failures:
        logger.warning(f"Some seeds failed. See: {failures_csv}")


if __name__ == "__main__":
    log_root = Path("logs")        # or cfg.data.base_dir / "logs"
    logger = setup_logger(log_dir=log_root)
    run_pipeline(seeds_table_path="data\seed_data\seed.xlsx")