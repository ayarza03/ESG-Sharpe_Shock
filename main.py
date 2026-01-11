# ==========================================================================================
# main.py
# Runs the pipeline:
# 1) Load raw datasets
# 2) Build Table 0 (universe)
# 3) Build Table 1 (risk-free)
# 4) Build Table 2 (returns + rolling Sharpe)
# 5) Build Table 3 (ESG snapshot restricted to Table 2 + success)
# 6) Build Table 4 (final merged panel + market proxy + regimes)
# 7) Save outputs to data/processed/
# 8) EDA plots (Step 3)
# 9) Portfolio sorts EW (Step 4)
# ==========================================================================================

from pathlib import Path
import pandas as pd

from src.data_loader import (
    load_sp500,
    load_tb3ms,
    load_esg,
    build_table_0,
    build_table_1,
    build_table_2,
    build_table_3,
    build_table_4,
    save_table,
)

from src.evaluation import (
    plot_corr_heatmap_numeric,
    plot_scatter_esg_vs_mean_sharpe,
    build_esg_sorted_portfolios_ew,
    create_ew_portfolio_star_outputs,
    build_esg_sorted_portfolios_vw,
    create_vw_portfolio_star_outputs,
    save_ew_vs_vw_regime_bucket_comparison,
    export_regime_interaction_regressions,
    run_ml_with_without_esg,
    run_esg_total_vs_subscores_robustness_ml,
)


def assert_has_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[CHECK] {name} is missing columns: {missing}")


def main() -> None:
    # ----------------------------------------------------------
    # 0) Confirm you run from repo root
    # ----------------------------------------------------------
    root = Path(__file__).resolve().parent
    print(f"Running from: {root}")

    # ----------------------------------------------------------
    # 1) Load raw data
    # ----------------------------------------------------------
    sp500_raw = load_sp500("sp500_historical.csv")
    rf_raw = load_tb3ms("TB3MS.csv")
    esg_raw = load_esg("ESG_info.csv")

    print("\nLoaded raw datasets:")
    print(f" - sp500_raw shape: {sp500_raw.shape}")
    print(f" - rf_raw shape:    {rf_raw.shape}")
    print(f" - esg_raw shape:   {esg_raw.shape}")

    # ----------------------------------------------------------
    # 2) Build Table 0
    # ----------------------------------------------------------
    table0 = build_table_0(sp500_raw, n=100)

    print("\nTable 0 built:")
    print(f" - shape: {table0.shape}")
    print(f" - months: {table0['month_id'].nunique()}")
    print(" - sample:")
    print(table0.head(5))

    rows_per_month = table0.groupby("month_id").size()
    if (rows_per_month != 100).any():
        bad = rows_per_month[rows_per_month != 100].to_dict()
        print(f"[WARN] Table 0 months not equal to 100 rows: {bad}")
    else:
        print(" - OK: Table 0 has exactly 100 rows per month.")

    out0 = save_table(table0, "table_0.csv")
    print(f"Saved Table 0 to: {out0}")

    # ----------------------------------------------------------
    # 3) Build Table 1 (risk-free monthly)
    # ----------------------------------------------------------
    table1 = build_table_1(table0, rf_raw)

    print("\nTable 1 built:")
    print(f" - shape: {table1.shape}")
    print(" - sample:")
    print(table1.head(5))

    missing_rf_months = sorted(set(table0["month_id"]) - set(table1["month_id"]))
    if missing_rf_months:
        print(f"[WARN] Table 1 is missing rf for month_id(s): {missing_rf_months[:10]} ...")
    else:
        print(" - OK: Table 1 covers all Table 0 months.")

    out1 = save_table(table1, "table_1.csv")
    print(f"Saved Table 1 to: {out1}")

    # ----------------------------------------------------------
    # 4) Build Table 2 (returns + rolling Sharpe)
    # ----------------------------------------------------------
    table2 = build_table_2(table0, table1, window=12)

    print("\nTable 2 built:")
    print(f" - shape: {table2.shape}")
    print(" - sample:")
    print(table2.head(5))

    assert_has_columns(
        table2,
        ["month_id", "ticker", "monthly_return", "excess_return", "sharpe_12m", "rf_monthly"],
        "Table 2",
    )
    if table2["sharpe_12m"].isna().any():
        raise ValueError("[CHECK] Table 2 has NaN sharpe_12m (unexpected after dropna).")

    out2 = save_table(table2, "table_2.csv")
    print(f"Saved Table 2 to: {out2}")

    # ----------------------------------------------------------
    # 5) Build Table 3 (ESG snapshot restricted to Table 2)
    # ----------------------------------------------------------
    table3 = build_table_3(table2=table2, esg_raw=esg_raw)

    print("\nTable 3 built (ESG snapshot restricted to Table 2):")
    print(f" - shape: {table3.shape}")
    print(" - sample:")
    print(table3.head(5))

    assert_has_columns(
        table3,
        ["ticker", "esg_score", "environment_score", "social_score", "governance_score"],
        "Table 3",
    )

    out3 = save_table(table3, "table_3.csv")
    print(f"Saved Table 3 to: {out3}")

    # ----------------------------------------------------------
    # 6) Build Table 4 (final merged panel + market proxy + regimes)
    # ----------------------------------------------------------
    table4, market_monthly = build_table_4(table0, table2, table3)

    print("\nTable 4 built (final panel):")
    print(f" - shape: {table4.shape}")
    print(" - sample:")
    print(table4.head(5))

    # --- Sanity checks for new regime columns ---
    assert_has_columns(table4, ["regime", "year", "month"], "Table 4")
    assert_has_columns(market_monthly, ["regime", "year", "month"], "market_monthly")

    valid_regimes = {"normal", "covid", "tightening"}
    regimes_found = set(table4["regime"].dropna().unique().tolist())
    if not regimes_found.issubset(valid_regimes):
        raise ValueError(f"[CHECK] Unexpected regimes found in Table 4: {sorted(regimes_found)}")

    check_points = {201809: "normal", 202003: "covid", 202201: "tightening"}
    months_in_table4 = set(table4["month_id"].unique().tolist())
    for m, expected in check_points.items():
        if m in months_in_table4:
            observed = table4.loc[table4["month_id"] == m, "regime"].iloc[0]
            if observed != expected:
                raise ValueError(f"[CHECK] Regime mapping wrong for {m}: got {observed}, expected {expected}")
        else:
            print(f"[WARN] month_id {m} not present in Table 4 (skipping boundary check).")

    print("\n[CHECK] Table 4 rows by regime:")
    print(table4["regime"].value_counts(dropna=False))

    # --- Existing sanity checks (ESG + market proxy) ---
    esg_cols = ["esg_score", "environment_score", "social_score", "governance_score"]
    if table4[esg_cols].isna().any().any():
        raise ValueError("[CHECK] Unexpected NaNs in ESG columns in Table 4 after merge.")

    if table4["rm_proxy"].isna().any() or table4["rm_excess"].isna().any():
        raise ValueError("[CHECK] Unexpected NaNs in rm_proxy / rm_excess in Table 4.")

    assert_has_columns(market_monthly, ["month_id", "rm_proxy", "rm_excess", "rf_monthly"], "market_monthly")
    if market_monthly[["rm_proxy", "rm_excess"]].isna().any().any():
        raise ValueError("[CHECK] market_monthly has NaNs in rm_proxy/rm_excess (unexpected).")

    out4 = save_table(table4, "table_4.csv")
    print(f"Saved Table 4 to: {out4}")

    outm = save_table(market_monthly, "market_proxy_monthly.csv")
    print(f"Saved market proxy to: {outm}")

    print("\nAll checks passed. Pipeline OK.")

    # ----------------------------------------------------------
    # 7) EDA (Step 3): heatmap + scatter
    # ----------------------------------------------------------
    results_dir = root / "results"
    fig_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    heatmap_path = plot_corr_heatmap_numeric(table4, fig_dir / "corr_heatmap_numeric.png")
    print(f"Saved EDA heatmap to: {heatmap_path}")

    scatter_path = plot_scatter_esg_vs_mean_sharpe(
        table4, fig_dir / "scatter_esg_vs_mean_sharpe_sector.png"
    )
    print(f"Saved EDA scatter to: {scatter_path}")

    # ----------------------------------------------------------
    # 8) Step 4: ESG portfolio sorts (EW) + save tables
    # ----------------------------------------------------------
    out_ret, out_shp = build_esg_sorted_portfolios_ew(
        table4,
        results_tables_dir=tables_dir,
        n_buckets=3,
    )
    print(f"Saved Step 4 EW portfolio returns to: {out_ret}")
    print(f"Saved Step 4 EW portfolio sharpe to:  {out_shp}")

    portfolio_sharpe_ew = pd.read_csv(out_shp)

    fig_dir = root / "results" / "figures"
    tables_dir = root / "results" / "tables"

    # ----------------------------------------------------------
    # 9) Step 5
    # ----------------------------------------------------------
    p1, p2, p3 = create_ew_portfolio_star_outputs(
        portfolio_sharpe_ew=portfolio_sharpe_ew,
        figures_dir=fig_dir,
        tables_dir=tables_dir,
    )

    print(f"Saved EW Sharpe timeseries to: {p1}")
    print(f"Saved EW regime bars to:      {p2}")
    print(f"Saved EW High–Low table to:   {p3}")

    # ----------------------------------------------------------
    # step 6:VW robustness: build portfolios + star outputs + EW vs VW comparison
    # ----------------------------------------------------------
    out_ret_vw, out_shp_vw = build_esg_sorted_portfolios_vw(
        table4,
        results_tables_dir=tables_dir,
        n_buckets=3,
    )
    print(f"Saved VW portfolio returns to: {out_ret_vw}")
    print(f"Saved VW portfolio sharpe to:  {out_shp_vw}")

    portfolio_sharpe_vw = pd.read_csv(out_shp_vw)

    p1_vw, p2_vw, p3_vw = create_vw_portfolio_star_outputs(
        portfolio_sharpe_vw=portfolio_sharpe_vw,
        figures_dir=fig_dir,
        tables_dir=tables_dir,
    )
    print(f"Saved VW Sharpe timeseries to: {p1_vw}")
    print(f"Saved VW regime bars to:      {p2_vw}")
    print(f"Saved VW High–Low table to:   {p3_vw}")

    cmp_path = save_ew_vs_vw_regime_bucket_comparison(
        portfolio_sharpe_ew=portfolio_sharpe_ew,
        portfolio_sharpe_vw=portfolio_sharpe_vw,
        outpath=tables_dir / "ew_vs_vw_mean_sharpe_by_regime_bucket.csv",
    )
    print(f"Saved EW vs VW comparison table to: {cmp_path}")

    # ----------------------------------------------------------
    # Step 7: Regime-interaction regression (base + robustness)
    # ----------------------------------------------------------
    reg_a, reg_b = export_regime_interaction_regressions(
        table4=table4,
        tables_dir=tables_dir,
        esg_col="esg_score",
        cov_type="cluster",   # clustered SE by ticker (panel-friendly)
    )
    print(f"Saved regression (base) to: {reg_a}")
    if reg_b is not None:
        print(f"Saved regression (with index_weight) to: {reg_b}")
    else:
        print("[REG] index_weight model was skipped (unstable or failed).")
    # ----------------------------------------------------------
    # Step 8 — ML block (with vs without ESG)
    # ----------------------------------------------------------
    ml_csv, ml_fig = run_ml_with_without_esg(
        table4=table4,
        tables_dir=tables_dir,
        figures_dir=fig_dir,
    )
    print(f"Saved ML metrics table to: {ml_csv}")
    print(f"Saved ML metrics figure to: {ml_fig}")

    # ----------------------------------------------------------
    # Step 9
    # ----------------------------------------------------------
    robust_csv = run_esg_total_vs_subscores_robustness_ml(
            table4=table4,
            tables_dir=tables_dir,
        )
    print(f"Saved robustness table (ESG total vs E/S/G) to: {robust_csv}")
    
if __name__ == "__main__":
    main()