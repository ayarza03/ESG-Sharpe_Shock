# ==========================================================================================
# main.py
# Runs the pipeline:
# 1) Load raw datasets
# 2) Build Table 0 (universe)
# 3) Build Table 1 (risk-free)
# 4) Build Table 2 (returns + rolling Sharpe)
# 5) Save outputs to data/processed/
# ==========================================================================================

from pathlib import Path

import pandas as pd

from src.data_loader import (
    load_sp500,
    load_tb3ms,
    build_table_0,
    build_table_1,
    build_table_2,
    save_table,
)


def main() -> None:
    # ----------------------------------------------------------
    # 0) Paths (optional prints to confirm you run from repo root)
    # ----------------------------------------------------------
    root = Path(__file__).resolve().parent
    print(f"Running from: {root}")

    # ----------------------------------------------------------
    # 1) Load raw data
    # ----------------------------------------------------------
    sp500_raw = load_sp500("sp500_historical.csv")
    rf_raw = load_tb3ms("TB3MS.csv")

    print("\nLoaded raw datasets:")
    print(f" - sp500_raw shape: {sp500_raw.shape}")
    print(f" - rf_raw shape:    {rf_raw.shape}")

    # ----------------------------------------------------------
    # 2) Build Table 0
    # ----------------------------------------------------------
    table0 = build_table_0(sp500_raw, n=100)

    print("\nTable 0 built:")
    print(f" - shape: {table0.shape}")
    print(f" - months: {table0['month_id'].nunique()}")
    print(" - sample:")
    print(table0.head(5))

    # Quick sanity check: 100 rows per month expected
    rows_per_month = table0.groupby("month_id").size()
    if (rows_per_month != 100).any():
        bad = rows_per_month[rows_per_month != 100].to_dict()
        print(f"[WARN] Table 0 months not equal to 100 rows: {bad}")
    else:
        print(" - OK: Table 0 has exactly 100 rows per month.")

    # Save Table 0
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

    # Sanity checks: months align with Table 0
    missing_rf_months = sorted(set(table0["month_id"]) - set(table1["month_id"]))
    if missing_rf_months:
        print(f"[WARN] Table 1 is missing rf for month_id(s): {missing_rf_months[:10]} ...")
    else:
        print(" - OK: Table 1 covers all Table 0 months.")

    out1 = save_table(table1, "table_1.csv")
    print(f"Saved Table 1 to: {out1}")

    # ----------------------------------------------------------
    # 4) Build Table 2 (Sharpe + coverage report)
    # ----------------------------------------------------------
    table2, coverage = build_table_2(table0, table1, window=12)

    print("\nTable 2 built:")
    print(f" - shape: {table2.shape}")
    print(" - sample:")
    print(table2.head(5))

    # Sanity check: should have 100 tickers * number_of_months rows (minus rolling NaNs early on)
    months = table0["month_id"].nunique()
    unique_tickers = table0["ticker"].nunique()
    expected_max = unique_tickers * months

    print(f" - unique tickers in Table 0 (across all months): {unique_tickers}")
    print(f" - Table 2 rows (max possible): {expected_max}")
    print(f" - Table 2 rows (actual):       {len(table2)}")
    out2 = save_table(table2, "table_2.csv")
    print(f"Saved Table 2 to: {out2}")

    out_cov = save_table(coverage, "coverage_report.csv")
    print(f"Saved coverage report to: {out_cov}")

    # Print “worst coverage” tickers (useful to detect Yahoo ticker issues)
    print("\nWorst 15 tickers by coverage:")
    print(coverage.head(15))


if __name__ == "__main__":
    main()