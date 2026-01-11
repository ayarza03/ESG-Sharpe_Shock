# ==========================================================================================
# main.py
# Runs the pipeline:
# 1) Load raw datasets
# 2) Build Table 0 (universe)
# 3) Build Table 1 (risk-free)
# 4) Build Table 2 (returns + rolling Sharpe)
# 5) Build Table 3 (ESG snapshot restricted to Table 2 + success)
# 6) Build Table 4 (final merged panel + market proxy)
# 7) Save outputs to data/processed/
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

    nan_rate = table2["sharpe_12m"].isna().mean()
    print(f"[CHECK] % NaN sharpe_12m (panel only): {nan_rate:.3%}")

    nan_tickers = (
        table2.loc[table2["sharpe_12m"].isna(), "ticker"]
        .value_counts()
        .head(15)
    )
    print("[CHECK] Top tickers with NaN sharpe_12m counts:\n", nan_tickers)

    print("\nTable 2 built:")
    print(f" - shape: {table2.shape}")
    print(" - sample:")
    print(table2.head(5))

    months = table0["month_id"].nunique()
    unique_tickers = table0["ticker"].nunique()
    expected_max = unique_tickers * months

    print(f" - unique tickers in Table 0 (across all months): {unique_tickers}")
    print(f" - Table 2 rows (max possible): {expected_max}")
    print(f" - Table 2 rows (actual):       {len(table2)}")

    out2 = save_table(table2, "table_2.csv")
    print(f"Saved Table 2 to: {out2}")

    # ----------------------------------------------------------
    # 5) Build Table 3 (ESG snapshot restricted to Table 2)
    # ----------------------------------------------------------
    # Your build_table_3 signature is: build_table_3(table2: pd.DataFrame, esg_raw: pd.DataFrame)
    table3 = build_table_3(table2=table2, esg_raw=esg_raw)

    print("\nTable 3 built (ESG snapshot restricted to Table 2):")
    print(f" - shape: {table3.shape}")
    print(" - sample:")
    print(table3.head(5))

    out3 = save_table(table3, "table_3.csv")
    print(f"Saved Table 3 to: {out3}")

    # ----------------------------------------------------------
    # 6) Build Table 4 (final merged panel + market proxy)
    # ----------------------------------------------------------
    table4, market_monthly = build_table_4(table0, table2, table3)

    print("\nTable 4 built (final panel):")
    print(f" - shape: {table4.shape}")
    print(" - sample:")
    print(table4.head(5))

    # Sanity: no NaNs in ESG columns in Table 4
    esg_cols = ["esg_score", "environment_score", "social_score", "governance_score"]
    if table4[esg_cols].isna().any().any():
        raise ValueError("Unexpected NaNs in ESG columns in Table 4 after drop.")

    # Sanity: market proxy exists for all rows
    if table4["rm_proxy"].isna().any() or table4["rm_excess"].isna().any():
        raise ValueError("Unexpected NaNs in rm_proxy / rm_excess in Table 4.")

    out4 = save_table(table4, "table_4.csv")
    print(f"Saved Table 4 to: {out4}")

    outm = save_table(market_monthly, "market_proxy_monthly.csv")
    print(f"Saved market proxy to: {outm}")


if __name__ == "__main__":
    main()