#==========================================================================================================
# This document reads the data stocked in the file data/raw and returns clean datasets ready for use
#==========================================================================================================

# Necessary Imports
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

# Sets the root of the proyect in order to find the raw csv files
root_file = Path(__file__).resolve().parents[1]
directory_1 = root_file / "data" / "raw"


def load_sp500(document: str = "sp500_historical.csv") -> pd.DataFrame: # S&PP500 Dataset

    """
    This function aims to read sp500_historical and return a DataFrame.
    1. Ensures the good types for each data
    2. Erases null rows of the raw dataset
    Columns desired are: date, ticker, weight, rank.
    """
    # Complete path of the CSV file for S&P500
    path = directory_1 / document

    if not path.exists():
        raise FileNotFoundError(f"Error:{path} was not found in the corresponding file.") # Raises error if the document is not found

    # Read the CSV File
    df = pd.read_csv(path, low_memory=False)

    # Transform values from column "date" in datetime format, returns NaT if invalid.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Transform values from column "rank" in numeric format, returns NaN if no value.
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    # Transform values from column "weight" in numeric format, returns NaN if no value.
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Drop rows when essential values are missing
    df = df.dropna(subset=["date", "ticker", "weight", "rank"]).copy()

    # Just in case there some typos, ensure that every ticker is spelled the same (ex: "AAPL " -> "AAPL")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    return df # Ready to use S&P500 dataset

def load_tb3ms(document: str = "TB3MS.csv") -> pd.DataFrame: # Treasury-Bills Dataset
    """
    This function aims to read the Treasury-Bills csv from FED
    """
    # Complete path of the CSV file for T-Bills
    path = directory_1 / document
    if not path.exists():
        raise FileNotFoundError(f"Error: {path} not found.")
    # Read the CSV File
    df = pd.read_csv(path)

    # Assign a name to each column
    date = df.columns[0]
    values = df.columns[1]
    df = df.rename(columns={date: "date", values: "rf_annual"}).copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rf_annual"] = pd.to_numeric(df["rf_annual"], errors="coerce") # rf_annual express in percentage

    # Drop invalid dates; keep NaNs in rf_annual (we'll forward fill)
    df = df.dropna(subset=["date"]).copy()

    return df # Return the clean dataset ready to work on

timeline_start = pd.Timestamp("2018-09-01") # Creation of a timestamp for the start date
timeline_end = pd.Timestamp("2024-10-31") # Creation of a timestamp for the end date

#==========================================================================================================
# Definition of Table 0: Top 100 weighted companies of S&P500 from 2018 to 2024 (monthly)
#==========================================================================================================

def build_table_0(df_raw: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    df = df_raw.copy()

    # Shrink to desired timestamps
    df = df[(df["date"] >= timeline_start) & (df["date"] <= timeline_end)].copy()

    # Month_id first (monthly priority)
    df["month_id"] = df["date"].dt.year * 100 + df["date"].dt.month

    # Choose ONE snapshot date per month (last available date)
    last_date = df.groupby("month_id")["date"].transform("max")
    df = df[df["date"] == last_date].copy()

    # Now select only the top n companies (for that monthly snapshot)
    df = df[df["rank"] <= n].copy()

    # Placeholder for later ticker-change handling
    df["firm_id"] = df["ticker"]

    table0 = pd.DataFrame({
        "date": df["date"].dt.date,
        "month_id": df["month_id"].astype(int),
        "rank_weight": df["rank"].astype(int),
        "firm_id": df["firm_id"],
        "ticker": df["ticker"],
        "company": df["name"].astype(str).str.strip(),
        "sector": df["sector"].astype(str).str.strip(),
        "index_weight": df["weight"].astype(float),
    })

    return table0.reset_index(drop=True) # Return Table 0 with a new and proper index

#==========================================================================================================
# Definition of Table 1: Risk-free rate 
#==========================================================================================================

def build_table_1(table0: pd.DataFrame, rf_historic: pd.DataFrame) -> pd.DataFrame:
    rf = rf_historic.copy()
    
    # Create month_id key (cannot copy directly from Table 0 since rows are different)
    rf["month_id"] = rf["date"].dt.year * 100 + rf["date"].dt.month
    rf["month_id"] = rf["month_id"].astype(int)

    # Keep only the month_id that are in Table 0
    panel_months = sorted(table0["month_id"].unique().tolist())
    rf = rf[rf["month_id"].isin(panel_months)].copy()
    
    rf = rf.sort_values("month_id").copy()
    rf["rf_annual"] = rf["rf_annual"].ffill()
    if rf["rf_annual"].isna().any():
        missing_months = rf.loc[rf["rf_annual"].isna(), "month_id"].unique().tolist()
        raise ValueError(f"Risk-free series has missing values in panel months: {missing_months}")
    
    # Pass from annualization to monthly return
    rf["rf_monthly"] = (1 + rf["rf_annual"] / 100.0) ** (1/12) - 1

    table1 = rf[["month_id", "rf_annual", "rf_monthly"]].copy()

    return table1.reset_index(drop=True) # Return Table 1 with a new and proper index

# =========================================================================================================
# Definition of Table 2:  Calculate and gather the Sharpe-values of each studied company
# =========================================================================================================

def build_price_coverage_report(ret_m: pd.DataFrame, panel_months: list[int]) -> pd.DataFrame:
    """
    Builds a coverage report of monthly returns availability by ticker over the panel.

    Parameters
    ----------
    ret_m : pd.DataFrame
        Monthly returns in WIDE format:
        - index = month_id (YYYYMM)
        - columns = tickers
        - values = monthly returns (float)
    panel_months : list[int]
        List of month_id values you consider part of your analysis panel.

    Returns
    -------
    pd.DataFrame
        One row per ticker with:
        - months_in_panel
        - months_available
        - pct_available
        - first_month_available
        - last_month_available
        - missing_from_end_month (if data ends early)
        - start_gap (True if missing at start)
        - interior_missing (True if missing inside the available range)
    """

    # Ensure panel months are sorted (chronological order)
    panel_months = sorted(panel_months)

    # Align returns to the panel months only (some downloaded months might be outside)
    aligned = ret_m.reindex(panel_months)

    # Availability mask (True = have return, False = missing)
    avail = aligned.notna()

    # Count how many months exist in the panel
    months_in_panel = len(panel_months)

    # For each ticker, count how many months have a non-missing return
    months_available = avail.sum(axis=0)

    # Percentage coverage for each ticker
    pct_available = months_available / months_in_panel

    # Prepare dictionaries to store first and last available month per ticker
    first_month = {}
    last_month = {}

    # Track if data stops before the last panel month (when does missing-at-end begin?)
    missing_from_end = {}

    # Loop through each ticker column and compute first/last availability
    for t in aligned.columns:
        # ok_months is the list of months where return exists for ticker t
        ok_months = aligned.index[avail[t]].tolist()

        if len(ok_months) == 0:
            # No data at all in the panel window
            first_month[t] = None
            last_month[t] = None
            missing_from_end[t] = panel_months[0]  # missing from the start
        else:
            # Data exists; take first and last month with data
            first_month[t] = ok_months[0]
            last_month[t] = ok_months[-1]

            # If last available month is not the last panel month,
            # then the ticker is missing at the end of the sample
            if ok_months[-1] != panel_months[-1]:
                idx = panel_months.index(ok_months[-1])
                missing_from_end[t] = panel_months[idx + 1]  # first missing month after last observed
            else:
                missing_from_end[t] = None

    # start_gap: True if data starts after the first panel month (or no data)
    start_gap = {}
    for t in aligned.columns:
        fm = first_month[t]
        start_gap[t] = (fm is None) or (fm != panel_months[0])

    # interior_missing: True if there is a missing value between first and last available months
    interior_missing = {}
    for t in aligned.columns:
        fm = first_month[t]
        lm = last_month[t]
        if fm is None or lm is None:
            # If no data, we do not label interior missing as True (it’s a different issue)
            interior_missing[t] = False
        else:
            # Slice only between first and last available month and see if any NaN exists
            sub = aligned.loc[fm:lm, t]
            interior_missing[t] = sub.isna().any()

    # Build the final report DataFrame
    report = pd.DataFrame({
        "ticker": aligned.columns,
        "months_in_panel": months_in_panel,
        "months_available": months_available.values,
        "pct_available": pct_available.values,
        "first_month_available": [first_month[t] for t in aligned.columns],
        "last_month_available": [last_month[t] for t in aligned.columns],
        "missing_from_end_month": [missing_from_end[t] for t in aligned.columns],
        "start_gap": [start_gap[t] for t in aligned.columns],
        "interior_missing": [interior_missing[t] for t in aligned.columns],
    })

    # Sort with worst coverage first (lowest pct_available)
    report = report.sort_values(["pct_available", "ticker"], ascending=[True, True]).reset_index(drop=True)

    return report

def build_table_2(table0: pd.DataFrame, table1: pd.DataFrame, window: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Creation of the unique company's tickers list
    tickers_search = sorted(table0["ticker"].unique().tolist())

    # Timestamp needed for Sharpe calculations
    start_date = pd.to_datetime(table0["date"]).min() - pd.DateOffset(months=13)
    end_date = pd.to_datetime(table0["date"]).max() + pd.DateOffset(days=10)

    # Download market data (Contains Open/High/Low/Close)
    full_data = yf.download(
        tickers=tickers_search,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(full_data.columns, pd.MultiIndex):
        close_prices = full_data.xs("Close", axis=1, level=1)
    else:
        close_prices = full_data[["Close"]].rename(columns={"Close": tickers_search[0]})

    # Clean the new data that we have taken
    close_prices = close_prices.dropna(how="all")
    close_prices = close_prices.resample("ME").last()

    # FIX: avoid FutureWarning + be explicit about NA handling
    returns_month = close_prices.pct_change(fill_method=None)

    # Creation of the month_id keys (same as in table 0 and table 1)
    returns_month.index = returns_month.index.year * 100 + returns_month.index.month
    returns_month.index.name = "month_id"

    # Panel months from Table 0
    panel_months = sorted(table0["month_id"].unique().tolist())

    # FIX: keep ONLY panel months before merge (removes 201709-201808 and 202411)
    returns_month = returns_month.reindex(panel_months)

    # Coverage report now correctly computed on the panel only
    coverage_report = build_price_coverage_report(returns_month, panel_months)

    # Swap position of columns-rows
    r = returns_month.stack().reset_index()
    r.columns = ["month_id", "ticker", "monthly_return"]
    r = r.sort_values(["ticker", "month_id"]).copy()

    # Merge Table 1 into Table 2 using the month_id
    r = r.merge(table1[["month_id", "rf_monthly"]], on="month_id", how="left")

    # Safety check: if rf missing in some months, Sharpe becomes NaN
    if r["rf_monthly"].isna().any():
        missing_months = r.loc[r["rf_monthly"].isna(), "month_id"].drop_duplicates().tolist()
        raise ValueError(f"Missing rf_monthly for month_id(s): {missing_months}")

    # Excess returns calculations
    r["excess_return"] = r["monthly_return"] - r["rf_monthly"]

    # Calculate mean and volatility on excess returns
    r["avg_excess_return_12m"] = r.groupby("ticker")["excess_return"].transform(
        lambda s: s.rolling(window).mean()
    )
    r["vol_excess_return_12m"] = r.groupby("ticker")["excess_return"].transform(
        lambda s: s.rolling(window).std(ddof=1)
    )

    # Calculate annualized Sharpe
    r["sharpe_12m"] = np.sqrt(12) * (r["avg_excess_return_12m"] / r["vol_excess_return_12m"])

    # Present the results in a final clean table
    table2 = r[[
        "month_id", "ticker",
        "rf_monthly", "monthly_return", "excess_return",
        "avg_excess_return_12m", "vol_excess_return_12m", "sharpe_12m"
    ]].reset_index(drop=True)

    return table2, coverage_report

#==========================================================================================================
# Definition of Table 3: Companies consistent metadata provider. Bridge between future Table 1 and Table 2 
#==========================================================================================================

#===================================================================================================
# Creation of the function needed to save new tables in the project file
#===================================================================================================

def save_table(table: pd.DataFrame, file_name: str) -> Path:
    """
    Saves a table into data/processed/ as CSV.
    """
    directory_2 = root_file / "data" / "processed" # Write the output directory path
    directory_2.mkdir(parents=True, exist_ok=True) # Or it creates the directory if it does not exist

    path = directory_2 / file_name # Set where the file will be finally located

    table.to_csv(path, index=False) # Saves the table as a CSV file

    return path # So main.py can open the documents