#==========================================================================================================
#                                       Section: data_loader.py
# In this section:
# -- Sets up a reader for the data stocked in .csv files in data/raw
# -- Cleans the data and stocks the data in different tables
# -- Sets the timeline and regimes of the project
# -- Returns clean datasets ready for use and save them in data/processed
#==========================================================================================================

# Necessary Imports
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

# Sets the root of the proyect in order to find the raw .csv files
root_file = Path(__file__).resolve().parents[1]
directory = root_file / "data" / "raw"


def load_sp500(doc: str): # S&PP500 Dataset

    # Complete path of the CSV file for S&P500
    path = directory / doc

    if not path.exists():
        raise FileNotFoundError(f"Error:{path} was not found in the corresponding file.") # Just in case, raises error if the document is not found

    # Read the CSV doc
    df = pd.read_csv(path, low_memory=False)

    # Transform values into the correct format, check and return NaN if no value
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Erase rows when essential values are missing
    df = df.dropna(subset=["date", "ticker", "weight", "rank"]).copy()

    # Just in case there some typos, ensure that every ticker is spelled the same (ex: "AAPL " -> "AAPL")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    return df # Ready to use S&P500 dataset

def load_tb3ms(doc: str): # Treasury-Bills Dataset

    # Complete path of the CSV file for T-Bills
    path = directory / doc

    if not path.exists():
        raise FileNotFoundError(f"Error: {path} not found.") # Just in case, raises error if the document is not found
    
    # Read the CSV doc
    df = pd.read_csv(path)

    # Change the name of columns, project-focused terms for more clarity
    date = df.columns[0]
    values = df.columns[1]
    df = df.rename(columns={date: "date", values: "rf_annual"}).copy()

    # Transform values into the correct format, check and return NaN if no value
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rf_annual"] = pd.to_numeric(df["rf_annual"], errors="coerce") # Note: rf_annual in %

    # Erase invalid dates
    df = df.dropna(subset=["date"]).copy()

    return df # Return the clean dataset ready to work on

def load_esg(doc: str): # ESG scores Dataset

    # Complete path of the CSV file for T-Bills
    path = directory / doc

    if not path.exists():
        raise FileNotFoundError(f"Error: {path} not found.") # Just in case, raises error if the document is not found

     # Read the CSV doc
    df = pd.read_csv(path, low_memory=False).copy()

    # Assure that the ticker names are in the same format as the ones in other documents
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    return df

# Set the project´s timeline of study.
timeline_start = pd.Timestamp("2018-09-01") # Start date: 1 September 2018 
timeline_end = pd.Timestamp("2024-10-31") # End date: 31 October 2024

#===========================================================================================================
# Regime definition: Normal (2018.09 - 2020.02) / COVID (2020.03 - 2021.12) / Tightening (2022.01 - 20241.0)
#===========================================================================================================

r_normal_end = 202002
r_covid_start = 202003
r_covid_end = 202112
r_tight_start = 202201

def add_regime_columns(df: pd.DataFrame, month_col: str = "month_id") -> pd.DataFrame:

    out = df.copy()
    out[month_col] = pd.to_numeric(out[month_col], errors="coerce").astype("Int64")

    if out[month_col].isna().any():
        raise ValueError("[DEBUGGER] month_id contains NaN after coercion. Check inputs before adding regimes.")

    mid = out[month_col].astype(int)

    out["year"] = (mid // 100).astype(int)
    out["month"] = (mid % 100).astype(int)

    out["regime"] = np.select(
        [
            mid <= r_normal_end,
            (mid >= r_covid_start) & (mid <= r_covid_end),
            mid >= r_tight_start,
        ],
        ["normal", "covid", "tightening"],
        default="unknown",
    )

    if (out["regime"] == "unknown").any():
        bad = out.loc[out["regime"] == "unknown", month_col].drop_duplicates().tolist()
        raise ValueError(f"[DEBUGGER] Some month_id values fall outside regime mapping: {bad[:10]}")

    return out

#==========================================================================================================
# Creation of Table 0: Top 100 weighted companies of S&P500 from 2018 to 2024 (monthly)
#==========================================================================================================

def build_table_0(df_raw: pd.DataFrame, n: int = 100):
    df = df_raw.copy()

    # Shrink to desired timelines
    df = df[(df["date"] >= timeline_start) & (df["date"] <= timeline_end)].copy()

    # Creation of month_id, a column that will help at merging all the tables since there will be many different variables (the key)
    df["month_id"] = df["date"].dt.year * 100 + df["date"].dt.month # 2018-9-1 --> 201809

    # Choose just one date since month_id doesn´t keep track of the day
    last_date = df.groupby("month_id")["date"].transform("max")
    df = df[df["date"] == last_date].copy()

    # We keep the number of companies we want: 100
    df = df[df["rank"] <= n].copy()

    # Proper definition of the Table 0
    table0 = pd.DataFrame({
        "date": df["date"].dt.date,
        "month_id": df["month_id"].astype(int),
        "rank_weight": df["rank"].astype(int),
        "company": df["name"].astype(str).str.strip(),
        "ticker": df["ticker"],
        "sector": df["sector"].astype(str).str.strip(),
        "index_weight": df["weight"].astype(float),
    })

    return table0.reset_index(drop = True) # Return Table 0 as a dataset with a new and proper index (the old one was all mixed)

#==========================================================================================================
# Creation of Table 1: Risk-free rate across our studied dates
#==========================================================================================================

def build_table_1(table0: pd.DataFrame, rf_h: pd.DataFrame, n: int = 12):

    # Need to work on a separate dataset so the original data remains untouched
    rf = rf_h.copy()

    # Create month_id key
    rf["month_id"] = rf["date"].dt.year * 100 + rf["date"].dt.month
    rf["month_id"] = rf["month_id"].astype(int)

    # In case some months are missing, assume the last known annual T-bill rate.
    rf = rf.sort_values("month_id").copy() # Force a chronological order
    rf["rf_annual"] = rf["rf_annual"].ffill()

    # Little check to see if everything was done correctly
    if rf["rf_annual"].isna().any():
        raise ValueError("Risk-free series still has missing values after forward fill.")
    
    #------------------- Adjusts needed for Table 2 -------------------------
    panel_months = sorted(table0["month_id"].unique().tolist()) # Take the months we need
    first_panel = pd.to_datetime(str(panel_months[0]) + "01", format="%Y%m%d")
    last_panel  = pd.to_datetime(str(panel_months[-1]) + "01", format="%Y%m%d")

    # For the sharpe calculations, we need at least the values of the previous 12 months and just in case, the month after the timeline studied
    start_need = first_panel - pd.DateOffset(months = n + 1)
    end_need = last_panel + pd.DateOffset(months = 1)
    need_months = pd.period_range(start = start_need, end = end_need, freq = "M")
    need_month_id = [p.year * 100 + p.month for p in need_months]

    # Keep only the month_id needed for Table 2
    rf = rf[rf["month_id"].isin(need_month_id)].copy()

    # Pass from annualization to monthly return (returns in Table 2 are monthly returns)
    rf["rf_monthly"] = (1 + rf["rf_annual"] / 100.0) ** (1 / 12) - 1
    # ------------------------------------------------------------------------

    # Proper definition of the Table 1
    table1 = rf[["month_id", "rf_annual", "rf_monthly"]].copy()

    return table1.reset_index(drop = True)

# =========================================================================================================
# Creation of Table 2:  Calculate and gather the Sharpe-values of each studied company
# =========================================================================================================
   
def build_table_2(table0: pd.DataFrame, table1: pd.DataFrame, window: int = 12):

    # Creation of the unique company's tickers list
    tickers_search = sorted(table0["ticker"].unique().tolist())

    # Timeline needed for Sharpe calculations
    start_date = pd.to_datetime(table0["date"]).min() - pd.DateOffset(months = 13)
    end_date = pd.to_datetime(table0["date"]).max() + pd.DateOffset(days = 10)

    # Some tickers are named differently in yahoo or have changed during the years
    yahoo_tickers = {
        "FB": "META",      # Facebook -> Meta
        "ANTM": "ELV",     # Anthem -> Elevance
        "BRKB": "BRK-B",   # Berkshire Hathaway Class B (Yahoo format)
        "BFB": "BF-B",     # Brown-Forman Class B (Yahoo format) - if it appears
    }

    # Map original tickers -> Yahoo tickers for download
    original_to_yahoo = {t: yahoo_tickers.get(t, t) for t in tickers_search}
    yahoo_tickers = sorted(set(original_to_yahoo.values())) # The download list must be unique

    # Download market data (Contains Open/High/Low/Close)
    full_data = yf.download(
        tickers = yahoo_tickers,
        start = start_date.strftime("%Y-%m-%d"),
        end = end_date.strftime("%Y-%m-%d"),
        auto_adjust = True, progress = False, group_by = "ticker", threads = True,)

    # Check whether we received multiple prices or just one. "Close" price needed
    if isinstance(full_data.columns, pd.MultiIndex):
        prices_yahoo = full_data.xs("Close", axis = 1, level = 1)
    else:
        prices_yahoo = full_data[["Close"]].rename(columns = {"Close": yahoo_tickers[0]})

    # Table 2 needs to use same tickers as Table 0 --> Rebuild close_prices so we can merge table2 later on
    series_dict = {
        orig: (
            prices_yahoo[y].copy()
            if y in prices_yahoo.columns
            else pd.Series(index = prices_yahoo.index, dtype = "float64")
        )
        for orig, y in original_to_yahoo.items()
    }
    close_prices = pd.DataFrame(series_dict, index = prices_yahoo.index) # Series_dict --> DataFrame: index = dates, columns = original tickers, values = close prices

    # Cleaning the new DataFrame
    close_prices = close_prices.dropna(how = "all")
    close_prices = close_prices.resample("ME").last()
    returns_month = close_prices.pct_change(fill_method = None) # Just to avoid annoying FutureWarning message

    # Creation of the month_id keys for a succesful merging
    returns_month.index = returns_month.index.year * 100 + returns_month.index.month
    returns_month.index.name = "month_id"

    # Panel months from Table 0
    panel_months = sorted(table0["month_id"].unique().tolist())

    # ---------------------- Sharpe Ratio Calculation -------------------------
    # This is the same logic as stated in the definition of function build_table_1
    first_panel = pd.to_datetime(str(panel_months[0]) + "01", format = "%Y%m%d")
    last_panel  = pd.to_datetime(str(panel_months[-1]) + "01", format = "%Y%m%d")
    start_desired = first_panel - pd.DateOffset(months = window + 1)
    end_desired   = last_panel + pd.DateOffset(months = 1)

    # 3) Build the list of months we want, then convert them back to month_id integers (YYYYMM)
    needed_months = pd.period_range(start = start_desired, end = end_desired, freq = "M")
    needed_month_ids = [p.year * 100 + p.month for p in needed_months]

    # 4) Reindex returns to this expanded set of months (missing months become NaN)
    returns_month = returns_month.reindex(needed_month_ids)

    # Swap position of columns-rows
    r = returns_month.stack().reset_index()
    r.columns = ["month_id", "ticker", "monthly_return"]
    r = r.sort_values(["ticker", "month_id"]).copy()

    # Merge Table 1 into Table 2 using the month_id
    r = r.merge(table1[["month_id", "rf_monthly"]], on = "month_id", how = "left")

    # Debugger: if rf missing in some months --> Sharpe becomes NaN
    if r["rf_monthly"].isna().any():
        missing_months = r.loc[r["rf_monthly"].isna(), "month_id"].drop_duplicates().tolist()
        raise ValueError(f"[DEBUGGER] There are missing rf_monthly for months: {missing_months}")

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
    #----------------------------------------------------------------------------
    
    r = r[r["month_id"].isin(panel_months)].copy()
    r = r.dropna(subset=["sharpe_12m"]).copy()

    # Proper definition of the Table 2
    table2 = r[[
        "month_id", "ticker",
        "rf_monthly", "monthly_return", "excess_return",
        "avg_excess_return_12m", "vol_excess_return_12m", "sharpe_12m"
    ]].reset_index(drop = True)

    return table2

#==========================================================================================================
# Creation of Table 3: Values of ESG and sub-scores E/S/G 
#==========================================================================================================

def build_table_3(table2: pd.DataFrame, esg_raw: pd.DataFrame):

    # We need the list of succesful tickers in Table 2
    tickers_panel = sorted(table2["ticker"].astype(str).str.strip().str.upper().unique().tolist())

    # Take the ESG document that we read
    esg = esg_raw.copy()
    esg["Ticker"] = esg["Ticker"].astype(str).str.strip().str.upper() # Ensure the correct format

    # Keep only successful rows
    esg["ESG Status"] = esg["ESG Status"].astype(str).str.strip().str.lower()
    esg = esg[esg["ESG Status"] == "success"].copy()

    # Shrink the dataset to keep just what we want
    columns_desired = ["Ticker", "ESG Score", "Governance Score", "Environment Score", "Social Score"]
    esg = esg[columns_desired].copy()

    # Ensure numeric stability in the dataset
    for column in ["ESG Score", "Governance Score", "Environment Score", "Social Score"]:
        esg[column] = pd.to_numeric(esg[column], errors = "coerce")

    # Again, some tickers are named differently
    tickers_fix = {
        "FB": "META",       # Facebook -> Meta
        "ANTM": "ELV",      # Anthem -> Elevance
        "BRKB": "BRK-B",    # Berkshire
        "AET": "CVS",       # Aetna acquired by CVS 
        "GOOG": "GOOGL",    # Google
    }

    # Build Table 3 by looking directly in the dataset for each ticker
    rows = []
    missing = []

    # Make ESG quickly searchable by putting tickers as index
    esg_indexed = esg.set_index("Ticker", drop = False)

    for ticker in tickers_panel:
        ticker_esg = tickers_fix.get(ticker, ticker) # To apply the just defined dictionnary
        if ticker_esg in esg_indexed.index:
            row = esg_indexed.loc[ticker_esg] # returns the ESG row for that ticker.
            # Debugger: careful of duplicate of tickers (normally not necessary)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            rows.append({
                "ticker": ticker,
                "esg_score": row["ESG Score"],
                "governance_score": row["Governance Score"],
                "environment_score": row["Environment Score"],
                "social_score": row["Social Score"],
            })
        else:
            missing.append(ticker) # Just for debugging purposes

    # For visually appealing, sort it by alphabet
    table3 = pd.DataFrame(rows).sort_values("ticker").reset_index(drop = True)

    #---------------------- Optional checkers -------------------------
    # print("\n[DEBUGGER] ESG overlap with Table 2:")
    # print(f" -> total tickers in Table 2: {len(tickers_panel)}")
    # print(f" -> total tickers in Table 3: {table3['ticker'].nunique()}")
    # print(f" -> overlap: {table3['ticker'].nunique()}")
    # if missing: # Gives the list of all the tickers where data is missing
        #print(f"[DEBUGGER] List of the {len(missing)} missing tickers: {missing[:20]}")
    #-------------------------------------------------------------------

    return table3

#==========================================================================================================
# Creation of Final Table: Fusion of all tables and creation of market_proxy for economic shocks definition
#==========================================================================================================

def build_table_4(table0: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Work on an independent dataset copy
    t0 = table0[["month_id", "ticker","index_weight", "sector", "company", "rank_weight"]].copy()

    # Merging Table 0 with Table 2 using only month_id and ticker
    base = table2.merge(t0, on = ["month_id", "ticker"], how = "inner", validate = "one_to_one")

    #------------------- Calculation of monthly market proxy --------------------------
    tmp = base[["month_id", "index_weight", "monthly_return", "rf_monthly"]].copy() # Take the info from our base table 4
    tmp["wret"] = tmp["index_weight"] * tmp["monthly_return"] # Stock’s weighted return contribution
    # Aggregation by month
    agg = tmp.groupby("month_id", as_index=False).agg(sum_wret = ("wret", "sum"), sum_w = ("index_weight", "sum"), rf_monthly = ("rf_monthly", "first"))
    agg["rm_proxy"] = agg["sum_wret"] / agg["sum_w"] # Market proxy return (value-weighted)
    agg["rm_excess"] = agg["rm_proxy"] - agg["rf_monthly"] # Market excess return
    #-----------------------------------------------------------------------------------

    # Create a clean monthly market table
    market_monthly = agg[["month_id", "rm_proxy", "rm_excess", "rf_monthly", "sum_w"]].copy()

    # Merge the base of our Table 4 with Table 3 using ticker
    base_esg = base.merge(table3, on = "ticker", how = "inner", validate = "many_to_one")

    # Add the newly calculated market proxy to every row
    table4 = base_esg.merge(market_monthly[["month_id", "rm_proxy", "rm_excess"]], on = "month_id", how = "left", validate = "many_to_one")

    # Debugger: ensure that market proxy exists in every row
    if table4["rm_proxy"].isna().any() or table4["rm_excess"].isna().any():
             # Find the rows where rm_proxy is NaN. Then, take the month_id column (ensure there are no duplicates) and make a list
            missing_months = table4.loc[table4["rm_proxy"].isna(), "month_id"].drop_duplicates().tolist()
            raise ValueError(f"[DEBUGGER] Missing rm_proxy or rm_excess for following dates: {missing_months[:20]}")

    # Add the regimes definition
    table4 = add_regime_columns(table4, month_col="month_id")
    market_monthly = add_regime_columns(market_monthly, month_col="month_id")

    return table4.reset_index(drop=True), market_monthly.reset_index(drop=True) # Return also market_monthly for possible debugging/comparison in the future

#==========================================================================================================
# Creation of the function needed to save new tables in the project file
#==========================================================================================================

def save_table(table: pd.DataFrame, file_name: str) -> Path:
    
    # First, where we are going to save it
    directory = root_file / "data" / "processed" # Write the output directory path
    directory.mkdir(parents=True, exist_ok=True) # Or it creates the directory if it does not exist
    path = directory / file_name # Set where the file will be finally located

    # Now, save the table as a CSV file
    table.to_csv(path, index=False)

    return path