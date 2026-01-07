#===================================================================================================
# This document reads the data stocked in the file data/raw and returns clean datasets ready for use
#===================================================================================================

# Necessary Imports
from pathlib import Path
import pandas as pd

# Sets the root of the proyect in order to find the raw csv files
root_file = Path(__file__).resolve().parents[1]
directory_1 = root_file / "data" / "raw"


def load_sp500(document: str = "sp500_historical.csv") -> pd.DataFrame:
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

    return df # Ready to use S&P500 dataset for pipeline

#===================================================================================================
# Creation of Table 0 --> Top 100 weighted companies of S&P500 from 2018 to 2024 (monthly)
#===================================================================================================

timeline_start = pd.Timestamp("2018-09-01") # Creation of a timestamp for the start date
timeline_end = pd.Timestamp("2024-10-31") # Creation of a timestamp for the end date

def build_table_0(df_raw: pd.DataFrame, n: int = 100) -> pd.DataFrame:

    df = df_raw.copy() # Work on a copy to avoid working on the original df

    # Shrink the dataset to the desired timestamps
    df = df[(df["date"] >= timeline_start) & (df["date"] <= timeline_end)].copy() # .copy() so we ensure that new df is independant.

    # Select only the top n companies in the list
    df = df[df["rank"] <= n].copy()

    # Gives a tag to every date, solution to past merging problems
    df["month_id"] = df["date"].dt.year * 100 + df["date"].dt.month

    # Placeholder for a later problem of companies changing their ticker (useful later on)
    df["firm_id"] = df["ticker"] #firm_id will be our personal identifier for each company 

    # Creation of the Table 0
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

    # Soft validation: expect 100 rows per month
    #counts = table0.groupby("yyyymm").size()
    # For each month, count the number of rows we have.
    #bad = counts[counts != n]
    # Identify months where the count is not exactly top_n (usually should be 100).
    #if not bad.empty:
        # If there are “weird” months, we warn but do not stop execution.
        #print(f"[WARN] Months with != {n} rows: {bad.to_dict()}")

    return table0.reset_index(drop=True) # Return Table 0 with a new and proper index

#===================================================================================================
# Definition of the function to save new tables in the project file
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