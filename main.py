# main.py
# Entry point: load raw S&P500 file -> build Table 0 -> save it in data/processed/

from src.data_loader import load_sp500, build_table_0, save_table


def main() -> None:
    # 1) Load raw dataset from data/raw/
    df_raw = load_sp500()  # uses default "sp500_historical.csv"

    # 2) Build Table 0 (Top 100 per month in Sep2018–Oct2024)
    table0 = build_table_0(df_raw, n=100)

    # 3) Save Table 0 to data/processed/
    out_path = save_table(table0, "table_0.csv")

    # 4) Quick sanity prints
    print("Saved Table 0 to:", out_path)
    print("Table 0 shape:", table0.shape)
    print(table0.head())


if __name__ == "__main__":
    main()