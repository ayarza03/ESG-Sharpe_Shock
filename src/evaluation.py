#==========================================================================================================
#                                       Section: evaluation.py
# In this section:
#==========================================================================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.models import RegressionSpec, TemporalSplit, fit_ols, fit_predict_ml_suite
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

#===========================================================================================================
# step 3
#===========================================================================================================

def plot_corr_heatmap_numeric(df: pd.DataFrame, outpath: Path) -> Path:
    """
    Correlation heatmap for all numeric variables in df (whole period).
    Saves a PNG to outpath.
    """
    _ensure_dir(outpath.parent)

    numeric = df.select_dtypes(include=[np.number]).copy()

    # Drop columns that are all-NA or constant (corr not defined / not useful)
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

    if numeric.shape[1] < 2:
        raise ValueError("[EDA] Not enough numeric columns to compute correlation heatmap.")

    corr = numeric.corr().fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_title("Correlation heatmap (numeric variables)")
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    return outpath

def plot_scatter_esg_vs_mean_sharpe(df: pd.DataFrame, outpath: Path) -> Path:
    """
    Scatter:
      - each point = ticker
      - x = esg_score
      - y = mean(sharpe_12m) over entire period
      - color = sector
    Saves a PNG to outpath.
    """
    _ensure_dir(outpath.parent)

    required = {"ticker", "esg_score", "sharpe_12m", "sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[EDA] Missing required columns for scatter: {sorted(missing)}")

    by_ticker = (
        df.groupby("ticker", as_index=False)
        .agg(
            esg_score=("esg_score", "first"),
            mean_sharpe=("sharpe_12m", "mean"),
            sector=("sector", "first"),
        )
        .dropna(subset=["esg_score", "mean_sharpe", "sector"])
        .copy()
    )

    # Sector -> color mapping (stable ordering)
    sectors = sorted(by_ticker["sector"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    sector_to_color = {s: cmap(i % 20) for i, s in enumerate(sectors)}
    colors = by_ticker["sector"].map(sector_to_color).tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        by_ticker["esg_score"],
        by_ticker["mean_sharpe"],
        c=colors,
        alpha=0.75,
    )

    ax.set_title("ESG score vs mean Sharpe (per ticker, whole period)")
    ax.set_xlabel("ESG score")
    ax.set_ylabel("Mean Sharpe (sharpe_12m)")

    # Legend (can be big; keep readable)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, color=sector_to_color[s], label=s)
        for s in sectors
    ]
    ax.legend(
        handles=handles,
        title="Sector",
        fontsize=8,
        title_fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return outpath

#===========================================================================================================
# Step 4
#===========================================================================================================

def assign_esg_bucket_by_ticker(
    df: pd.DataFrame,
    n_buckets: int = 3,
    score_col: str = "esg_score",
    bucket_col: str = "esg_bucket",
) -> pd.DataFrame:
    required = {"ticker", score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[Step4] Missing required columns: {sorted(missing)}")

    tick = (
        df[["ticker", score_col]]
        .dropna(subset=[score_col])
        .drop_duplicates(subset=["ticker"])
        .copy()
    )

    if tick["ticker"].nunique() < n_buckets:
        raise ValueError("[Step4] Not enough tickers to form buckets.")

    try:
        tick[bucket_col] = pd.qcut(
            tick[score_col],
            q=n_buckets,
            labels=list(range(1, n_buckets + 1)),
        ).astype(int)
    except ValueError:
        r = tick[score_col].rank(method="average", pct=True)
        tick[bucket_col] = pd.cut(
            r,
            bins=n_buckets,
            labels=list(range(1, n_buckets + 1)),
            include_lowest=True,
        ).astype(int)

    out = df.merge(tick[["ticker", bucket_col]], on="ticker", how="left", validate="many_to_one")

    if out[bucket_col].isna().any():
        missing_tickers = out.loc[out[bucket_col].isna(), "ticker"].drop_duplicates().tolist()
        raise ValueError(f"[Step4] Some tickers have no bucket assignment: {missing_tickers[:10]}")

    return out

def compute_portfolio_returns_ew(
    df: pd.DataFrame,
    bucket_col: str = "esg_bucket",
    ret_col: str = "monthly_return",
    rf_col: str = "rf_monthly",
) -> pd.DataFrame:
    required = {"month_id", "ticker", bucket_col, ret_col, rf_col, "regime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[Step4] Missing required columns for portfolio returns: {sorted(missing)}")

    tmp = df.copy()
    tmp["month_id"] = pd.to_numeric(tmp["month_id"], errors="coerce").astype("Int64")
    if tmp["month_id"].isna().any():
        raise ValueError("[Step4] month_id contains NaN in portfolio returns computation.")

    grp = (
        tmp.groupby(["month_id", "regime", bucket_col], as_index=False)
        .agg(
            n_stocks=("ticker", "nunique"),
            port_ret_ew=(ret_col, "mean"),
        )
        .copy()
    )

    rf = (
        tmp.groupby("month_id", as_index=False)
        .agg(rf_monthly=(rf_col, "first"))
        .copy()
    )

    out = grp.merge(rf, on="month_id", how="left", validate="many_to_one")
    out["port_excess"] = out["port_ret_ew"] - out["rf_monthly"]

    if out["rf_monthly"].isna().any():
        bad = out.loc[out["rf_monthly"].isna(), "month_id"].drop_duplicates().tolist()
        raise ValueError(f"[Step4] Missing rf_monthly for month_id(s): {bad[:10]}")

    return out.sort_values(["esg_bucket", "month_id"]).reset_index(drop=True)

def compute_portfolio_sharpe_12m(
    port_returns: pd.DataFrame,
    bucket_col: str = "esg_bucket",
    excess_col: str = "port_excess",
    window: int = 12,
) -> pd.DataFrame:
    required = {"month_id", "regime", bucket_col, excess_col}
    missing = required - set(port_returns.columns)
    if missing:
        raise ValueError(f"[Step4] Missing required columns for portfolio sharpe: {sorted(missing)}")

    df = port_returns.copy().sort_values([bucket_col, "month_id"])

    df["avg_excess_12m"] = df.groupby(bucket_col)[excess_col].transform(
        lambda s: s.rolling(window).mean()
    )
    df["vol_excess_12m"] = df.groupby(bucket_col)[excess_col].transform(
        lambda s: s.rolling(window).std(ddof=1)
    )
    df["sharpe_12m"] = np.sqrt(12) * (df["avg_excess_12m"] / df["vol_excess_12m"])

    return df.reset_index(drop=True)

def build_esg_sorted_portfolios_ew(
    table4: pd.DataFrame,
    results_tables_dir: Path,
    n_buckets: int = 3,
) -> tuple[Path, Path]:
    """
    Builds ESG-sorted (tercile) portfolios using Equal-Weighting (EW),
    computes monthly portfolio returns and rolling 12m Sharpe, then saves CSV outputs.
    """
    results_tables_dir.mkdir(parents=True, exist_ok=True)

    t4b = assign_esg_bucket_by_ticker(table4, n_buckets=n_buckets)

    dist = (
        t4b[["ticker", "esg_bucket"]]
        .drop_duplicates()
        .groupby("esg_bucket")["ticker"]
        .nunique()
        .sort_index()
    )
    print("\n[Step4] Unique tickers per ESG bucket:\n", dist)

    port_ret = compute_portfolio_returns_ew(t4b)
    port_sharpe = compute_portfolio_sharpe_12m(port_ret, window=12)

    out_ret = results_tables_dir / "portfolio_returns_EW.csv"
    out_shp = results_tables_dir / "portfolio_sharpe_EW.csv"

    port_ret.to_csv(out_ret, index=False)
    port_sharpe.to_csv(out_shp, index=False)

    return out_ret, out_shp

# =========================
# Step 5: Star finance plots (EW)
# =========================

def plot_ew_portfolio_sharpe_timeseries(
    portfolio_sharpe_ew: pd.DataFrame,
    outpath: Path,
    bucket_col: str = "esg_bucket",
    sharpe_col: str = "sharpe_12m",
) -> Path:
    """
    Line plot: rolling 12m portfolio Sharpe by ESG bucket across time (EW).
    Drops NaNs (first 11 months by design).
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    required = {"month_id", bucket_col, sharpe_col}
    missing = required - set(portfolio_sharpe_ew.columns)
    if missing:
        raise ValueError(f"[PLOTS] Missing columns for timeseries: {sorted(missing)}")

    df = portfolio_sharpe_ew.dropna(subset=[sharpe_col]).copy()

    df["month_id"] = pd.to_numeric(df["month_id"], errors="coerce")
    if df["month_id"].isna().any():
        raise ValueError("[PLOTS] month_id contains NaN after coercion.")

    df["date"] = pd.to_datetime(df["month_id"].astype(int).astype(str) + "01", format="%Y%m%d")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for b in sorted(df[bucket_col].unique().tolist()):
        tmp = df[df[bucket_col] == b].sort_values("date")
        ax.plot(tmp["date"], tmp[sharpe_col], label=f"Bucket {b}")

    ax.set_title("Portfolio Sharpe (12m rolling) by ESG bucket — Equal-Weighted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe (annualized)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    return outpath

def plot_ew_mean_sharpe_by_regime(
    portfolio_sharpe_ew: pd.DataFrame,
    outpath: Path,
    bucket_col: str = "esg_bucket",
    sharpe_col: str = "sharpe_12m",
    regime_col: str = "regime",
) -> Path:
    """
    Bar chart: mean Sharpe by regime and ESG bucket (EW).
    Drops NaNs (rolling warm-up).
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    required = {bucket_col, sharpe_col, regime_col}
    missing = required - set(portfolio_sharpe_ew.columns)
    if missing:
        raise ValueError(f"[PLOTS] Missing columns for regime bars: {sorted(missing)}")

    df = portfolio_sharpe_ew.dropna(subset=[sharpe_col]).copy()

    agg = (
        df.groupby([regime_col, bucket_col], as_index=False)[sharpe_col]
        .mean()
        .rename(columns={sharpe_col: "mean_sharpe"})
    )

    pivot = agg.pivot(index=regime_col, columns=bucket_col, values="mean_sharpe")

    desired_order = ["normal", "covid", "tightening"]
    pivot = pivot.reindex([r for r in desired_order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Mean Portfolio Sharpe by Regime and ESG Bucket — Equal-Weighted")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Mean Sharpe (annualized)")
    ax.legend(title="ESG bucket", loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    return outpath

def save_ew_high_minus_low_by_regime(
    portfolio_sharpe_ew: pd.DataFrame,
    outpath: Path,
    bucket_col: str = "esg_bucket",
    sharpe_col: str = "sharpe_12m",
    regime_col: str = "regime",
    low_bucket: int = 1,
    high_bucket: int = 3,
) -> Path:
    """
    CSV table: (High bucket - Low bucket) mean Sharpe by regime (EW).
    Drops NaNs (rolling warm-up).
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    required = {"month_id", regime_col, bucket_col, sharpe_col}
    missing = required - set(portfolio_sharpe_ew.columns)
    if missing:
        raise ValueError(f"[TABLE] Missing columns for High-Low: {sorted(missing)}")

    df = portfolio_sharpe_ew.dropna(subset=[sharpe_col]).copy()

    wide = df.pivot_table(
        index=["month_id", regime_col],
        columns=bucket_col,
        values=sharpe_col,
        aggfunc="mean",
    ).reset_index()

    if low_bucket not in wide.columns or high_bucket not in wide.columns:
        found = sorted([c for c in wide.columns if isinstance(c, int)])
        raise ValueError(f"[TABLE] Missing bucket columns in pivot. Found: {found}")

    wide["high_minus_low"] = wide[high_bucket] - wide[low_bucket]

    out = (
        wide.groupby(regime_col, as_index=False)["high_minus_low"]
        .mean()
        .rename(columns={"high_minus_low": "mean_high_minus_low_sharpe"})
        .copy()
    )

    desired_order = ["normal", "covid", "tightening"]
    out[regime_col] = pd.Categorical(out[regime_col], categories=desired_order, ordered=True)
    out = out.sort_values(regime_col).reset_index(drop=True)

    out.to_csv(outpath, index=False)
    return outpath

def create_ew_portfolio_star_outputs(
    portfolio_sharpe_ew: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
) -> tuple[Path, Path, Path]:
    """
    Creates the main EW portfolio outputs:
      1) Sharpe timeseries by bucket
      2) Mean Sharpe by regime and bucket
      3) High-minus-low by regime table
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_ew_portfolio_sharpe_timeseries(
        portfolio_sharpe_ew,
        figures_dir / "portfolio_sharpe_timeseries_EW.png",
    )
    p2 = plot_ew_mean_sharpe_by_regime(
        portfolio_sharpe_ew,
        figures_dir / "sharpe_by_regime_bucket_EW.png",
    )
    p3 = save_ew_high_minus_low_by_regime(
        portfolio_sharpe_ew,
        tables_dir / "high_minus_low_by_regime_EW.csv",
    )

    return p1, p2, p3

# =========================
# Step 6: Star finance plots (VW)
# =========================

def compute_portfolio_returns_vw(
    df: pd.DataFrame,
    bucket_col: str = "esg_bucket",
    ret_col: str = "monthly_return",
    weight_col: str = "index_weight",
    rf_col: str = "rf_monthly",
) -> pd.DataFrame:
    """
    Computes value-weighted (VW) portfolio monthly returns by (month_id, bucket),
    using index_weight as weights within each month-bucket.

    Output columns:
      month_id, regime, esg_bucket, n_stocks, sum_w, port_ret_vw, rf_monthly, port_excess
    """
    required = {"month_id", "ticker", "regime", bucket_col, ret_col, weight_col, rf_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[VW] Missing required columns: {sorted(missing)}")

    tmp = df.copy()
    tmp["month_id"] = pd.to_numeric(tmp["month_id"], errors="coerce").astype("Int64")
    if tmp["month_id"].isna().any():
        raise ValueError("[VW] month_id contains NaN after coercion.")

    # Drop rows without weights/returns/buckets
    tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce")
    tmp[ret_col] = pd.to_numeric(tmp[ret_col], errors="coerce")

    tmp = tmp.dropna(subset=[bucket_col, ret_col, weight_col]).copy()

    # Weighted return contribution
    tmp["wret"] = tmp[weight_col] * tmp[ret_col]

    grp = (
        tmp.groupby(["month_id", "regime", bucket_col], as_index=False)
        .agg(
            n_stocks=("ticker", "nunique"),
            sum_w=(weight_col, "sum"),
            sum_wret=("wret", "sum"),
        )
        .copy()
    )

    grp["port_ret_vw"] = grp["sum_wret"] / grp["sum_w"].replace({0.0: np.nan})

    if grp["port_ret_vw"].isna().any():
        bad = grp.loc[grp["port_ret_vw"].isna(), ["month_id", bucket_col]].head(10)
        raise ValueError(f"[VW] NaN portfolio returns due to zero/NaN weights. Sample:\n{bad}")

    rf = (
        tmp.groupby("month_id", as_index=False)
        .agg(rf_monthly=(rf_col, "first"))
        .copy()
    )

    out = grp.merge(rf, on="month_id", how="left", validate="many_to_one")

    if out["rf_monthly"].isna().any():
        bad_months = out.loc[out["rf_monthly"].isna(), "month_id"].drop_duplicates().tolist()
        raise ValueError(f"[VW] Missing rf_monthly for month_id(s): {bad_months[:10]}")

    out["port_excess"] = out["port_ret_vw"] - out["rf_monthly"]

    return out.sort_values([bucket_col, "month_id"]).reset_index(drop=True)

def build_esg_sorted_portfolios_vw(
    table4: pd.DataFrame,
    results_tables_dir: Path,
    n_buckets: int = 3,
) -> tuple[Path, Path]:
    """
    Builds ESG-sorted portfolios (VW), computes monthly VW returns and rolling 12m Sharpe,
    and saves CSVs.
    """
    results_tables_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the same ticker-level bucket assignment logic
    t4b = assign_esg_bucket_by_ticker(table4, n_buckets=n_buckets)

    dist = (
        t4b[["ticker", "esg_bucket"]]
        .drop_duplicates()
        .groupby("esg_bucket")["ticker"]
        .nunique()
        .sort_index()
    )
    print("\n[VW] Unique tickers per ESG bucket:\n", dist)

    port_ret = compute_portfolio_returns_vw(t4b)
    port_sharpe = compute_portfolio_sharpe_12m(port_ret, window=12)

    out_ret = results_tables_dir / "portfolio_returns_VW.csv"
    out_shp = results_tables_dir / "portfolio_sharpe_VW.csv"

    port_ret.to_csv(out_ret, index=False)
    port_sharpe.to_csv(out_shp, index=False)

    return out_ret, out_shp

def plot_vw_portfolio_sharpe_timeseries(
    portfolio_sharpe_vw: pd.DataFrame,
    outpath: Path,
    bucket_col: str = "esg_bucket",
    sharpe_col: str = "sharpe_12m",
) -> Path:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    required = {"month_id", bucket_col, sharpe_col}
    missing = required - set(portfolio_sharpe_vw.columns)
    if missing:
        raise ValueError(f"[VW PLOTS] Missing columns for timeseries: {sorted(missing)}")

    df = portfolio_sharpe_vw.dropna(subset=[sharpe_col]).copy()
    df["month_id"] = pd.to_numeric(df["month_id"], errors="coerce")
    if df["month_id"].isna().any():
        raise ValueError("[VW PLOTS] month_id contains NaN after coercion.")

    df["date"] = pd.to_datetime(df["month_id"].astype(int).astype(str) + "01", format="%Y%m%d")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for b in sorted(df[bucket_col].unique().tolist()):
        tmp = df[df[bucket_col] == b].sort_values("date")
        ax.plot(tmp["date"], tmp[sharpe_col], label=f"Bucket {b}")

    ax.set_title("Portfolio Sharpe (12m rolling) by ESG bucket — Value-Weighted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe (annualized)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def plot_vw_mean_sharpe_by_regime(
    portfolio_sharpe_vw: pd.DataFrame,
    outpath: Path,
    bucket_col: str = "esg_bucket",
    sharpe_col: str = "sharpe_12m",
    regime_col: str = "regime",
) -> Path:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    required = {bucket_col, sharpe_col, regime_col}
    missing = required - set(portfolio_sharpe_vw.columns)
    if missing:
        raise ValueError(f"[VW PLOTS] Missing columns for regime bars: {sorted(missing)}")

    df = portfolio_sharpe_vw.dropna(subset=[sharpe_col]).copy()

    agg = (
        df.groupby([regime_col, bucket_col], as_index=False)[sharpe_col]
        .mean()
        .rename(columns={sharpe_col: "mean_sharpe"})
        .copy()
    )
    pivot = agg.pivot(index=regime_col, columns=bucket_col, values="mean_sharpe")

    desired_order = ["normal", "covid", "tightening"]
    pivot = pivot.reindex([r for r in desired_order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Mean Portfolio Sharpe by Regime and ESG Bucket — Value-Weighted")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Mean Sharpe (annualized)")
    ax.legend(title="ESG bucket", loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def save_vw_high_minus_low_by_regime(
    portfolio_sharpe_vw: pd.DataFrame,
    outpath: Path,
    bucket_col: str = "esg_bucket",
    sharpe_col: str = "sharpe_12m",
    regime_col: str = "regime",
    low_bucket: int = 1,
    high_bucket: int = 3,
) -> Path:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    required = {"month_id", regime_col, bucket_col, sharpe_col}
    missing = required - set(portfolio_sharpe_vw.columns)
    if missing:
        raise ValueError(f"[VW TABLE] Missing columns: {sorted(missing)}")

    df = portfolio_sharpe_vw.dropna(subset=[sharpe_col]).copy()

    wide = df.pivot_table(
        index=["month_id", regime_col],
        columns=bucket_col,
        values=sharpe_col,
        aggfunc="mean",
    ).reset_index()

    if low_bucket not in wide.columns or high_bucket not in wide.columns:
        found = sorted([c for c in wide.columns if isinstance(c, int)])
        raise ValueError(f"[VW TABLE] Missing bucket columns in pivot. Found: {found}")

    wide["high_minus_low"] = wide[high_bucket] - wide[low_bucket]

    out = (
        wide.groupby(regime_col, as_index=False)["high_minus_low"]
        .mean()
        .rename(columns={"high_minus_low": "mean_high_minus_low_sharpe"})
        .copy()
    )

    desired_order = ["normal", "covid", "tightening"]
    out[regime_col] = pd.Categorical(out[regime_col], categories=desired_order, ordered=True)
    out = out.sort_values(regime_col).reset_index(drop=True)

    out.to_csv(outpath, index=False)
    return outpath

def create_vw_portfolio_star_outputs(
    portfolio_sharpe_vw: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
) -> tuple[Path, Path, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_vw_portfolio_sharpe_timeseries(
        portfolio_sharpe_vw,
        figures_dir / "portfolio_sharpe_timeseries_VW.png",
    )
    p2 = plot_vw_mean_sharpe_by_regime(
        portfolio_sharpe_vw,
        figures_dir / "sharpe_by_regime_bucket_VW.png",
    )
    p3 = save_vw_high_minus_low_by_regime(
        portfolio_sharpe_vw,
        tables_dir / "high_minus_low_by_regime_VW.csv",
    )
    return p1, p2, p3

def save_ew_vs_vw_regime_bucket_comparison(
    portfolio_sharpe_ew: pd.DataFrame,
    portfolio_sharpe_vw: pd.DataFrame,
    outpath: Path,
    bucket_col: str = "esg_bucket",
    sharpe_col: str = "sharpe_12m",
    regime_col: str = "regime",
) -> Path:
    """
    Saves a compact comparison table:
      mean Sharpe (EW), mean Sharpe (VW), and VW - EW by (regime, bucket).
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    def _mean_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
        x = df.dropna(subset=[sharpe_col]).copy()
        t = (
            x.groupby([regime_col, bucket_col], as_index=False)[sharpe_col]
            .mean()
            .rename(columns={sharpe_col: f"mean_sharpe_{label}"})
        )
        return t

    ew = _mean_table(portfolio_sharpe_ew, "EW")
    vw = _mean_table(portfolio_sharpe_vw, "VW")

    merged = ew.merge(vw, on=[regime_col, bucket_col], how="inner", validate="one_to_one")
    merged["VW_minus_EW"] = merged["mean_sharpe_VW"] - merged["mean_sharpe_EW"]

    desired_order = ["normal", "covid", "tightening"]
    merged[regime_col] = pd.Categorical(merged[regime_col], categories=desired_order, ordered=True)
    merged = merged.sort_values([regime_col, bucket_col]).reset_index(drop=True)

    merged.to_csv(outpath, index=False)
    return outpath

# =========================
# Step 7: Regime-interaction regression exports
# =========================

def prepare_regression_panel(
    table4: pd.DataFrame,
    esg_col: str = "esg_score",
    y_col: str = "sharpe_12m",
    require_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Minimal cleaning for the regression panel.
    Ensures regime has 'normal' as reference category and drops missing rows.
    """
    df = table4.copy()

    base_required = ["ticker", "sector", "regime", "rm_excess", esg_col, y_col]
    if require_cols:
        base_required += require_cols

    missing = [c for c in base_required if c not in df.columns]
    if missing:
        raise ValueError(f"[REG] Table 4 missing required columns: {missing}")

    # Categorical regime with stable ordering (important for interpretation)
    df["regime"] = df["regime"].astype(str)
    df["regime"] = pd.Categorical(df["regime"], categories=["normal", "covid", "tightening"], ordered=True)

    # sector categorical
    df["sector"] = df["sector"].astype(str)

    # numeric coercions
    for col in [y_col, esg_col, "rm_excess"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "index_weight" in df.columns:
        df["index_weight"] = pd.to_numeric(df["index_weight"], errors="coerce")

    # drop missing
    df = df.dropna(subset=base_required).copy()

    # Sanity: remove infinite values
    num_cols = [y_col, esg_col, "rm_excess"] + (["index_weight"] if "index_weight" in df.columns else [])
    for col in num_cols:
        df = df[df[col].map(lambda x: pd.notna(x) and pd.api.types.is_number(x))].copy()

    return df.reset_index(drop=True)

def regression_results_table(res, model_name: str) -> pd.DataFrame:
    """
    Creates a tidy coefficient table from a statsmodels result.
    """
    out = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "t": res.tvalues.values,
            "p_value": res.pvalues.values,
        }
    )
    ci = res.conf_int()
    out["ci_low"] = ci[0].values
    out["ci_high"] = ci[1].values

    # Add model-level metadata as columns (repeated per row for convenience)
    out["model"] = model_name
    out["n_obs"] = int(res.nobs)
    out["r2"] = float(res.rsquared)
    out["adj_r2"] = float(res.rsquared_adj)

    return out

def export_regime_interaction_regressions(
    table4: pd.DataFrame,
    tables_dir: Path,
    esg_col: str = "esg_score",
    cov_type: str = "cluster",
) -> tuple[Path, Path | None]:
    """
    Runs:
      A) without index_weight
      B) with index_weight (only if it fits without errors)

    Exports:
      regression_base.csv
      regression_with_index_weight.csv (optional)
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_regression_panel(table4, esg_col=esg_col)

    # Use Treatment(reference="normal") explicitly
    base_formula = (
        f"sharpe_12m ~ {esg_col}"
        f" + C(regime, Treatment(reference='normal'))"
        f" + {esg_col}:C(regime, Treatment(reference='normal'))"
        f" + rm_excess"
        f" + C(sector)"
    )

    spec_a = RegressionSpec(
        name="base",
        formula=base_formula,
        cov_type=cov_type,
        cluster_col="ticker" if cov_type == "cluster" else None,
    )

    res_a = fit_ols(spec_a, df)
    tab_a = regression_results_table(res_a, model_name="A_base_no_index_weight")

    out_a = tables_dir / "regression_base.csv"
    tab_a.to_csv(out_a, index=False)

    # Model B: add index_weight
    out_b: Path | None = None
    try:
        formula_b = base_formula + " + index_weight"
        spec_b = RegressionSpec(
            name="with_index_weight",
            formula=formula_b,
            cov_type=cov_type,
            cluster_col="ticker" if cov_type == "cluster" else None,
        )
        res_b = fit_ols(spec_b, df)
        tab_b = regression_results_table(res_b, model_name="B_with_index_weight")

        out_b = tables_dir / "regression_with_index_weight.csv"
        tab_b.to_csv(out_b, index=False)

    except Exception as e:
        # If unstable/singular, we skip as agreed
        print(f"[REG][WARN] Model with index_weight failed; skipping export. Reason: {e}")

    return out_a, out_b

# =========================
# Step 8: ML block: with vs without ESG (Step 8)
# =========================

def _temporal_split(df: pd.DataFrame, split: TemporalSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["month_id"] = pd.to_numeric(df["month_id"], errors="coerce").astype("Int64")
    if df["month_id"].isna().any():
        raise ValueError("[ML] month_id contains NaN after coercion.")

    train = df[(df["month_id"] >= split.train_start) & (df["month_id"] <= split.train_end)].copy()
    test = df[(df["month_id"] >= split.test_start) & (df["month_id"] <= split.test_end)].copy()

    if train.empty or test.empty:
        raise ValueError(f"[ML] Empty train/test after split. Train rows={len(train)}, Test rows={len(test)}")

    return train, test


def run_ml_with_without_esg(
    table4: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    split: TemporalSplit | None = None,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """
    Builds ML datasets, runs (With ESG) vs (Without ESG) on Ridge/RF/GB,
    and exports:
      - results/tables/ml_metrics_with_without_esg.csv
      - results/figures/ml_metrics_comparison.png
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    split = split or TemporalSplit()

    df = table4.copy()

    # --- Define target ---
    y_col = "sharpe_12m"

    # --- Base features (always included) ---
    # Keep these stable across both specs to isolate ESG incremental signal
    base_numeric = ["rm_excess", "index_weight", "year", "month"]
    base_categorical = ["sector", "regime"]

    # --- ESG features (incremental block) ---
    esg_candidates = ["esg_score", "environment_score", "social_score", "governance_score"]
    esg_features = [c for c in esg_candidates if c in df.columns]

    # Sanity checks
    required = ["month_id", "ticker", y_col] + base_numeric + base_categorical
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[ML] Table 4 missing required columns: {missing}")

    # Coerce numerics
    for c in [y_col] + base_numeric + esg_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop missing core rows
    drop_cols = required + (esg_features if esg_features else [])
    df = df.dropna(subset=required).copy()

    train_df, test_df = _temporal_split(df, split)

    def _evaluate_one_setting(include_esg: bool) -> pd.DataFrame:
        numeric_features = base_numeric + (esg_features if include_esg else [])
        categorical_features = base_categorical

        # Prepare X/y
        X_train = train_df[numeric_features + categorical_features].copy()
        y_train = train_df[y_col].copy()
        X_test = test_df[numeric_features + categorical_features].copy()
        y_test = test_df[y_col].copy()

        preds = fit_predict_ml_suite(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=random_state,
        )

        rows = []
        for model_name, y_pred in preds.items():
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            rows.append(
                {
                    "feature_set": "with_esg" if include_esg else "without_esg",
                    "model": model_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                }
            )

        return pd.DataFrame(rows)

    # Run both settings
    metrics_wo = _evaluate_one_setting(include_esg=False)
    metrics_w = _evaluate_one_setting(include_esg=True)

    metrics = pd.concat([metrics_wo, metrics_w], ignore_index=True)

    # Add deltas per model: with_esg - without_esg
    wide = metrics.pivot(index="model", columns="feature_set", values=["mae", "rmse", "r2"])
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    wide = wide.reset_index()

    wide["delta_r2"] = wide["r2_with_esg"] - wide["r2_without_esg"]
    wide["delta_mae"] = wide["mae_with_esg"] - wide["mae_without_esg"]
    wide["delta_rmse"] = wide["rmse_with_esg"] - wide["rmse_without_esg"]

    # Merge deltas back (optional but helpful)
    metrics_out = metrics.merge(wide[["model", "delta_r2", "delta_mae", "delta_rmse"]], on="model", how="left")

    out_csv = tables_dir / "ml_metrics_with_without_esg.csv"
    metrics_out.to_csv(out_csv, index=False)

    # ---- Figure: 3 metrics side-by-side (With vs Without ESG) ----
    out_fig = figures_dir / "ml_metrics_comparison.png"

    plot_df = metrics.copy()
    plot_df["feature_set"] = pd.Categorical(plot_df["feature_set"], ["without_esg", "with_esg"], ordered=True)
    plot_df = plot_df.sort_values(["model", "feature_set"])

    models = plot_df["model"].unique().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    metric_specs = [("mae", "MAE (lower is better)"), ("rmse", "RMSE (lower is better)"), ("r2", "R² (higher is better)")]

    for ax, (mcol, title) in zip(axes, metric_specs):
        # build grouped bars
        xs = np.arange(len(models))
        w = 0.35

        vals_wo = plot_df[plot_df["feature_set"] == "without_esg"].set_index("model")[mcol].reindex(models).values
        vals_w = plot_df[plot_df["feature_set"] == "with_esg"].set_index("model")[mcol].reindex(models).values

        ax.bar(xs - w / 2, vals_wo, width=w, label="Without ESG")
        ax.bar(xs + w / 2, vals_w, width=w, label="With ESG")

        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(models)

    axes[0].legend(loc="best")
    fig.suptitle("ML Performance: With ESG vs Without ESG (Temporal Split)")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    return out_csv, out_fig

# =========================
# Step 9: ML block: with vs without ESG (Step 8)
# =========================

def run_esg_total_vs_subscores_robustness_ml(
    table4: pd.DataFrame,
    tables_dir: Path,
    split: TemporalSplit | None = None,
    random_state: int = 42,
) -> Path:
    """
    Robustness ML benchmark:
    Compare ESG signal definitions:
      - total_esg: esg_score
      - subscores_esg: environment_score + social_score + governance_score
      - total_plus_subscores: esg_score + E + S + G

    Output:
      results/tables/robustness_esg_vs_esg_subscores.csv
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    split = split or TemporalSplit()

    df = table4.copy()

    # Target
    y_col = "sharpe_12m"

    # Base features (keep identical to Step 8 for comparability)
    base_numeric = ["rm_excess", "index_weight", "year", "month"]
    base_categorical = ["sector", "regime"]

    # ESG columns present?
    needed_esg = ["esg_score", "environment_score", "social_score", "governance_score"]
    missing_esg = [c for c in needed_esg if c not in df.columns]
    if missing_esg:
        raise ValueError(f"[ROBUST ML] Missing ESG columns in table4: {missing_esg}")

    # Required columns for baseline
    required = ["month_id", y_col] + base_numeric + base_categorical + needed_esg
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[ROBUST ML] Missing required columns: {missing}")

    # Coerce numerics
    for c in [y_col] + base_numeric + needed_esg:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["month_id", y_col] + base_numeric + base_categorical).copy()

    # Temporal split
    df["month_id"] = pd.to_numeric(df["month_id"], errors="coerce").astype("Int64")
    train_df = df[(df["month_id"] >= split.train_start) & (df["month_id"] <= split.train_end)].copy()
    test_df = df[(df["month_id"] >= split.test_start) & (df["month_id"] <= split.test_end)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"[ROBUST ML] Empty train/test after split. Train={len(train_df)}, Test={len(test_df)}")

    # Define ESG variants
    variants = {
        "baseline_no_esg": [],
        "total_esg": ["esg_score"],
        "subscores_esg": ["environment_score", "social_score", "governance_score"],
        "total_plus_subscores": ["esg_score", "environment_score", "social_score", "governance_score"],
    }

    rows = []
    for variant_name, esg_feats in variants.items():
        numeric_features = base_numeric + esg_feats
        categorical_features = base_categorical

        X_train = train_df[numeric_features + categorical_features].copy()
        y_train = train_df[y_col].copy()
        X_test = test_df[numeric_features + categorical_features].copy()
        y_test = test_df[y_col].copy()

        preds = fit_predict_ml_suite(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=random_state,
        )

        for model_name, y_pred in preds.items():
            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            rows.append(
                {
                    "variant": variant_name,
                    "model": model_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": r2,
                }
            )

    out = pd.DataFrame(rows)

    # Add deltas vs baseline_no_esg per model
    base = out[out["variant"] == "baseline_no_esg"][["model", "mae", "rmse", "r2"]].rename(
        columns={"mae": "mae_base", "rmse": "rmse_base", "r2": "r2_base"}
    )
    out = out.merge(base, on="model", how="left")
    out["delta_mae"] = out["mae"] - out["mae_base"]
    out["delta_rmse"] = out["rmse"] - out["rmse_base"]
    out["delta_r2"] = out["r2"] - out["r2_base"]

    out_csv = tables_dir / "robustness_esg_vs_esg_subscores.csv"
    out.to_csv(out_csv, index=False)

    return out_csv
