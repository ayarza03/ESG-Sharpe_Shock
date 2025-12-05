# Final Project Proposal
## ESG Risk-Adjusted Performance in Normal Markets and Under Macroeconomic Stress (2010–2024)
ESG ratings have become increasingly influential in investment decision frameworks, yet empirical
results remain mixed and often context-dependent. Some studies suggest that high-ESG firms show 
greater resilience in uncertainty, while others find no performance premium once risk is adjusted.

Does ESG correlate with superior or inferior risk-adjusted performance under normal market conditions,
and does this relationship strengthen, weaken or disappear during macroeconomic shocks?

--> What I Will Build

I will develop a reproducible Python-based analysis that estimates the link between ESG scores and Sharpe-adjusted
returns for a broad sample of public firms from 2010 to 2024. The project will be implemented in two stages.
First, I will construct a baseline estimate of how ESG relates to Sharpe ratio during a stable expansion period.
This establishes a neutral reference point for interpretation.
Second, I will segment the timeline into two crisis windows and re-evaluate the coefficients. These results will
reveal whether ESG behaves as a defensive factor, loses relevance, or presents no statistically measurable impact in stress environments.

The model and data pipeline will be executed through a structured repository including main.py for reproducibility,
a data/ module for processing, and results/ for outputs and figures.

--> What Data I Will Use

ESG_total and E/S/G sub-scores will be obtained from Yahoo Finance Sustainability fields and/or ESG datasets available
on Kaggle. Financial price data will be collected via yfinance for 200–400 publicly traded firms, ideally from the S&P 500
for sector breadth and liquidity.

Monthly returns will be computed from adjusted close prices. Annualized return and volatility will be used to construct Sharpe ratios.
Controls will include log-market-capitalization and sector classification to reduce confounding effects.
