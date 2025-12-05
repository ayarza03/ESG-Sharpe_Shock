# Proposal for the final project (D&AP25-26)
## ESG Risk-Adjusted Performance in Normal Markets and Under Macroeconomic Stress (2010–2024)
Although sustainability, governance quality, and long-term risk exposure are evaluated using ESG scores, it is still unclear if these metrics result in better risk-adjusted financial performance. The goal of this project is to answer the follwing question:
**Does the ESG rating correlate with risk-adjusted performance under normal market conditions, and if so, how does this relationship behave during macroeconomic shocks?**

### --> To answer this question

This project will feature an analysis that estimates the link between ESG scores and Sharpe-adjusted
returns from a broad sample of public firms for years 2010-2024. The project will be built in two stages.
Firstly, a baseline estimation of how ESG relates to Sharpe ratio during a stable expansion period from 2010 to 2019.
The main objective is to establish a neutral reference point for interpretation and further manipulations.
Secondly, a segmentation of the left timeline into two crisis windows, COVID-19 shock (2020-2021) and an inflation tightening shock. Then, a re-evaluation of the coefficients. The results are expected to reveal whether the ESG–Sharpe relationship observed under normal market conditions changes or keeps stable during periods of macroeconomic stress.

### --> About the data used

ESG and E/S/G sub-scores will be obtained from datasets available on public and trusted sources. Financial price data will be collected via 300–500 publicly traded firms, should be from the S&P 500 for a sector-varied and liquidity dataset. Monthly returns will be computed from adjusted close prices. To build the Sharpe ratios, annualized return and volatility will be used.
About the tools used in this project, the use of linear regression for modelling the ESG–Sharpe relationship should be the right call, as well as supervised learning methods (tree-based models) for non-linear extensions. To work with the dataset, methods covered in class (ex: train/test split) can be useful and should be implemented.
To ensure the correct measuring of the ESG ~ Sharpe correlation, there's a need to include control variables such as log-market-capitalization and sector classification in order to reduce confounding effects (a.k.a firm sizes, market distinctions effects). Similarly, the implementation of evaluation methods will ensure model validity and coefficient stability.
