# Project
This project uses hierarchical clustering and Sharpe optimization to construct a diversified investment portfolio and compares its dollar-cost-averaging performance to the S&amp;P 500, illustrating both diversification benefits and look-ahead bias.

Project Overview

This project explores portfolio diversification and investment strategy design using financial data and machine learning techniques. Historical asset price data is used to identify relationships between assets, construct diversified portfolios, and compare investment outcomes using dollar-cost averaging. The goal is to demonstrate how unsupervised learning and portfolio optimization can be combined, while also highlighting common pitfalls such as look-ahead bias.

Data Collection and Preprocessing

Historical adjusted closing prices are downloaded using the yfinance library for a diversified set of assets, including equities, bonds, commodities, real estate, and Bitcoin. Prices are converted into log returns to ensure stationarity and comparability across assets. Missing data is handled by aligning assets on common trading dates to ensure valid correlation and covariance estimates.

Correlation Analysis

A correlation matrix of asset returns is computed to capture how assets move relative to one another. This matrix serves as the foundation for all subsequent analysis. A heatmap visualization is used to provide an intuitive overview of asset relationships and to highlight potential diversification benefits.

Hierarchical Clustering

Hierarchical clustering is applied to the correlation-based distance matrix to group assets based on similarity in return behavior. The resulting dendrogram reveals natural groupings of assets that correspond to economically meaningful risk factors, such as equities, defensive assets, and alternative investments. This unsupervised learning step provides a data-driven way to identify diversification structures in the market.

Cluster Assignment

The dendrogram is converted into a fixed number of clusters using flat clustering. Each asset is assigned a cluster label, transforming the hierarchical structure into discrete risk groups. These clusters are later used to constrain portfolio optimization and prevent excessive concentration in highly correlated assets.

Cluster-Constrained Sharpe Optimization

A Sharpe ratio–maximizing portfolio is constructed using historical mean returns and the covariance matrix of asset returns. To enforce diversification, constraints are imposed that limit the maximum total weight allocated to any single cluster. This approach combines classical mean–variance optimization with machine learning–based structure, producing a portfolio that balances risk-adjusted performance and diversification.

Dollar-Cost Averaging Strategy

Two dollar-cost averaging (DCA) strategies are implemented and compared. The first invests exclusively in the S&P 500, while the second invests in the Sharpe-optimized, cluster-diversified portfolio. Both strategies invest a fixed amount at regular intervals, allowing for a fair comparison of portfolio construction rather than timing.

Performance Comparison

The cumulative value of each DCA strategy is tracked over time and visualized using equity curves. Risk-adjusted performance metrics, including volatility and Sharpe ratio, are computed to complement raw return comparisons. This highlights differences not only in final portfolio value but also in risk exposure.

Look-Ahead Bias and Limitations

Portfolio weights are estimated using the full historical sample, which introduces look-ahead bias. In particular, assets such as Bitcoin with strong historical performance receive large weights that may not have been predictable in real time. Rather than correcting for this bias, the project intentionally highlights it as a learning outcome, illustrating the sensitivity of Sharpe optimization to estimation error.

Visualization Dashboard

An interactive dashboard built with Plotly and Dash summarizes key results, including descriptive statistics, correlation structure, cluster assignments, portfolio weights, and DCA performance. The dashboard supports exploratory analysis and provides a clear visual narrative connecting clustering, optimization, and investment outcomes.

Key Takeaways

This project demonstrates how unsupervised learning can enhance portfolio construction by identifying meaningful diversification structures. At the same time, it illustrates the limitations of in-sample optimization and the importance of careful interpretation of backtest results. The combination of clustering, optimization, and strategy evaluation provides a realistic view of both the power and the pitfalls of quantitative portfolio design.
