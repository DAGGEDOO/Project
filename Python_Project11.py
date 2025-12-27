#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 11:26:00 2025

@author: dagvallien
"""
#%%

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram
from matplotlib.collections import LineCollection
from scipy.optimize import minimize


#%%

# Assets
tickers = ["SPY", "QQQ", "IWM", "EFA", "IEF", "GLD", "VNQ", "DBC", "BTC-USD", "DAX"]

# Adjusted close prices
prices = yf.download(
    tickers,
    start="2015-01-02",
    end="2025-12-01",
    auto_adjust=True,
)["Close"]
#Drop missing data
prices=prices.dropna()

print(prices.head())

#%%

#Converting prices to returns

returns=np.log(prices/prices.shift(1)).dropna()

print(returns.head())

#%%
# Returns Correlation
corr=returns.corr()
print(corr)


#%%

# Correlation heatmap 

plt.figure(figsize=(12,10))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)

plt.title("Asset Return Correlation Heatmap")
plt.tight_layout()
plt.show()


#%%

# Convert correlation matric to distance matrix
distance_matrix=np.sqrt(2*(1-corr))

# Convert to condensed form (required by scipy)
distance_condensed=squareform(distance_matrix, checks=False)

# Hierarchical clustering
linkage_matrix=linkage(distance_condensed, method="average")


#%%

# Creating dendrogram

fig, ax = plt.subplots(figsize=(13, 9))

dendrogram(
    linkage_matrix,
    labels=corr.columns,
    leaf_rotation=45,
    leaf_font_size=11,
    color_threshold=cut_height,
    above_threshold_color="grey",
    ax=ax
)

# Make dendrogram branches thicker 
for coll in ax.collections:
    if isinstance(coll, LineCollection):
        coll.set_linewidth(3.3)   

ax.set_title("Hierarchical Clustering of Asset Returns", fontsize=14, fontweight="bold")
ax.set_ylabel("Distance (Correlation-based)", fontsize=11)
ax.set_xlabel("Assets", fontsize=11)
ax.grid(axis="y", linestyle=":", alpha=0.4)

plt.tight_layout()
plt.show()

#%%
from scipy.cluster.hierarchy import fcluster

# Grouping the different assets into clusters.
n_clusters = 4

cluster_labels = fcluster(
    linkage_matrix,
    t=n_clusters,
    criterion="maxclust"
)

clusters = pd.Series(cluster_labels, index=returns.columns, name="cluster")
print(clusters.sort_values())


#%%

# Expected returns annulized
ex = returns.mean() * 252
# Covariance matrix annulized
cov=returns.cov() * 252

#%%

cluster_map = {
    c: clusters[clusters == c].index.tolist()
    for c in clusters.unique()
}



#%%

# Cluster-constrained Sharpe optimization

def max_sharpe_cluster(ex, cov, cluster_map, cap=0.45, rf=0.0):
    assets=ex.index.tolist()
    n=len(assets)
    # Initial guess: equal-weight portfolio
    x0=np.ones(n)/n 
    
    # Objective function: negative Sharpe ratio
    def neg_sharpe(w):
        ret=w @ ex.values
        vol=np.sqrt(w @ cov.values @ w)
        return -(ret-rf)/vol
    
    # Constraint: portfolio weights must sum to 1
    constraints=[{"type": "eq", "fun": lambda w: np.sum(w)-1}]
    
    # Cluster-level diversification constraints
    for names in cluster_map.values():
        idx=[assets.index(a) for a in names]
        constraints.append({
            "type":"ineq",
            "fun": lambda w, idx=idx: cap - np.sum(w[idx])
            })
    
    # Long-only weights
    bounds=[(0,1)]*n
    res=minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
    return pd.Series(res.x, index=assets)

w_sharpe_cluster=max_sharpe_cluster(ex, cov, cluster_map, cap=0.45)
print(w_sharpe_cluster.sort_values(ascending=False))


#%%

monthly_contribution=1000

# Monthly investment dates (first trading day of each month)

dca_dates = prices.resample("MS").first().index

#%%
# Prices for SPY and Price of all asssets in Sharpe Portfolio
assets_sharpe = w_sharpe_cluster[w_sharpe_cluster > 1e-6].index.tolist()
prices_sharpe = prices[assets_sharpe]
#%%
# Dollar Cost Averaging engine

def dca_portfolio(prices, weights, dca_dates, contribution):
    shares = pd.Series(0.0, index=prices.columns)
    values = []

    for date in prices.index:
        # Invest on DCA dates
        if date in dca_dates:
            for asset in prices.columns:
                invest_amount = contribution * weights[asset]
                shares[asset] += invest_amount / prices.loc[date, asset]

        # Portfolio value today
        value = (shares * prices.loc[date]).sum()
        values.append(value)

    return pd.Series(values, index=prices.index)

#%%
# DCA into SP500

# SP500-only weights
w_spy = pd.Series({"SPY": 1.0})

prices_spy = prices[["SPY"]]

portfolio_spy_dca = dca_portfolio(
    prices=prices_spy,
    weights=w_spy,
    dca_dates=dca_dates,
    contribution=monthly_contribution
)

#%%
# DCA in Sharpe Portfolio

portfolio_sharpe_dca = dca_portfolio(
    prices=prices_sharpe,
    weights=w_sharpe_cluster[assets_sharpe],
    dca_dates=dca_dates,
    contribution=monthly_contribution
)

#%%

portfolio_spy_dca.plot(label="DCA SP500", linewidth=2)
portfolio_sharpe_dca.plot(label="DCA Sharpe Portfolio", linewidth=2)

plt.title("Dollar Cost Averaging: SP500 vs Sharpe Portfolio")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.tight_layout()
plt.show()

#%%
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

#%%
# ----- Descriptive stats table -----
stats = pd.DataFrame({
    "Ann. Return": returns.mean() * 252,
    "Ann. Vol": returns.std() * np.sqrt(252),
})
stats["Sharpe (rf=0)"] = stats["Ann. Return"] / stats["Ann. Vol"]
stats = stats.round(3)

# ----- Figures -----
fig_heatmap = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    title="Correlation Heatmap (Returns)"
)

fig_weights = px.bar(
    w_sharpe_cluster[w_sharpe_cluster > 1e-6].sort_values(ascending=False),
    title="Sharpe-Optimized Portfolio Weights",
    labels={"index": "Asset", "value": "Weight"}
)

fig_dca = go.Figure()
fig_dca.add_trace(go.Scatter(x=portfolio_spy_dca.index, y=portfolio_spy_dca.values,
                             mode="lines", name="DCA SP500"))
fig_dca.add_trace(go.Scatter(x=portfolio_sharpe_dca.index, y=portfolio_sharpe_dca.values,
                             mode="lines", name="DCA Sharpe Portfolio"))
fig_dca.update_layout(title="Dollar Cost Averaging: SP500 vs Sharpe Portfolio",
                      yaxis_title="Portfolio Value ($)")

cluster_df = pd.DataFrame({
    "Asset": clusters.index,
    "Cluster": clusters.values
}).sort_values(["Cluster", "Asset"])

# ----- Dash app -----
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Clustering + Sharpe Optimization Dashboard"),

    html.H3("Key Descriptive Statistics"),
    html.Div([
        html.Pre(stats.to_string())
    ], style={"backgroundColor": "#f6f6f6", "padding": "10px"}),

    html.H3("Clustering"),
    dcc.Graph(figure=fig_heatmap),

    html.H4("Cluster Membership"),
    html.Div([
        html.Pre(cluster_df.to_string(index=False))
    ], style={"backgroundColor": "#f6f6f6", "padding": "10px"}),

    html.H3("Portfolio Optimization"),
    dcc.Graph(figure=fig_weights),

    html.H3("Backtest / Narrative Plot"),
    dcc.Graph(figure=fig_dca),
])

if __name__ == "__main__":
    app.run(debug=True, port=8051)
    
