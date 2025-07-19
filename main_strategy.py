import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import load_price_data
from src.feature_engineering import create_features
from evaluation.performance_metrics import evaluate_strategy

# Define target ETFs
tickers = ["XLU", "XLF", "XLE", "XLB", "XBI", "VNQ", "EFA", "IEF", "LQD", "GLD", "TLT", "DBC", "ICLN"]

# Load price and return data
prices, returns = load_price_data(tickers)

# Feature creation and target setup
features = create_features(prices)
target = returns.shift(-1)

# Strategy logic
strategy_returns = []
net_returns = []
dates_out = []
prev_weights = None
turnovers = []
transaction_cost_rate = 0.001

# Backtest loop
dates = features.index[60:-1]
for date in dates:
    X_train = features.loc[:date].iloc[-60:]
    y_train = target.loc[:date].iloc[-60:]
    X_pred = features.loc[[date - pd.offsets.MonthEnd(1)]]

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_pred)[0], index=returns.columns)

    volatility = returns.rolling(12).std().loc[date]
    risk_adjusted_score = y_pred / volatility
    top_assets = risk_adjusted_score.nlargest(6)

    past_12m_return = prices.pct_change(12).loc[date]
    filtered_assets = top_assets[top_assets.index.map(lambda x: (y_pred[x] > 0) and (past_12m_return[x] > 0))]

    if filtered_assets.empty:
        weights = pd.Series(0, index=returns.columns)
    else:
        weights = filtered_assets / filtered_assets.sum()

    next_date = date + pd.offsets.MonthEnd(1)

    if prev_weights is not None:
        aligned_weights = weights.reindex(prev_weights.index).fillna(0)
        turnover = (aligned_weights - prev_weights).abs().sum()
    else:
        turnover = weights.abs().sum()

    turnovers.append(turnover)
    prev_weights = weights.copy()

    gross_ret = (returns.loc[next_date, weights.index] * weights).sum()
    cost = turnover * transaction_cost_rate
    net_ret = gross_ret - cost

    strategy_returns.append(gross_ret)
    net_returns.append(net_ret)
    dates_out.append(next_date)

strategy_returns = pd.Series(strategy_returns, index=dates_out)
net_returns = pd.Series(net_returns, index=dates_out)

# Kelly leverage application
m = strategy_returns.mean()
s2 = strategy_returns.var()
kelly_leverage = m / s2
f_adjusted = kelly_leverage * 0.2

kelly_returns = net_returns * f_adjusted
kelly_cum = (1 + kelly_returns).cumprod()
net_cum = (1 + net_returns).cumprod()
gross_cum = (1 + strategy_returns).cumprod()

# Benchmark: S&P500 and Equal Weight Portfolio
spy = load_price_data(["SPY"])[0]["SPY"]
spy = spy.resample("ME").last()
spy_returns = spy.pct_change().dropna().reindex(strategy_returns.index)
spy_cum = (1 + spy_returns).cumprod()

equal_weight_returns = returns[tickers].mean(axis=1)
equal_weight_cum = (1 + equal_weight_returns).cumprod()

# Align start dates
common_start = max(spy_cum.index[0], kelly_cum.index[0], equal_weight_cum.index[0])
spy_cum = spy_cum[spy_cum.index >= common_start]
kelly_cum = kelly_cum[kelly_cum.index >= common_start]
equal_weight_cum = equal_weight_cum[equal_weight_cum.index >= common_start]

# Normalize
spy_cum /= spy_cum.iloc[0]
kelly_cum /= kelly_cum.iloc[0]
equal_weight_cum /= equal_weight_cum.iloc[0]

# Plot
plt.figure(figsize=(20, 8))
plt.plot(spy_cum, label="S&P500", linewidth=2, color='blue')
plt.plot(kelly_cum, label=f"Strategy (Kelly x {f_adjusted:.2f})", linewidth=2, color='orange')
plt.plot(equal_weight_cum, label="Equal Weight Buy & Hold", linestyle='--', linewidth=2, color='gray')

plt.yscale("log")
plt.title("ML Momentum Strategy vs SPY vs Equal Weight Buy & Hold")
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("results/backtest_plot.png")
plt.show()

# Evaluate strategies
evaluate_strategy(spy_cum, spy_returns, "S&P500")
evaluate_strategy(kelly_cum, kelly_returns, "Kelly Strategy")
