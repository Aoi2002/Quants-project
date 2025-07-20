import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ===== 1. PARAMETERS =====
tickers = ["XLU", "XLF", "XLE", "XLB", "XBI", "VNQ", "EFA", "IEF", "LQD", "GLD", "TLT", "DBC", "ICLN"]
start_date = "2005-01-01"
transaction_cost_rate = 0.001  # 0.1% per round-trip
top_n_assets = 6  # Top assets based on risk-adjusted score

# ===== 2. DATA LOADING =====
prices = yf.download(tickers, start=start_date, auto_adjust=True)["Close"]
prices = prices.resample("ME").last()
returns = prices.pct_change().dropna()

# ===== 3. FEATURE ENGINEERING =====
def create_features(prices):
    """
    Create monthly momentum, volatility, and mean-reversion features.
    """
    features = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        features[f"{ticker}_mom"] = prices[ticker].pct_change(12) - prices[ticker].pct_change(1)
        features[f"{ticker}_vol"] = prices[ticker].pct_change().rolling(12).std()
        features[f"{ticker}_resid"] = prices[ticker] - prices[ticker].rolling(12).mean()
    return features.dropna()

features = create_features(prices)
target = returns.shift(-1)  # Next month's return

# ===== 4. STRATEGY BACKTEST =====
strategy_returns = []
net_returns = []
dates_out = []
prev_weights = None
turnovers = []

dates = features.index[60:-1]  # Use 5 years of training data
for date in dates:
    X_train = features.loc[:date].iloc[-60:]
    y_train = target.loc[:date].iloc[-60:]
    X_pred = features.loc[[date - pd.offsets.MonthEnd(1)]]

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_pred)[0], index=returns.columns)

    volatility = returns.rolling(12).std().loc[date]
    risk_adjusted_score = y_pred / volatility

    top_assets_all = risk_adjusted_score.nlargest(top_n_assets)

    past_12m_return = prices.pct_change(12).loc[date]
    filtered_assets = top_assets_all[top_assets_all.index.map(lambda x: (y_pred[x] > 0) and (past_12m_return[x] > 0))]

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

# ===== 5. KELLY SCALING =====
m = strategy_returns.mean()
s2 = strategy_returns.var()
kelly_leverage = m / s2
f_adjusted = kelly_leverage * 0.2  # Fractional Kelly for risk control

kelly_returns = net_returns * f_adjusted
kelly_cum = (1 + kelly_returns).cumprod()
net_cum = (1 + net_returns).cumprod()
gross_cum = (1 + strategy_returns).cumprod()

# ===== 6. BENCHMARK: S&P500 =====
spy = yf.download("SPY", start=start_date, auto_adjust=True)["Close"]
spy = spy.resample("ME").last()
spy_returns = spy.pct_change().dropna().reindex(strategy_returns.index)
spy_cum = (1 + spy_returns.dropna()).cumprod()

# ===== 7. EQUAL-WEIGHTED BUY & HOLD =====
equal_weight_returns = returns[tickers].mean(axis=1)
equal_weight_cum = (1 + equal_weight_returns).cumprod()

common_start = max(spy_cum.index[0], kelly_cum.index[0], equal_weight_cum.index[0])
spy_cum = spy_cum[spy_cum.index >= common_start]
kelly_cum = kelly_cum[kelly_cum.index >= common_start]
equal_weight_cum = equal_weight_cum[equal_weight_cum.index >= common_start]

# ===== 8. VISUALIZATION =====
plt.figure(figsize=(20, 8))
plt.plot(spy_cum, label="S&P500", linewidth=2)
plt.plot(kelly_cum, label=f"ML Strategy (Kelly x {f_adjusted:.2f})", linewidth=2)
plt.plot(equal_weight_cum, label="Equal Weight Buy & Hold", linewidth=2, color="green")
plt.yscale("log")
plt.title("ML Momentum Strategy vs S&P500 vs Equal Weight Buy & Hold")
plt.legend(loc="upper left", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("results/performance_chart.png")
plt.close()

# ===== 9. PERFORMANCE METRICS =====
def evaluate_strategy(cum_returns, monthly_returns, name):
    if isinstance(cum_returns, pd.DataFrame):
        cum_returns = cum_returns.iloc[:, 0]
    if isinstance(monthly_returns, pd.DataFrame):
        monthly_returns = monthly_returns.iloc[:, 0]

    total_months = len(monthly_returns)
    total_years = total_months / 12
    cumulative_return = cum_returns.iloc[-1] - 1
    annual_return = (cum_returns.iloc[-1])**(1 / total_years) - 1
    sharpe_ratio = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)

    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1
    max_drawdown = drawdown.min()

    dd_flag = (drawdown < 0).astype(int)
    dd_periods = dd_flag.groupby((dd_flag == 0).cumsum()).sum()
    max_dd_duration = dd_periods.max()

    print(f"\n=== {name} ===")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Annual Return (CAGR): {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Max Drawdown Duration (months): {max_dd_duration}")

evaluate_strategy(spy_cum, spy_returns, "S&P500")
evaluate_strategy(kelly_cum, kelly_returns, "ML Kelly Strategy")


# Evaluate strategies
evaluate_strategy(spy_cum, spy_returns, "S&P500")
evaluate_strategy(kelly_cum, kelly_returns, "Kelly Strategy")
