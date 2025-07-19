import numpy as np
import pandas as pd

# Compute compound annual growth rate (CAGR)
def calculate_cagr(cumulative_returns):
    n_years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
    return cumulative_returns.iloc[-1] ** (1 / n_years) - 1

# Compute annualized Sharpe ratio (assumes monthly returns)
def calculate_sharpe(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(12)

# Compute maximum drawdown and the date it occurred
def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = cumulative_returns / peak - 1
    return drawdown.min(), drawdown.idxmin()

# Print evaluation summary of the strategy performance
def evaluate_strategy(cum_returns, monthly_returns, name):
    # Convert to Series if DataFrame
    if isinstance(cum_returns, pd.DataFrame):
        cum_returns = cum_returns.iloc[:, 0]
    if isinstance(monthly_returns, pd.DataFrame):
        monthly_returns = monthly_returns.iloc[:, 0]

    total_months = len(monthly_returns)
    total_years = total_months / 12
    cumulative_return = cum_returns.iloc[-1] - 1
    annual_return = (cum_returns.iloc[-1]) ** (1 / total_years) - 1
    sharpe_ratio = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)

    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1
    max_drawdown = drawdown.min()

    dd_flag = (drawdown < 0).astype(int)
    dd_periods = dd_flag.groupby((dd_flag == 0).cumsum()).sum()
    max_dd_duration = dd_periods.max()

    # Print formatted results
    print(f"\n=== {name} ===")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Annualized Return (CAGR): {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Max Drawdown Duration (months): {max_dd_duration}")
