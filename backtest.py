import pandas as pd
import numpy as np

def evaluate_strategy(cum_returns, monthly_returns, name):
    """
    Print key performance metrics for a strategy.

    Parameters:
    - cum_returns: pd.Series. Cumulative returns (index: datetime).
    - monthly_returns: pd.Series. Monthly returns.
    - name: str. Name of the strategy for labeling.
    """
    if isinstance(cum_returns, pd.DataFrame):
        cum_returns = cum_returns.iloc[:, 0]
    if isinstance(monthly_returns, pd.DataFrame):
        monthly_returns = monthly_returns.iloc[:, 0]

    total_months = len(monthly_returns)
    total_years = total_months / 12

    cumulative_return = cum_returns.iloc[-1] - 1
    annual_return = (cum_returns.iloc[-1])**(1 / total_years) - 1
    sharpe_ratio = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)

    # Drawdown calculation
    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1
    max_drawdown = drawdown.min()

    # Max drawdown duration
    dd_flag = (drawdown < 0).astype(int)
    dd_periods = dd_flag.groupby((dd_flag == 0).cumsum()).sum()
    max_dd_duration = dd_periods.max()

    print(f"\n=== {name} ===")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Annual Return (CAGR): {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Max Drawdown Duration (months): {max_dd_duration}")

