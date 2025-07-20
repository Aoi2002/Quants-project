import yfinance as yf
import pandas as pd

# Download and preprocess monthly adjusted close prices
def load_price_data(tickers, start_date="2005-01-01"):
    """
    Fetches monthly adjusted close prices using Yahoo Finance and calculates monthly returns.

    Parameters:
    - tickers: list of str. Asset symbols to download.
    - start_date: str. Start date for historical data.

    Returns:
    - prices: pd.DataFrame. Monthly adjusted close prices.
    - returns: pd.DataFrame. Monthly percentage returns.
    """
    prices = yf.download(tickers, start=start_date, auto_adjust=True)["Close"]
    prices = prices.resample("ME").last()  # Resample to month-end frequency
    returns = prices.pct_change().dropna()
    return prices, returns

