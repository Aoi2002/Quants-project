import yfinance as yf
import pandas as pd

# Download and preprocess monthly adjusted close prices
def load_price_data(tickers, start_date="2005-01-01"):
    prices = yf.download(tickers, start=start_date, auto_adjust=True)["Close"]
    prices = prices.resample("ME").last()  # Resample to month-end frequency
    returns = prices.pct_change().dropna()  # Monthly log returns
    return prices, returns
