import pandas as pd

def create_features(prices):
    """
    Generates features based on 12-month momentum, volatility, and mean reversion.

    Parameters:
    - prices: pd.DataFrame. Monthly close prices for each ticker.

    Returns:
    - features: pd.DataFrame. Feature matrix with columns like ticker_mom, ticker_vol, ticker_resid.
    """
    features = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        # Momentum: 12-month minus 1-month return
        features[f"{ticker}_mom"] = prices[ticker].pct_change(12) - prices[ticker].pct_change(1)

        # Volatility: rolling standard deviation over 12 months
        features[f"{ticker}_vol"] = prices[ticker].pct_change().rolling(12).std()

        # Residual: price deviation from its 12-month moving average
        features[f"{ticker}_resid"] = prices[ticker] - prices[ticker].rolling(12).mean()
    return features.dropna()
