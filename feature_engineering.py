import pandas as pd

# Generate features for each ticker: momentum, volatility, residual (from MA)
def create_features(prices):
    features = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        features[f"{ticker}_mom"] = prices[ticker].pct_change(12) - prices[ticker].pct_change(1)
        features[f"{ticker}_vol"] = prices[ticker].pct_change().rolling(12).std()
        features[f"{ticker}_resid"] = prices[ticker] - prices[ticker].rolling(12).mean()
    return features.dropna()
