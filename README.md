# Quants-project

# 📈 ML Momentum Strategy with Kelly Leverage

A machine learning-based momentum strategy that systematically selects high-performing ETFs using 12-1 momentum, volatility scaling, and residual filtering. The model reallocates monthly and applies Kelly criterion to size risk optimally. Over 18+ years, this strategy consistently outperforms SPY with lower drawdown and transaction costs included.

## 🔍 Highlights

- ✅ **CAGR**: 19.9% | **Sharpe Ratio**: 0.97 | **Max Drawdown**: -22.9%
- ✅ **Benchmark**: S&P500 (CAGR: 13.0%), Equal-Weight Buy & Hold (~10.7%)
- ✅ **Turnover-aware**, **transaction-cost adjusted**, and **Kelly-leveraged**
- ✅ Fully reproducible, interpretable, and extensible ML allocation system

## 🧠 Strategy Logic

1. **Model**: Random Forest Regressor trained on past 60 months
2. **Features** per asset:
   - 12-1 momentum: difference between 12-month and 1-month return
   - Volatility: 12M rolling std
   - Residual: deviation from 12M moving average
3. **Scoring**: Risk-adjusted return = prediction / volatility
4. **Filtering**: Only assets with positive predicted and absolute momentum
5. **Weighting**: Proportional to score, normalized
6. **Kelly Leverage**: `f = (mean / variance) × 0.2`
7. **Transaction Costs**: 0.1% per unit turnover deducted from net return

## 📊 Backtest Performance

> Period: 2015–Present (10years) 
> Universe: 13 ETFs (XLU, XLF, XLE, XLB, XBI, VNQ, EFA, IEF, LQD, GLD, TLT, DBC, ICLN)

### 🧭 Design Principles
Practical: Monthly rebalance, realistic costs, turnover penalty
Modular: All logic separated into reusable functions/modules
Expandable: Ready for model upgrades (e.g. XGBoost, LightGBM, LSTM), new signals, and out-of-sample testing

## 📈 Backtest Result

The following chart compares the cumulative returns of:

- Machine Learning Momentum Strategy (with fractional Kelly scaling)
- S&P500 benchmark (SPY)
- Equal-weighted Buy & Hold portfolio

![Performance Chart](results/performance_chart.png)


## 📚 References & Inspirations

This project was inspired and informed by the following excellent resources:

- Ernest P. Chan, *Quantitative Trading: How to Build Your Own Algorithmic Trading Business*, 2nd Edition (Wiley Trading)
- Quantified Strategies Blog: ["Python Momentum Trading Strategy"](https://www.quantifiedstrategies.com/python-momentum-trading-strategy/)

Special thanks to the authors and contributors of these resources for their clear explanations and valuable insights into systematic trading design and implementation.


