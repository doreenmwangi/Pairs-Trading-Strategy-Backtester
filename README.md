<<<<<<< HEAD
# Pairs Trading Strategy Backtester

A quantitative pairs trading strategy built in Python — using cointegration analysis, rolling OLS hedge ratios, and Z-score signal generation to backtest a market-neutral mean-reversion strategy on real stock data.

**Author:** Doreen Mwangi  
**Stack:** Python · pandas · statsmodels · yfinance · matplotlib

---

## What is Pairs Trading?

Pairs trading is a **market-neutral** quantitative strategy. The idea:

1. Find two stocks whose prices are **cointegrated** — they move together long-term
2. When the spread between them diverges significantly, bet on **mean reversion**
3. Go **long the underperformer**, **short the outperformer**
4. Exit when the spread reverts to its historical mean

Because you're simultaneously long and short, the strategy is largely hedged against broad market moves — it profits from the *relative* performance of the two stocks, not the market direction.

---

## Strategy Logic

```
spread  = price_A - beta * price_B       (beta estimated via rolling OLS)
z_score = (spread - rolling_mean) / rolling_std

Signal:
  z > +2.0  →  SHORT spread  (sell A, buy B)
  z < -2.0  →  LONG spread   (buy A, sell B)
  |z| < 0.5 →  EXIT position
```

---

## Project Structure

```
pairs_trading/
├── pairs_trading.py          # Core backtester (data, signals, engine, plots)
├── notebooks/
│   └── pairs_trading_notebook.ipynb   # Step-by-step walkthrough
├── outputs/                  # Charts saved here
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/doreenmwangi/pairs-trading
cd pairs-trading
pip install -r requirements.txt

# 2. Run the backtester
python pairs_trading.py

# 3. Or explore the notebook
jupyter notebook notebooks/pairs_trading_notebook.ipynb
```

### Customise the universe

```python
from pairs_trading import run

# UK banks
run(tickers=["LLOY.L", "BARC.L", "HSBA.L", "NWG.L"],
    start="2020-01-01", end="2023-12-31")

# US tech
run(tickers=["MSFT", "GOOGL", "META", "AMZN"],
    start="2019-01-01", end="2023-12-31")

# Adjust thresholds
run(tickers=["LLOY.L", "BARC.L"],
    entry_z=1.5, exit_z=0.3, initial_capital=50_000)
```

---

## Output

The backtester produces a four-panel dashboard:

| Panel | Description |
|-------|-------------|
| Normalised prices | Both stocks rebased to 100 |
| Spread | Price spread with mean highlighted |
| Z-score | Signal line with entry/exit thresholds and trade markers |
| Portfolio value | Cumulative P&L vs initial capital |

Plus a printed performance summary:

```
Total Return (%)           +12.4%
Annualised Return (%)       +4.1%
Sharpe Ratio                 0.83
Max Drawdown (%)            -6.2%
Number of Trades               14
Final Portfolio (£)       112,400
```

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Cointegration** | Two series share a long-run equilibrium. Tested via Engle-Granger. |
| **Hedge ratio (beta)** | How many units of stock B to trade per unit of stock A. Estimated with OLS. |
| **Spread** | `price_A - beta * price_B` — the quantity that mean-reverts. |
| **Z-score** | Normalised spread. Tells us how many standard deviations we are from the mean. |
| **Sharpe Ratio** | Risk-adjusted return: `mean(daily_return) / std(daily_return) * sqrt(252)` |
| **Max Drawdown** | Largest peak-to-trough decline — measures downside risk. |

---

## Extensions

- Add **stop-loss** (exit if Z > ±3 — spread diverging, not converging)
- Deduct **transaction costs** for realistic P&L
- Run a **portfolio of pairs** simultaneously, weighted by Sharpe
- Replace rolling OLS with a **Kalman filter** for adaptive hedge ratios
- Add **regime filtering** — only trade in low-volatility environments

---

## Disclaimer

This project is for educational and portfolio purposes only. It is not financial advice.
=======
# Pairs-Trading-Strategy-Backtester
A quantitative pairs trading strategy built in Python using cointegration analysis, rolling OLS hedge ratios, and Z-score signal generation to backtest a market-neutral mean-reversion strategy on real stock data.
>>>>>>> 9a402f8afd92809909e7f5e5750f55fe9b4a608b
