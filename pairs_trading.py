"""
Pairs Trading Strategy Backtester
===================================
A quantitative pairs trading strategy using cointegration analysis.

Strategy logic:
  - Find two stocks whose prices are cointegrated (move together long-term)
  - Compute the spread between them and normalise to a Z-score
  - When Z-score > +2: short the spread (sell A, buy B)
  - When Z-score < -2: long the spread (buy A, sell B)
  - Exit when Z-score reverts to 0

Author: Doreen Mwangi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ── Colour palette (warm, consistent with portfolio) ──────────────────────────
C_TERRA   = "#C4623A"
C_SAGE    = "#7A8C6E"
C_BROWN   = "#3A2A1E"
C_AMBER   = "#D4922A"
C_CREAM   = "#FAF6EF"
C_MUTED   = "#8A6D5A"
C_BORDER  = "#E2D4C8"

plt.rcParams.update({
    "figure.facecolor": C_CREAM,
    "axes.facecolor":   C_CREAM,
    "axes.edgecolor":   C_BORDER,
    "axes.labelcolor":  C_BROWN,
    "axes.titlecolor":  C_BROWN,
    "xtick.color":      C_MUTED,
    "ytick.color":      C_MUTED,
    "grid.color":       C_BORDER,
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Usage:
        import yfinance as yf
        prices = yf.download(tickers, start=start, end=end)["Adj Close"]

    For demonstration, we generate synthetic correlated price series.
    Replace this function body with the yfinance call when running locally.
    """
    print(f"Loading price data for: {tickers}")
    print(f"Period: {start} to {end}\n")

    # ── Synthetic data (replace with yfinance in your local run) ─────────────
    np.random.seed(42)
    dates = pd.date_range(start=start, end=end, freq="B")  # business days
    n = len(dates)

    prices = {}
    # Create a shared "market" factor so stocks are correlated
    market = np.cumsum(np.random.normal(0.0003, 0.01, n))

    for i, ticker in enumerate(tickers):
        # Individual stock = market factor + idiosyncratic noise
        idio = np.cumsum(np.random.normal(0.0001, 0.008, n))
        shock = np.random.normal(0, 0.002, n)  # small daily noise
        log_price = market * (0.7 + i * 0.05) + idio * 0.3 + shock
        prices[ticker] = 100 * np.exp(log_price - log_price[0])

    return pd.DataFrame(prices, index=dates)


def load_data_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Real data version — uncomment and use this when running locally.

    pip install yfinance
    """
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers].dropna()
    print(f"Downloaded {len(prices)} trading days of data.\n")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2. COINTEGRATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def find_cointegrated_pairs(prices: pd.DataFrame, pvalue_threshold: float = 0.05):
    """
    Test all pairs in the universe for cointegration using the
    Engle-Granger two-step method.

    Returns a DataFrame of significant pairs sorted by p-value.
    """
    tickers = prices.columns.tolist()
    results = []

    print("=" * 55)
    print("  COINTEGRATION ANALYSIS")
    print("=" * 55)

    for t1, t2 in combinations(tickers, 2):
        score, pvalue, _ = coint(prices[t1], prices[t2])
        results.append({
            "pair":    f"{t1} / {t2}",
            "ticker1": t1,
            "ticker2": t2,
            "p_value": round(pvalue, 4),
            "cointegrated": pvalue < pvalue_threshold
        })

    df = pd.DataFrame(results).sort_values("p_value")
    cointegrated = df[df["cointegrated"]]

    print(f"\nPairs tested: {len(df)}")
    print(f"Cointegrated pairs (p < {pvalue_threshold}): {len(cointegrated)}")
    print()
    print(df[["pair", "p_value", "cointegrated"]].to_string(index=False))
    print()

    return df, cointegrated


def compute_hedge_ratio(prices: pd.DataFrame, ticker1: str, ticker2: str) -> float:
    """
    Estimate the hedge ratio (beta) using OLS regression:
        price1 = beta * price2 + alpha + epsilon
    """
    y = prices[ticker1].values
    x = add_constant(prices[ticker2].values)
    model = OLS(y, x).fit()
    beta = model.params[1]
    return beta


def compute_spread(prices: pd.DataFrame, ticker1: str, ticker2: str,
                   lookback: int = 60) -> pd.Series:
    """
    Compute rolling hedge ratio and spread.
    Uses a rolling OLS window so the hedge ratio adapts over time.
    """
    spread = pd.Series(index=prices.index, dtype=float)

    for i in range(lookback, len(prices)):
        window = prices.iloc[i - lookback:i]
        beta = compute_hedge_ratio(window, ticker1, ticker2)
        spread.iloc[i] = prices[ticker1].iloc[i] - beta * prices[ticker2].iloc[i]

    return spread.dropna()


def compute_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling Z-score of the spread:
        z = (spread - rolling_mean) / rolling_std
    """
    mean = spread.rolling(window).mean()
    std  = spread.rolling(window).std()
    return ((spread - mean) / std).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 3. BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def backtest(prices: pd.DataFrame, ticker1: str, ticker2: str,
             entry_z: float = 2.0, exit_z: float = 0.5,
             lookback: int = 60, zscore_window: int = 30,
             initial_capital: float = 100_000) -> dict:
    """
    Run the pairs trading backtest.

    Signal logic:
      z > +entry_z  → short spread (sell t1, buy t2)
      z < -entry_z  → long spread  (buy t1, sell t2)
      |z| < exit_z  → close position

    Returns a dict with trades, daily P&L, and portfolio value series.
    """
    spread  = compute_spread(prices, ticker1, ticker2, lookback)
    zscore  = compute_zscore(spread, zscore_window)

    # Align prices to z-score index
    px = prices.loc[zscore.index]

    position   = 0          # +1 = long spread, -1 = short spread, 0 = flat
    cash       = initial_capital
    portfolio  = []
    trades     = []
    pnl_daily  = []

    entry_price1 = entry_price2 = 0.0
    shares1 = shares2 = 0
    beta = 1.0

    for i, date in enumerate(zscore.index):
        z  = zscore.loc[date]
        p1 = px.loc[date, ticker1]
        p2 = px.loc[date, ticker2]

        # ── Entry signals ──────────────────────────────────────────────────
        if position == 0:
            if z > entry_z:
                # Short spread: sell t1, buy t2
                beta   = compute_hedge_ratio(px.iloc[max(0, i-lookback):i+1], ticker1, ticker2)
                shares1 = int(cash * 0.4 / p1)
                shares2 = int(shares1 * beta)
                cash   += shares1 * p1   # proceeds from short
                cash   -= shares2 * p2   # cost of long
                position = -1
                entry_price1, entry_price2 = p1, p2
                trades.append({"date": date, "action": "SHORT SPREAD",
                               "z_score": round(z, 2), "price1": p1, "price2": p2})

            elif z < -entry_z:
                # Long spread: buy t1, sell t2
                beta   = compute_hedge_ratio(px.iloc[max(0, i-lookback):i+1], ticker1, ticker2)
                shares1 = int(cash * 0.4 / p1)
                shares2 = int(shares1 * beta)
                cash   -= shares1 * p1   # cost of long
                cash   += shares2 * p2   # proceeds from short
                position = 1
                entry_price1, entry_price2 = p1, p2
                trades.append({"date": date, "action": "LONG SPREAD",
                               "z_score": round(z, 2), "price1": p1, "price2": p2})

        # ── Exit signal ────────────────────────────────────────────────────
        elif abs(z) < exit_z:
            if position == 1:   # unwind long spread
                cash += shares1 * p1
                cash -= shares2 * p2
            elif position == -1:  # unwind short spread
                cash -= shares1 * p1
                cash += shares2 * p2

            trades.append({"date": date, "action": "EXIT",
                           "z_score": round(z, 2), "price1": p1, "price2": p2})
            position = 0
            shares1 = shares2 = 0

        # ── Mark-to-market portfolio value ─────────────────────────────────
        unrealised = 0
        if position == 1:
            unrealised = shares1 * (p1 - entry_price1) - shares2 * (p2 - entry_price2)
        elif position == -1:
            unrealised = -shares1 * (p1 - entry_price1) + shares2 * (p2 - entry_price2)

        portfolio_value = cash + unrealised
        portfolio.append(portfolio_value)
        pnl_daily.append(portfolio_value - (portfolio[-2] if len(portfolio) > 1 else initial_capital))

    portfolio_series = pd.Series(portfolio, index=zscore.index)
    pnl_series       = pd.Series(pnl_daily, index=zscore.index)

    return {
        "portfolio":   portfolio_series,
        "pnl":         pnl_series,
        "trades":      pd.DataFrame(trades),
        "spread":      spread,
        "zscore":      zscore,
        "ticker1":     ticker1,
        "ticker2":     ticker2,
        "entry_z":     entry_z,
        "exit_z":      exit_z,
        "initial_capital": initial_capital,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: dict) -> dict:
    """Compute key quantitative performance metrics."""
    portfolio = results["portfolio"]
    pnl       = results["pnl"]
    trades    = results["trades"]
    capital   = results["initial_capital"]

    returns = portfolio.pct_change().dropna()

    total_return = (portfolio.iloc[-1] / capital - 1) * 100
    annual_return = ((portfolio.iloc[-1] / capital) ** (252 / len(portfolio)) - 1) * 100

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    rolling_max  = portfolio.cummax()
    drawdown     = (portfolio - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    n_trades = len(trades[trades["action"] != "EXIT"]) if not trades.empty else 0
    win_rate = None  # would need P&L per trade for this

    metrics = {
        "Total Return (%)":      round(total_return, 2),
        "Annualised Return (%)": round(annual_return, 2),
        "Sharpe Ratio":          round(sharpe, 3),
        "Max Drawdown (%)":      round(max_drawdown, 2),
        "Number of Trades":      n_trades,
        "Final Portfolio (£)":   round(portfolio.iloc[-1], 2),
    }

    print("=" * 55)
    print("  PERFORMANCE METRICS")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<28} {v:>12}")
    print()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, metrics: dict, save_path: str = None):
    """
    Four-panel dashboard:
      1. Normalised stock prices
      2. Spread over time
      3. Z-score with entry/exit signals
      4. Cumulative portfolio value
    """
    t1       = results["ticker1"]
    t2       = results["ticker2"]
    spread   = results["spread"]
    zscore   = results["zscore"]
    portfolio= results["portfolio"]
    trades   = results["trades"]
    entry_z  = results["entry_z"]
    exit_z   = results["exit_z"]

    fig = plt.figure(figsize=(14, 11), facecolor=C_CREAM)
    fig.suptitle(
        f"Pairs Trading Backtester  ·  {t1} / {t2}",
        fontsize=15, fontweight="bold", color=C_BROWN, y=0.98
    )

    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # ── Panel 1: Normalised prices ─────────────────────────────────────────
    ax = axes[0]
    idx = zscore.index
    # only plot prices over the backtest window
    p = results["spread"].index  # use spread index as reference

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    # Reload prices subset for plotting
    prices_plot = pd.DataFrame({
        t1: 100 * np.ones(len(idx)),  # placeholder; real script uses actual prices
        t2: 100 * np.ones(len(idx)),
    }, index=idx)

    # Pull actual prices from the spread computation context (passed via closure-like dict)
    if "prices" in results:
        px = results["prices"].loc[idx]
        norm1 = px[t1] / px[t1].iloc[0] * 100
        norm2 = px[t2] / px[t2].iloc[0] * 100
        ax.plot(idx, norm1, color=C_TERRA,  linewidth=1.4, label=t1)
        ax.plot(idx, norm2, color=C_SAGE,   linewidth=1.4, label=t2)

    ax.set_title("Normalised Price Series (rebased to 100)", fontsize=10, pad=6)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.4)

    # ── Panel 2: Spread ────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(spread.index, spread, color=C_AMBER, linewidth=1.2, label="Spread")
    ax.axhline(spread.mean(), color=C_MUTED, linestyle="--", linewidth=0.9, label="Mean")
    ax.fill_between(spread.index, spread, spread.mean(),
                    where=(spread > spread.mean()), alpha=0.15, color=C_TERRA)
    ax.fill_between(spread.index, spread, spread.mean(),
                    where=(spread < spread.mean()), alpha=0.15, color=C_SAGE)
    ax.set_title("Price Spread", fontsize=10, pad=6)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.4)

    # ── Panel 3: Z-score with signals ─────────────────────────────────────
    ax = axes[2]
    ax.plot(zscore.index, zscore, color=C_BROWN, linewidth=1.1, alpha=0.85, label="Z-score")
    ax.axhline(0,        color=C_MUTED,  linestyle="-",  linewidth=0.8)
    ax.axhline(entry_z,  color=C_TERRA,  linestyle="--", linewidth=0.9, label=f"+{entry_z} (entry)")
    ax.axhline(-entry_z, color=C_SAGE,   linestyle="--", linewidth=0.9, label=f"-{entry_z} (entry)")
    ax.axhline(exit_z,   color=C_AMBER,  linestyle=":",  linewidth=0.8, label=f"±{exit_z} (exit)")
    ax.axhline(-exit_z,  color=C_AMBER,  linestyle=":",  linewidth=0.8)
    ax.fill_between(zscore.index, entry_z,  zscore.max(), alpha=0.07, color=C_TERRA)
    ax.fill_between(zscore.index, zscore.min(), -entry_z, alpha=0.07, color=C_SAGE)

    # Plot trade signals
    if not trades.empty:
        for _, trade in trades.iterrows():
            if trade["date"] in zscore.index:
                z_val = zscore.loc[trade["date"]]
                if trade["action"] == "SHORT SPREAD":
                    ax.scatter(trade["date"], z_val, marker="v", color=C_TERRA, s=60, zorder=5)
                elif trade["action"] == "LONG SPREAD":
                    ax.scatter(trade["date"], z_val, marker="^", color=C_SAGE, s=60, zorder=5)
                elif trade["action"] == "EXIT":
                    ax.scatter(trade["date"], z_val, marker="x", color=C_AMBER, s=40, zorder=5)

    ax.set_title("Z-score  ▲ Long spread  ▼ Short spread  ✕ Exit", fontsize=10, pad=6)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.grid(True, alpha=0.4)

    # ── Panel 4: Portfolio value ───────────────────────────────────────────
    ax = axes[3]
    capital = results["initial_capital"]
    ax.plot(portfolio.index, portfolio, color=C_TERRA, linewidth=1.5, label="Portfolio")
    ax.axhline(capital, color=C_MUTED, linestyle="--", linewidth=0.9, label=f"Capital (£{capital:,.0f})")
    ax.fill_between(portfolio.index, capital, portfolio,
                    where=(portfolio >= capital), alpha=0.18, color=C_SAGE)
    ax.fill_between(portfolio.index, capital, portfolio,
                    where=(portfolio < capital),  alpha=0.18, color=C_TERRA)

    # Annotate final return
    final = portfolio.iloc[-1]
    ret   = (final / capital - 1) * 100
    ax.annotate(
        f"{'▲' if ret >= 0 else '▼'} {ret:+.1f}%  (£{final:,.0f})",
        xy=(portfolio.index[-1], final),
        xytext=(-100, 12), textcoords="offset points",
        fontsize=9, color=C_TERRA if ret >= 0 else C_SAGE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=0.8)
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.set_title("Cumulative Portfolio Value", fontsize=10, pad=6)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.4)

    # Shared x-label
    axes[-1].set_xlabel("Date", fontsize=9)

    # Metrics box
    metrics_text = (
        f"Sharpe: {metrics['Sharpe Ratio']}   "
        f"Return: {metrics['Total Return (%)']:+.1f}%   "
        f"Max DD: {metrics['Max Drawdown (%)']:.1f}%   "
        f"Trades: {metrics['Number of Trades']}"
    )
    fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=9,
             color=C_BROWN, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_CREAM)
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(tickers=None, start="2020-01-01", end="2023-12-31",
        entry_z=2.0, exit_z=0.5, lookback=60,
        zscore_window=30, initial_capital=100_000,
        save_chart="outputs/pairs_trading_results.png"):

    if tickers is None:
        # UK bank / financial sector pairs — great for quant finance framing
        tickers = ["LLOY.L", "BARC.L", "HSBA.L", "NWG.L", "STAN.L"]

    print("\n" + "=" * 55)
    print("  PAIRS TRADING STRATEGY BACKTESTER")
    print("  Author: Doreen Mwangi")
    print("=" * 55 + "\n")

    # 1. Load data
    # prices = load_data_yfinance(tickers, start, end)   # ← use this locally
    prices = load_data(tickers, start, end)               # synthetic demo

    print(f"Universe: {tickers}")
    print(f"Observations: {len(prices)} trading days\n")

    # 2. Find best pair
    all_pairs, cointegrated = find_cointegrated_pairs(prices)

    if cointegrated.empty:
        print("No cointegrated pairs found. Try a different universe or time period.")
        return

    best = cointegrated.iloc[0]
    t1, t2 = best["ticker1"], best["ticker2"]
    print(f"Best pair selected: {t1} / {t2}  (p = {best['p_value']})\n")

    # 3. Run backtest
    print("=" * 55)
    print("  RUNNING BACKTEST")
    print("=" * 55)
    results = backtest(prices, t1, t2,
                       entry_z=entry_z, exit_z=exit_z,
                       lookback=lookback, zscore_window=zscore_window,
                       initial_capital=initial_capital)
    results["prices"] = prices  # pass through for plotting

    # 4. Metrics
    metrics = compute_metrics(results)

    # 5. Trades log
    if not results["trades"].empty:
        print("=" * 55)
        print("  TRADE LOG (first 10)")
        print("=" * 55)
        print(results["trades"].head(10).to_string(index=False))
        print()

    # 6. Plot
    plot_results(results, metrics, save_path=save_chart)

    print("\nDone! Check outputs/ for the chart.")
    return results, metrics


if __name__ == "__main__":
    run(
        tickers=["LLOY.L", "BARC.L", "HSBA.L", "NWG.L", "STAN.L"],
        start="2020-01-01",
        end="2023-12-31",
        entry_z=2.0,
        exit_z=0.5,
        lookback=60,
        zscore_window=30,
        initial_capital=100_000,
        save_chart="outputs/pairs_trading_results.png"
    )
