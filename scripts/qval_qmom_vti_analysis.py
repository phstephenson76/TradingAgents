#!/usr/bin/env python3
"""Measure what adding QVAL and QMOM to a VTI core portfolio is worth.

QVAL (Alpha Architect U.S. Quantitative Value) and QMOM (Alpha Architect U.S.
Quantitative Momentum) are concentrated factor ETFs. This script asks whether
they would have improved a plain VTI (total U.S. market) holding, using:

  1. Stand-alone stats for each fund (CAGR, volatility, Sharpe, Sortino, drawdown).
  2. Correlations, including QVAL vs QMOM (value and momentum tend to diversify).
  3. A CAPM regression of each fund on VTI: alpha, beta, and their t-stats.
  4. The "does it raise Sharpe?" test: an asset improves a portfolio's Sharpe
     ratio when its own Sharpe exceeds corr(asset, VTI) * Sharpe(VTI).
  5. A set of VTI/QVAL/QMOM mixes with periodic rebalancing, compared with VTI.
  6. The long-only max-Sharpe mix found in-sample (an upper bound, not a forecast).
  7. Rolling 36-month return of a 70/15/15 mix minus VTI, to show how
     consistent any edge was.

Prices come from Yahoo Finance via yfinance (adjusted for dividends, and net of
each fund's expense ratio since fees come out of NAV). Use --prices-csv to run
on your own data instead.

Examples:
    python scripts/qval_qmom_vti_analysis.py
    python scripts/qval_qmom_vti_analysis.py --start 2018-01-01 --rebalance 3
    python scripts/qval_qmom_vti_analysis.py --rf 0.04 --output-dir results/
    python scripts/qval_qmom_vti_analysis.py --prices-csv my_prices.csv

The CSV must have a date column first and one column per ticker (VTI, QVAL, QMOM).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CORE = "VTI"
SATELLITES = ["QVAL", "QMOM"]
TICKERS = [CORE, *SATELLITES]
PERIODS_PER_YEAR = 12

# (VTI, QVAL, QMOM) weights to compare against 100% VTI.
DEFAULT_MIXES = [
    (1.00, 0.00, 0.00),
    (0.90, 0.05, 0.05),
    (0.80, 0.10, 0.10),
    (0.70, 0.15, 0.15),
    (0.60, 0.20, 0.20),
    (0.50, 0.25, 0.25),
    (0.80, 0.20, 0.00),
    (0.80, 0.00, 0.20),
    (0.00, 0.50, 0.50),
]


# --------------------------------------------------------------------------- data


def load_prices_yahoo(start: str, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed: pip install yfinance (or use --prices-csv)")

    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        sys.exit("Yahoo Finance returned no data. Check your connection or use --prices-csv.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return prices[TICKERS]


def load_risk_free_yahoo(start: str, end: str | None) -> pd.Series | None:
    """Monthly T-bill return from ^IRX (13-week yield, in percent). None on failure."""
    try:
        import yfinance as yf

        irx = yf.download("^IRX", start=start, end=end, auto_adjust=False, progress=False)
        if irx is None or irx.empty:
            return None
        close = irx["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        annual_yield = close.resample("ME").mean() / 100.0
        return (1 + annual_yield) ** (1 / PERIODS_PER_YEAR) - 1
    except Exception:  # noqa: BLE001 - the risk-free rate is optional
        return None


def load_prices_csv(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.columns = [c.strip().upper() for c in prices.columns]
    missing = [t for t in TICKERS if t not in prices.columns]
    if missing:
        sys.exit(f"{path} is missing columns: {', '.join(missing)}")
    return prices[TICKERS].sort_index()


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # Drop rows before every fund existed so all stats share one window.
    prices = prices.dropna()
    month_end = prices.resample("ME").last()
    return month_end.pct_change().dropna()


# ------------------------------------------------------------------------ metrics


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summarize(returns: pd.Series, rf: pd.Series) -> dict[str, float]:
    excess = returns - rf
    years = len(returns) / PERIODS_PER_YEAR
    vol = returns.std() * math.sqrt(PERIODS_PER_YEAR)
    downside = excess.clip(upper=0)
    downside_dev = math.sqrt((downside**2).mean()) * math.sqrt(PERIODS_PER_YEAR)
    mdd = max_drawdown(returns)
    cagr = (1 + returns).prod() ** (1 / years) - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": excess.mean() * PERIODS_PER_YEAR / vol if vol else np.nan,
        "Sortino": excess.mean() * PERIODS_PER_YEAR / downside_dev if downside_dev else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd else np.nan,
        "Worst month": returns.min(),
        "Best month": returns.max(),
    }


def capm_regression(asset: pd.Series, market: pd.Series, rf: pd.Series) -> dict[str, float]:
    """OLS of asset excess returns on VTI excess returns."""
    y = (asset - rf).to_numpy()
    x = (market - rf).to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - 2
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    tracking_error = resid.std(ddof=2) * math.sqrt(PERIODS_PER_YEAR)
    alpha_annual = coef[0] * PERIODS_PER_YEAR
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {
        "Alpha (ann.)": alpha_annual,
        "Alpha t-stat": coef[0] / se[0],
        "Beta": coef[1],
        "Beta t-stat": coef[1] / se[1],
        "R-squared": r2,
        "Tracking error": tracking_error,
        "Information ratio": alpha_annual / tracking_error if tracking_error else np.nan,
    }


def portfolio_returns(
    returns: pd.DataFrame, weights: tuple[float, ...], rebalance: int
) -> pd.Series:
    """Monthly returns of a portfolio rebalanced to target every `rebalance` months.

    Between rebalances the weights drift with each fund's performance.
    """
    target = np.asarray(weights, dtype=float)
    holdings = target.copy()
    out = []
    for i, row in enumerate(returns.to_numpy()):
        if i % rebalance == 0:
            holdings = target * holdings.sum() if i else target.copy()
        start_value = holdings.sum()
        holdings = holdings * (1 + row)
        out.append(holdings.sum() / start_value - 1)
    return pd.Series(out, index=returns.index)


def max_sharpe_mix(
    returns: pd.DataFrame, rf: pd.Series, rebalance: int, step: float
) -> tuple[tuple[float, ...], float]:
    """Long-only weight grid search for the highest in-sample Sharpe ratio."""
    grid = np.round(np.arange(0, 1 + step / 2, step), 6)
    best_w, best_sharpe = (1.0, 0.0, 0.0), -np.inf
    for w_qval, w_qmom in itertools.product(grid, grid):
        w_vti = round(1 - w_qval - w_qmom, 6)
        if w_vti < 0:
            continue
        weights = (w_vti, float(w_qval), float(w_qmom))
        sharpe = summarize(portfolio_returns(returns, weights, rebalance), rf)["Sharpe"]
        if sharpe > best_sharpe:
            best_w, best_sharpe = weights, sharpe
    return best_w, best_sharpe


# ------------------------------------------------------------------------ report


def pct(x: float) -> str:
    return f"{x:7.2%}" if pd.notna(x) else "    n/a"


def num(x: float) -> str:
    return f"{x:7.2f}" if pd.notna(x) else "    n/a"


PCT_FIELDS = {
    "CAGR",
    "Volatility",
    "Max drawdown",
    "Worst month",
    "Best month",
    "Alpha (ann.)",
    "Tracking error",
    "Excess CAGR vs VTI",
}


def format_table(df: pd.DataFrame) -> str:
    formatted = df.copy().astype(object)
    for col in df.columns:
        formatter = pct if col in PCT_FIELDS else num
        formatted[col] = df[col].map(formatter)
    return formatted.to_string()


def mix_label(weights: tuple[float, ...]) -> str:
    return "/".join(f"{w * 100:.0f}" for w in weights)


def section(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--start", default="2015-01-01", help="first date to download (QMOM began trading Dec 2015)"
    )
    parser.add_argument("--end", default=None, help="last date to download (default: today)")
    parser.add_argument(
        "--prices-csv", type=Path, help="read prices from a CSV instead of Yahoo Finance"
    )
    parser.add_argument(
        "--rf",
        type=float,
        default=None,
        help="constant annual risk-free rate, e.g. 0.03 (default: ^IRX T-bill yield, else 0)",
    )
    parser.add_argument(
        "--rebalance", type=int, default=12, help="months between rebalances (default: 12)"
    )
    parser.add_argument(
        "--grid-step", type=float, default=0.05, help="weight step for the max-Sharpe search"
    )
    parser.add_argument("--output-dir", type=Path, help="also write the tables as CSV files here")
    args = parser.parse_args()
    if args.rebalance < 1:
        parser.error("--rebalance must be at least 1")

    prices = (
        load_prices_csv(args.prices_csv)
        if args.prices_csv
        else load_prices_yahoo(args.start, args.end)
    )
    returns = monthly_returns(prices)
    if len(returns) < 24:
        sys.exit(f"Only {len(returns)} months of overlapping data; need at least 24.")

    if args.rf is not None:
        rf = pd.Series((1 + args.rf) ** (1 / PERIODS_PER_YEAR) - 1, index=returns.index)
        rf_source = f"constant {args.rf:.2%}/yr"
    else:
        irx = None if args.prices_csv else load_risk_free_yahoo(args.start, args.end)
        if irx is not None:
            rf = irx.reindex(returns.index).ffill().fillna(0.0)
            rf_source = "13-week T-bill (^IRX)"
        else:
            rf = pd.Series(0.0, index=returns.index)
            rf_source = "0 (pass --rf to set one)"

    print(
        f"Window: {returns.index[0]:%Y-%m} to {returns.index[-1]:%Y-%m} "
        f"({len(returns)} months, {len(returns) / 12:.1f} years)"
    )
    print(f"Risk-free rate: {rf_source}; rebalancing every {args.rebalance} month(s)")

    # 1. Stand-alone stats
    section("1. Stand-alone performance")
    standalone = pd.DataFrame({t: summarize(returns[t], rf) for t in TICKERS}).T
    print(format_table(standalone))

    # 2. Correlations
    section("2. Correlation of monthly returns")
    corr = returns.corr()
    print(corr.round(2).to_string())

    # 3. CAPM vs VTI and 4. Sharpe-improvement test
    section("3. Regression on VTI (excess returns)")
    capm = pd.DataFrame({t: capm_regression(returns[t], returns[CORE], rf) for t in SATELLITES}).T
    print(format_table(capm))
    print("  |alpha t-stat| > 2 is the usual bar for alpha that is unlikely to be luck.")

    section("4. Would a small allocation raise VTI's Sharpe ratio?")
    vti_sharpe = standalone.loc[CORE, "Sharpe"]
    for t in SATELLITES:
        hurdle = corr.loc[t, CORE] * vti_sharpe
        own = standalone.loc[t, "Sharpe"]
        verdict = "YES" if own > hurdle else "NO"
        print(f"  {t}: Sharpe {own:.2f} vs hurdle corr x Sharpe(VTI) = {hurdle:.2f}  ->  {verdict}")
    print(
        "  (Equivalent to a positive alpha against VTI. Ignores the effect of adding both together.)"
    )

    # 5. Blended portfolios
    section(f"5. VTI/QVAL/QMOM mixes (rebalanced every {args.rebalance} months)")
    vti_cagr = standalone.loc[CORE, "CAGR"]
    rows = {}
    mix_returns = {}
    for weights in DEFAULT_MIXES:
        series = portfolio_returns(returns, weights, args.rebalance)
        mix_returns[mix_label(weights)] = series
        stats = summarize(series, rf)
        stats["Excess CAGR vs VTI"] = stats["CAGR"] - vti_cagr
        rows[mix_label(weights)] = stats
    mixes = pd.DataFrame(rows).T
    mixes.index.name = "VTI/QVAL/QMOM %"
    cols = [
        "CAGR",
        "Excess CAGR vs VTI",
        "Volatility",
        "Sharpe",
        "Sortino",
        "Max drawdown",
        "Calmar",
    ]
    print(format_table(mixes[cols]))

    # 6. Optimal in-sample mix
    section("6. In-sample max-Sharpe mix (long-only)")
    best_w, best_sharpe = max_sharpe_mix(returns, rf, args.rebalance, args.grid_step)
    print(
        f"  VTI {best_w[0]:.0%} / QVAL {best_w[1]:.0%} / QMOM {best_w[2]:.0%}  ->  Sharpe {best_sharpe:.2f} "
        f"(VTI alone: {vti_sharpe:.2f})"
    )
    print(
        "  Hindsight-optimised on this window only; treat it as an upper bound, not a recommendation."
    )

    # 7. Consistency
    window = 36
    section(f"7. Rolling {window}-month annualised return: 70/15/15 minus VTI")
    blend = mix_returns["70/15/15"]
    roll = lambda s: (1 + s).rolling(window).apply(np.prod, raw=True) ** (12 / window) - 1  # noqa: E731
    rolling_edge = (roll(blend) - roll(returns[CORE])).dropna()
    if rolling_edge.empty:
        print(f"  Not enough data for a {window}-month window.")
    else:
        print(f"  Windows ahead of VTI: {(rolling_edge > 0).mean():.0%} of {len(rolling_edge)}")
        print(
            f"  Median {rolling_edge.median():+.2%}/yr, best {rolling_edge.max():+.2%}/yr "
            f"({rolling_edge.idxmax():%Y-%m}), worst {rolling_edge.min():+.2%}/yr ({rolling_edge.idxmin():%Y-%m})"
        )

    # Bottom line
    section("Bottom line")
    blend_stats = mixes.loc["70/15/15"]
    vti_stats = mixes.loc["100/0/0"]
    print(
        f"  A 70/15/15 mix returned {blend_stats['CAGR']:.2%}/yr vs {vti_stats['CAGR']:.2%} for VTI "
        f"({blend_stats['Excess CAGR vs VTI']:+.2%}), with Sharpe {blend_stats['Sharpe']:.2f} vs "
        f"{vti_stats['Sharpe']:.2f} and max drawdown {blend_stats['Max drawdown']:.1%} vs "
        f"{vti_stats['Max drawdown']:.1%}."
    )
    print("  Fund returns are already net of expense ratios (QVAL/QMOM charge far more than VTI).")
    print("  Past factor performance is noisy and regime-dependent; this is not investment advice.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        standalone.to_csv(args.output_dir / "standalone.csv")
        corr.to_csv(args.output_dir / "correlations.csv")
        capm.to_csv(args.output_dir / "regression_vs_vti.csv")
        mixes.to_csv(args.output_dir / "mixes.csv")
        pd.DataFrame(mix_returns).to_csv(args.output_dir / "mix_monthly_returns.csv")
        print(f"\nCSV files written to {args.output_dir}/")


if __name__ == "__main__":
    main()
