import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt
from datetime import timedelta
from scipy.stats import norm

# =============================
# Core Move Model (Same Logic)
# =============================

def iv_expected_move(price, iv, dte):
    return price * iv * sqrt(dte / 365)


def historical_earnings_move_pct(ticker, lookback=8):
    stock = yf.Ticker(ticker)
    prices = stock.history(period="4y")
    earnings = stock.earnings_dates

    moves = []
    for date in earnings.index[:lookback]:
        date = pd.to_datetime(date)
        if date not in prices.index:
            continue
        try:
            before = prices.loc[:date].iloc[-2]["Close"]
            after = prices.loc[date:].iloc[1]["Close"]
            moves.append(abs(after - before) / before)
        except:
            continue

    return np.mean(moves) if len(moves) >= 3 else None


def realized_vol_move(price, hist, days=7):
    returns = hist["Close"].pct_change().dropna()
    return price * returns[-days:].std()


def expected_move_model(price, iv, dte, hist_move_pct, rv_move, straddle_proxy):
    components = []

    if straddle_proxy:
        components.append((0.45, straddle_proxy))
    components.append((0.25, iv_expected_move(price, iv, dte)))
    if hist_move_pct:
        components.append((0.20, hist_move_pct * price))
    components.append((0.10, rv_move))

    return sum(w * v for w, v in components)



# =============================
# Backtest Engine
# =============================

def backtest_earnings_range(
    ticker,
    lookback=8,
    buffer=1.25
):
    stock = yf.Ticker(ticker)
    earnings = stock.earnings_dates.index[:lookback]
    prices = stock.history(period="4y")

    results = []

    for earnings_date in earnings:
        earnings_date = pd.to_datetime(earnings_date)

        # Find trading days
        try:
            pre_day = prices.loc[:earnings_date].iloc[-2]
            post_day = prices.loc[earnings_date:].iloc[1]
        except:
            continue

        price_before = pre_day["Close"]
        price_after = post_day["Close"]
        actual_move = abs(price_after - price_before)

        # Proxy IV (7D realized → conservative)
        hist_slice = prices.loc[:earnings_date].tail(30)
        iv_proxy = hist_slice["Close"].pct_change().std() * sqrt(252)

        # Model components
        hist_move_pct = historical_earnings_move_pct(ticker)
        rv_move = realized_vol_move(price_before, hist_slice)

        # Straddle proxy (fallback)
        straddle_proxy = iv_expected_move(price_before, iv_proxy, 2)

        expected_move = expected_move_model(
            price_before,
            iv_proxy,
            dte=2,
            hist_move_pct=hist_move_pct,
            rv_move=rv_move,
            straddle_proxy=straddle_proxy
        )

        lower = price_before - expected_move * buffer
        upper = price_before + expected_move * buffer

        win = lower <= price_after <= upper

        results.append({
            "Earnings Date": earnings_date.date(),
            "Pre Price": round(price_before, 2),
            "Post Price": round(price_after, 2),
            "Actual Move ($)": round(actual_move, 2),
            "Expected Move ($)": round(expected_move, 2),
            "Range Low": round(lower, 2),
            "Range High": round(upper, 2),
            "Inside Range": win
        })

    df = pd.DataFrame(results)

    summary = {
        "Ticker": ticker,
        "Trades Tested": len(df),
        "Wins": int(df["Inside Range"].sum()),
        "Losses": int(len(df) - df["Inside Range"].sum()),
        "Win Rate (%)": round(df["Inside Range"].mean() * 100, 2),
        "Avg Actual Move ($)": round(df["Actual Move ($)"].mean(), 2),
        "Avg Expected Move ($)": round(df["Expected Move ($)"].mean(), 2)
    }

    return df, summary


# =============================
# Run
# =============================

if __name__ == "__main__":
    df, summary = backtest_earnings_range("FDX", lookback=8)

    print("\n--- Earnings Range Backtest (FDX) ---\n")
    print(df.to_string(index=False))

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
