import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
import os

LOOKBACK_EARNINGS = 8
SCAN_DAYS = 7
MIN_EXPECTED_MOVE_PCT = 0.01  # 1% floor


# =============================
# Helpers
# =============================

def get_stock_price(ticker):
    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        raise ValueError("No price data")
    return hist["Close"].iloc[-1]


def get_earnings_date(ticker):
    t = yf.Ticker(ticker)
    if hasattr(t, "earnings_dates") and not t.earnings_dates.empty:
        return pd.to_datetime(t.earnings_dates.index[0]).tz_localize(None)
    return None


def get_next_expiration(ticker, earnings_date):
    expirations = yf.Ticker(ticker).options
    for exp in expirations:
        exp_dt = pd.to_datetime(exp)
        if exp_dt > earnings_date:
            return exp
    return None


# =============================
# Volatility Models
# =============================

def iv_expected_move(price, iv, dte):
    return price * iv * sqrt(dte / 365)


def realized_vol_move(price, hist, days=7):
    returns = hist["Close"].pct_change().dropna()
    if len(returns) < days:
        return price * returns.std()
    return price * returns[-days:].std()


def get_atm_straddle_move(ticker, expiration, price):
    chain = yf.Ticker(ticker).option_chain(expiration)
    calls, puts = chain.calls, chain.puts

    atm_strike = calls.iloc[(calls["strike"] - price).abs().argsort()[:1]]["strike"].values[0]

    call = calls[calls["strike"] == atm_strike].iloc[0]
    put = puts[puts["strike"] == atm_strike].iloc[0]

    if call["bid"] <= 0 or put["bid"] <= 0:
        return None

    call_mid = (call["bid"] + call["ask"]) / 2
    put_mid = (put["bid"] + put["ask"]) / 2

    return call_mid + put_mid


# =============================
# Historical Earnings Behavior
# =============================

def historical_earnings_analysis(ticker, lookback=8):
    stock = yf.Ticker(ticker)
    prices = stock.history(period="4y")

    if prices.empty or not hasattr(stock, "earnings_dates"):
        return None

    earnings = stock.earnings_dates.index[:lookback]

    actual_moves = []
    implied_moves = []
    signed_moves = []

    for date in earnings:
        date = pd.to_datetime(date)

        try:
            prices_before = prices.loc[prices.index < date]
            prices_after = prices.loc[prices.index > date]

            if len(prices_before) < 1 or len(prices_after) < 1:
                continue

            before = prices_before.iloc[-1]["Close"]
            after = prices_after.iloc[0]["Close"]

            actual = abs(after - before)
            signed = after - before

            hist = prices_before.tail(20)
            iv_proxy = hist["Close"].pct_change().std() * sqrt(252)
            implied = before * iv_proxy * sqrt(2 / 365)

            if implied <= 0:
                continue

            actual_moves.append(actual)
            implied_moves.append(implied)
            signed_moves.append(signed)

        except:
            continue

    if len(actual_moves) < 5:
        return None

    actual_moves = np.array(actual_moves)
    implied_moves = np.array(implied_moves)
    signed_moves = np.array(signed_moves)

    rir = np.mean(actual_moves / implied_moves)
    stability = np.std(actual_moves / implied_moves)
    directional_bias = abs(np.mean(signed_moves)) / np.mean(actual_moves)

    if rir < 0.8 and stability < 0.6:
        regime = "IV_OVERPRICED"
    elif rir > 1.15 and directional_bias < 0.5:
        regime = "IV_UNDERPRICED"
    elif stability > 0.9:
        regime = "CHAOTIC"
    else:
        regime = "NEUTRAL"

    return {
        "Regime": regime,
        "RIR": round(rir, 2),
        "Stability": round(stability, 2),
        "Directional Bias": round(directional_bias, 2)
    }


# =============================
# Probability / Confidence
# =============================

def probability_price_in_range(price, lower, upper, expected_move):
    sigma = expected_move / price
    z_low = (lower - price) / (price * sigma)
    z_high = (upper - price) / (price * sigma)
    return norm.cdf(z_high) - norm.cdf(z_low)


def confidence_score(pop, stability, rir):
    score = (
        0.5 * pop +
        0.3 * (1 - min(stability, 1)) +
        0.2 * min(rir / 1.2, 1)
    )
    return round(score * 100, 2)


# =============================
# Main Analysis
# =============================

def analyze_earnings_trade(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")

    if hist.empty:
        return None

    price = get_stock_price(ticker)
    earnings_date = get_earnings_date(ticker)

    if not earnings_date:
        return None

    expiration = get_next_expiration(ticker, earnings_date)
    if not expiration:
        return None

    dte = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.today()).days
    if dte <= 0:
        return None

    chain = stock.option_chain(expiration)
    front_iv = chain.calls["impliedVolatility"].dropna().median()

    if not front_iv or front_iv <= 0:
        return None

    straddle = get_atm_straddle_move(ticker, expiration, price)
    iv_move = iv_expected_move(price, front_iv, dte)
    rv_move = realized_vol_move(price, hist)

    expected_move = (
        (0.5 * straddle if straddle else 0) +
        0.3 * iv_move +
        0.2 * rv_move
    )

    expected_move = max(expected_move, price * MIN_EXPECTED_MOVE_PCT)

    hist_behavior = historical_earnings_analysis(ticker)
    if not hist_behavior:
        return None

    regime = hist_behavior["Regime"]

    if regime == "IV_OVERPRICED":
        trade_type = "IRON_CONDOR"
        buffer = 1.3
    elif regime == "IV_UNDERPRICED":
        trade_type = "IRON_FLY"
        buffer = 0.9
    else:
        trade_type = "SKIP"
        buffer = 1.5

    lower = price - expected_move * buffer
    upper = price + expected_move * buffer

    pop = probability_price_in_range(price, lower, upper, expected_move)
    confidence = confidence_score(
        pop,
        hist_behavior["Stability"],
        hist_behavior["RIR"]
    )

    return {
        "Ticker": ticker,
        "Price": round(price, 2),
        "Earnings Date": earnings_date.date(),
        "Expiration": expiration,
        "Expected Move ($)": round(expected_move, 2),
        "Recommended Trade": trade_type,
        "Lower Bound": round(lower, 2) if trade_type != "SKIP" else "N/A",
        "Upper Bound": round(upper, 2) if trade_type != "SKIP" else "N/A",
        "Confidence Score": confidence,
        "Earnings Regime": regime
    }


# =============================
# SP500 Scan
# =============================

def get_sp500_tickers():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "sp500_tickers.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    return data["tickers"]


def scan_sp500_earnings():
    results = []
    tickers = get_sp500_tickers()
    cutoff = datetime.now().date() + timedelta(days=SCAN_DAYS)

    for t in tickers:
        try:
            r = analyze_earnings_trade(t)
            if not r:
                continue

            if (
                r["Confidence Score"] >= 60 and
                r["Earnings Date"] <= cutoff
            ):
                results.append(r)

        except:
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("Confidence Score", ascending=False)


# =============================
# Run
# =============================

if __name__ == "__main__":

    print("\nSelect mode:")
    print("1 → Analyze single ticker")
    print("2 → Scan S&P 500 earnings (next 7 days)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        ticker = input("Enter ticker: ").upper()
        result = analyze_earnings_trade(ticker)

        if not result:
            print("No valid earnings trade found.")
        else:
            print("\n--- Earnings Trade Analysis ---")
            for k, v in result.items():
                print(f"{k}: {v}")

    elif choice == "2":
        print("\nScanning S&P 500 earnings...\n")
        df = scan_sp500_earnings()

        if df.empty:
            print("No high-confidence earnings trades found.")
        else:
            print(df.head(15))
            df.to_csv("earnings_trade_candidates.csv", index=False)
            print("\nSaved to earnings_trade_candidates.csv")

    else:
        print("Invalid selection.")
