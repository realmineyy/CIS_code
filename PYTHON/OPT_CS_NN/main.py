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


def get_next_expirations(ticker, earnings_date):
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
    return price * returns[-days:].std()


def get_atm_straddle_move(ticker, expiration, price):
    chain = yf.Ticker(ticker).option_chain(expiration)
    calls, puts = chain.calls, chain.puts

    atm_strike = calls.iloc[(calls["strike"] - price).abs().argsort()[:1]]["strike"].values[0]

    call = calls[calls["strike"] == atm_strike].iloc[0]
    put = puts[puts["strike"] == atm_strike].iloc[0]

    if call["bid"] == 0 or put["bid"] == 0:
        return None

    return (call["bid"] + call["ask"]) / 2 + (put["bid"] + put["ask"]) / 2


# =============================
# Historical Earnings Behavior
# =============================

def historical_earnings_analysis(ticker, lookback=8):
    stock = yf.Ticker(ticker)
    prices = stock.history(period="4y")
    earnings = stock.earnings_dates.index[:lookback]

    actual_moves = []
    implied_moves = []
    signed_moves = []

    for date in earnings:
        date = pd.to_datetime(date)

        try:
            before = prices.loc[:date].iloc[-2]["Close"]
            after = prices.loc[date:].iloc[1]["Close"]

            actual = abs(after - before)
            signed = after - before

            # IMPLIED MOVE PROXY:
            # Use 1-day IV-style proxy (earnings are overnight)
            hist = prices.loc[:date].tail(20)
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
        return {"Regime": "INSUFFICIENT_DATA"}

    actual_moves = np.array(actual_moves)
    implied_moves = np.array(implied_moves)
    signed_moves = np.array(signed_moves)

    # ---------------------------
    # Metrics
    # ---------------------------

    rir = np.mean(actual_moves / implied_moves)
    stability = np.std(actual_moves / implied_moves)
    directional_bias = abs(np.mean(signed_moves)) / np.mean(actual_moves)

    # ---------------------------
    # Classification
    # ---------------------------

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


def confidence_score(hist, expected_move):
    volatility_penalty = expected_move / (hist["std"] + 1e-6)
    score = 1 - min(1, volatility_penalty / 3)
    return round(score * 100, 2)


def blended_expected_move(hist, iv_move):
    return (
        0.45 * hist["avg"] +
        0.25 * hist["p75"] +
        0.30 * iv_move
    )

def iv_event_move(price, iv, dte):
    return (price * iv * sqrt(dte / 365)) / price



# =============================
# Main Analysis
# =============================

def analyze_earnings_trade(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")

    price = get_stock_price(ticker)
    earnings_date = get_earnings_date(ticker)
    expiration = get_next_expirations(ticker, earnings_date)

    dte = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.today()).days

    chain = stock.option_chain(expiration)
    front_iv = chain.calls["impliedVolatility"].mean()

    # -------- Expected Move --------
    straddle = get_atm_straddle_move(ticker, expiration, price)
    iv_move = iv_expected_move(price, front_iv, dte)
    rv_move = realized_vol_move(price, hist)

    expected_move = (
        (0.5 * straddle if straddle else 0) +
        0.3 * iv_move +
        0.2 * rv_move
    )

    # -------- Historical Regime --------
    hist_behavior = historical_earnings_analysis(ticker)

    if not hist_behavior:
        trade_type = "SKIP"
        confidence = 0
    else:
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
        efficiency = expected_move / max(expected_move, 0.01)

        confidence = confidence_score(pop, efficiency)

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
        "Earnings Regime": hist_behavior["Regime"] if hist_behavior else "UNKNOWN"
    }

def get_sp500_tickers():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "sp500_tickers.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    return data["tickers"]

def scan_sp500_earnings():
    results = []
    tickers = get_sp500_tickers()

    for t in tickers:
        try:
            r = analyze_earnings_trade(t)
            if r and r["Confidence"] > 60:
                if r["Earnings"] <= (datetime.now().date() + timedelta(days=SCAN_DAYS)):
                    results.append(r)
        except:
            continue

    return pd.DataFrame(results).sort_values("Confidence Score", ascending=False)

def plot_histogram(hist, expected):
    plt.hist(hist["raw"], bins=10, alpha=0.7)
    plt.axvline(expected, color="red", label="Expected Move")
    plt.legend()
    plt.title("Historical Earnings Moves vs Model")
    plt.show()


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
