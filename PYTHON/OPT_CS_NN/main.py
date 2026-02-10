import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime
from scipy.stats import norm

# =============================
# Core Helpers
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
    raise ValueError("Earnings date unavailable")


def get_next_expirations(ticker, earnings_date):
    expirations = yf.Ticker(ticker).options
    for i, exp in enumerate(expirations):
        exp_date = pd.to_datetime(exp)
        if exp_date > earnings_date:
            return expirations[i], expirations[i + 1]
    raise ValueError("No expirations found")


# =============================
# Volatility & Move Models
# =============================

def get_atm_straddle_move(ticker, expiration, price):
    chain = yf.Ticker(ticker).option_chain(expiration)
    calls, puts = chain.calls, chain.puts

    atm_strike = calls.iloc[(calls["strike"] - price).abs().argsort()[:1]]["strike"].values[0]

    call = calls[calls["strike"] == atm_strike].iloc[0]
    put = puts[puts["strike"] == atm_strike].iloc[0]

    if call["bid"] == 0 or put["bid"] == 0:
        return None

    call_mid = (call["bid"] + call["ask"]) / 2
    put_mid = (put["bid"] + put["ask"]) / 2

    return call_mid + put_mid


def iv_expected_move(price, iv, dte):
    return price * iv * sqrt(dte / 365)


def historical_earnings_move(ticker, lookback=6):
    stock = yf.Ticker(ticker)
    prices = stock.history(period="3y")
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
    vol = returns[-days:].std()
    return price * vol


# =============================
# IV Crush Estimation
# =============================

def estimate_iv_crush(front_iv, back_iv):
    """
    Earnings IV crush estimate using term structure
    """
    if front_iv <= 0 or back_iv <= 0:
        return None

    crush_pct = (front_iv - back_iv) / front_iv
    return max(min(crush_pct, 0.85), 0.25)  # clamp realistic bounds


# =============================
# Probability / EV Model
# =============================

def probability_price_in_range(price, lower, upper, expected_move):
    sigma = expected_move / price
    z_low = (lower - price) / (price * sigma)
    z_high = (upper - price) / (price * sigma)
    return norm.cdf(z_high) - norm.cdf(z_low)


# =============================
# Main Analysis
# =============================

def analyze_calendar(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")

    price = get_stock_price(ticker)
    earnings_date = get_earnings_date(ticker)
    front_exp, back_exp = get_next_expirations(ticker, earnings_date)

    front_dte = (datetime.strptime(front_exp, "%Y-%m-%d") - datetime.today()).days

    front_chain = stock.option_chain(front_exp)
    back_chain = stock.option_chain(back_exp)

    front_iv = front_chain.calls["impliedVolatility"].mean()
    back_iv = back_chain.calls["impliedVolatility"].mean()

    # -------- Expected Move Model --------
    straddle_move = get_atm_straddle_move(ticker, front_exp, price)
    iv_move = iv_expected_move(price, front_iv, front_dte)
    hist_move_pct = historical_earnings_move(ticker)
    rv_move = realized_vol_move(price, hist)

    components = []

    if straddle_move:
        components.append((0.45, straddle_move))
    components.append((0.25, iv_move))
    if hist_move_pct:
        components.append((0.20, hist_move_pct * price))
    components.append((0.10, rv_move))

    expected_move = sum(w * v for w, v in components)

    # -------- Iron Condor Range --------
    buffer = 1.25
    lower = price - expected_move * buffer
    upper = price + expected_move * buffer

    # -------- Probability / EV --------
    pop = probability_price_in_range(price, lower, upper, expected_move)

    # -------- IV Crush --------
    iv_crush = estimate_iv_crush(front_iv, back_iv)

    return {
        "Ticker": ticker,
        "Price": round(price, 2),
        "Earnings Date": earnings_date.date(),
        "Front Exp": front_exp,
        "Front IV (%)": round(front_iv * 100, 2),
        "Back IV (%)": round(back_iv * 100, 2),
        "Estimated IV Crush (%)": round(iv_crush * 100, 2) if iv_crush else "N/A",
        "ATM Straddle ($)": round(straddle_move, 2) if straddle_move else "N/A",
        "Expected Move ($)": round(expected_move, 2),
        "Condor Lower": round(lower, 2),
        "Condor Upper": round(upper, 2),
        "Probability of Success (%)": round(pop * 100, 2)
    }


# =============================
# Run
# =============================

if __name__ == "__main__":
    ticker = input("Enter ticker: ").upper()
    result = analyze_calendar(ticker)

    print("\n--- Earnings Iron Condor Analysis ---")
    for k, v in result.items():
        print(f"{k}: {v}")
