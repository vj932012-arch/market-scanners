import pytz
import os
import time
import datetime
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

# --- SECRETS LOADED FROM GITHUB ACTIONS ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

# --- LOAD CENTRAL CONFIGURATION ---
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)

config = load_config()["intraday_0dte"]
TICKERS = config["tickers"]

def send_telegram(msg: str):
    """Sends the alert directly to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing Telegram credentials.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def fetch_polygon_intraday(ticker_symbol: str) -> pd.DataFrame:
    """Fetches intraday 5-minute bars from Polygon.io and computes session VWAP."""
    if not POLYGON_API_KEY:
        print("Missing POLYGON_API_KEY environment variable.")
        return pd.DataFrame()

    # Request the last 7 calendar days to ensure at least 5 full trading days
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=7)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker_symbol}/range/5/minute/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
    except Exception as e:
        print(f"Network error fetching {ticker_symbol}: {e}")
        return pd.DataFrame()

    if "results" not in data or not data["results"]:
        print(f"No bar data returned for {ticker_symbol}.")
        return pd.DataFrame()

    # Convert JSON response to DataFrame and normalize columns
    df = pd.DataFrame(data["results"])
    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "timestamp"
    })

    # Convert Unix millisecond timestamps to localized US/Eastern market time
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_convert("America/New_York")

    # Filter strictly for regular market hours (9:30 AM to 4:00 PM ET)
    df = df.between_time("09:30", "16:00").copy()

    # Compute Day-Anchored VWAP
    df["date"] = df.index.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    df["cum_vp"] = (typical_price * df["volume"]).groupby(df["date"]).cumsum()
    df["cum_vol"] = (df["volume"]).groupby(df["date"]).cumsum()
    df["vwap"] = df["cum_vp"] / df["cum_vol"]
    df.drop(columns=["date", "cum_vp", "cum_vol"], inplace=True)

    return df

def evaluate_intraday_setup(ticker: str):
    """Calculates EMAs, VWAP displacement, RVOL, and ADX/DMI for momentum entries."""
    df = fetch_polygon_intraday(ticker)
    if df.empty or len(df) < 28:
        return None

    # 1. ATR (14) for dynamic normalization
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(window=14).mean()

    # 2. EMAs and Normalized Distances
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_spread_norm"] = (df["ema_9"] - df["ema_21"]) / df["atr"]
    df["vwap_dist_norm"] = (df["close"] - df["vwap"]) / df["atr"]

    # 3. Relative Volume (RVOL) against 20-period baseline
    df["vol_ma"] = df["volume"].rolling(window=20).mean()
    df["rvol"] = df["volume"] / df["vol_ma"]

    # 4. ADX and DMI (14) Trend Strength
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None and not adx_df.empty:
        df = pd.concat([df, adx_df], axis=1)
    else:
        df["ADX_14"], df["DMP_14"], df["DMN_14"] = 0.0, 0.0, 0.0

    latest = df.iloc[-1]
    price = latest["close"]

    # Pull dynamic parameters from config
    spread_width = config["spread_width"]
    rvol_thresh = config["rvol_threshold"]
    adx_thresh = config["adx_threshold"]

    # Call Logic
    call_spread = (
        latest["ema_spread_norm"] > 0.15 and
        0.20 <= latest["vwap_dist_norm"] <= 1.10 and
        latest["rvol"] >= rvol_thresh and
        latest["close"] > latest["open"] and
        latest.get("ADX_14", 0) >= adx_thresh and
        latest.get("DMP_14", 0) > latest.get("DMN_14", 0)
    )

    # Put Logic
    put_spread = (
        latest["ema_spread_norm"] < -0.15 and
        -1.10 <= latest["vwap_dist_norm"] <= -0.20 and
        latest["rvol"] >= rvol_thresh and
        latest["close"] < latest["open"] and
        latest.get("ADX_14", 0) >= adx_thresh and
        latest.get("DMN_14", 0) > latest.get("DMP_14", 0)
    )
    if call_spread:
        long_strike = np.floor(price)
        short_strike = long_strike + spread_width
        return (
            f"🟢 **{ticker} 0DTE CALL SPREAD**\n"
            f"Price: ${price:.2f} | Strong Bullish Momentum\n"
            f"• RVOL: {latest['rvol']:.2f}x | ADX: {latest.get('ADX_14', 0):.1f}\n"
            f"• Strikes: Buy ${long_strike:.0f}C / Sell ${short_strike:.0f}C"
        )
    elif put_spread:
        long_strike = np.ceil(price)
        short_strike = long_strike - spread_width
        return (
            f"🔴 **{ticker} 0DTE PUT SPREAD**\n"
            f"Price: ${price:.2f} | Strong Bearish Momentum\n"
            f"• RVOL: {latest['rvol']:.2f}x | ADX: {latest.get('ADX_14', 0):.1f}\n"
            f"• Strikes: Buy ${long_strike:.0f}P / Sell ${short_strike:.0f}P"
        )

    return None

def run_intraday_scan():
    print("Running 0DTE Intraday Scan via Polygon.io...")
    
    # Fix the UTC time bug so it prints actual Eastern Time
    now_et = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M ET')
    messages = [f"⚡ **0DTE Intraday Scanner** ({now_et})\n"]
    triggers = 0

    for ticker in TICKERS:
        signal = evaluate_intraday_setup(ticker)
        if signal:
            messages.append(signal)
            triggers += 1
        time.sleep(1)  # Buffer between API requests

    # Remove the silent gate and append a heartbeat message if no trades triggered
    if triggers == 0:
        messages.append("⚪ No active 0DTE breakout signals right now.")

    final_message = "\n\n".join(messages)
    print(final_message)
    send_telegram(final_message)


if __name__ == "__main__":
    run_intraday_scan()
