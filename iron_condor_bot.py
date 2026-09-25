import os
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import requests
import pytz

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TICKERS = ["QQQ", "MSFT"]

# Strategy Parameters
VOL_MIN = 0.20
VOL_MAX = 0.85
DELTA_MULT = 1.0
WING_WIDTH = 5.0

def send_telegram_alert(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing Telegram credentials.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")
        return False

def get_daily_signals(ticker: str):
    df = yf.Ticker(ticker).history(period="180d", interval="1d")
    if df.empty: return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna(subset=["close"])

    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None: df = pd.concat([df, bb], axis=1)

    df["volatility"] = df["close"].pct_change().rolling(window=20).std() * np.sqrt(252)
    vol_min_hist = df["volatility"].rolling(window=90).min()
    vol_max_hist = df["volatility"].rolling(window=90).max()
    df["iv_rank"] = (df["volatility"] - vol_min_hist) / (vol_max_hist - vol_min_hist)

    latest = df.iloc[-1]
    
    # Calculate Score
    score = 0
    if pd.notna(latest["iv_rank"]) and VOL_MIN <= latest["iv_rank"] <= VOL_MAX: score += 30
    if pd.notna(latest["rsi"]) and 40 <= latest["rsi"] <= 60: score += 25

    bbu_cols = [c for c in df.columns if "BBU" in c]
    bbl_cols = [c for c in df.columns if "BBL" in c]
    if bbu_cols and bbl_cols and pd.notna(latest[bbu_cols[0]]):
        bb_mid = (latest[bbu_cols[0]] + latest[bbl_cols[0]]) / 2
        bb_range = latest[bbu_cols[0]] - latest[bbl_cols[0]]
        if bb_range > 0 and abs(latest["close"] - bb_mid) < (bb_range * 0.25):
            score += 25

    atr_sma = df["atr"].iloc[-11:-1].mean()
    if atr_sma > 0 and 0.8 <= (latest["atr"] / atr_sma) <= 1.2: score += 20

    if score >= 70:
        expected_7_day_move = latest["atr"] * np.sqrt(5)
        short_distance = expected_7_day_move * DELTA_MULT
        
        short_c = np.ceil(latest["close"] + short_distance)
        short_p = np.floor(latest["close"] - short_distance)
        long_c = short_c + WING_WIDTH
        long_p = short_p - WING_WIDTH
        
        return (
            f"🦅 <b>{ticker} 7-DTE IRON CONDOR</b>\n"
            f"Price: ${latest['close']:.2f} | Score: {score:.0f}/100\n\n"
            f"<b>Calls:</b> Sell ${short_c:.0f}C / Buy ${long_c:.0f}C\n"
            f"<b>Puts:</b> Sell ${short_p:.0f}P / Buy ${long_p:.0f}P\n"
            f"<b>Max Profit Zone:</b> ${short_p:.0f} to ${short_c:.0f}"
        )
    return f"⚪ <b>{ticker}</b>: Monitoring (Score: {score:.0f}/100)"

def run_daily_scan():
    now_et = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%b %d, %Y - %H:%M ET')
    messages = [f"📊 <b>Daily Iron Condor Scan</b> ({now_et})\n"]
    
    for ticker in TICKERS:
        signal = get_daily_signals(ticker)
        if signal:
            messages.append(signal)

    final_message = "\n\n".join(messages)
    print(final_message)
    send_telegram_alert(final_message)

if __name__ == "__main__":
    run_daily_scan()
