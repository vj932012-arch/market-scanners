import os
import datetime
import json
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests

# --- SECRETS LOADED FROM GITHUB ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- LOAD CENTRAL CONFIGURATION ---
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)

config = load_config()["swing"]
TICKERS = config["tickers"]

def send_telegram_message(message: str):
    """Sends a push notification directly to your phone via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing Telegram credentials. Ensure secrets are set in GitHub.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Telegram message sent successfully.")
    except Exception as e:
        print(f"Failed to send message: {e}")

def get_daily_signals(ticker: str):
    """Fetches daily data and calculates swing thresholds."""
    df = yf.Ticker(ticker).history(period="1y", interval="1d")
    if df.empty: return None
    
    # Flatten yfinance MultiIndex if it exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    # Moving Averages & Volatility
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    # ADX / DMI for Trend Strength
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        df["ADX_14"], df["DMP_14"], df["DMN_14"] = 0.0, 0.0, 0.0

    latest = df.iloc[-1]
    price = latest["close"]
    atr = latest["atr"]
    
    # Call/Put Triggers
    call_trigger = (
        price > latest["ema_50"] and latest["ema_20"] > latest["ema_50"] and 
        45 <= latest["rsi"] <= 70 and latest["ADX_14"] >= 20 and latest["DMP_14"] > latest["DMN_14"]
    )
    
    put_trigger = (
        price < latest["ema_50"] and latest["ema_20"] < latest["ema_50"] and 
        30 <= latest["rsi"] <= 55 and latest["ADX_14"] >= 20 and latest["DMN_14"] > latest["DMP_14"]
    )
    
    # Dynamic Strike Calculations based on ATR
    spread_width = max(round(atr), 1)
    
    if call_trigger:
        long_strike = round(price)
        short_strike = long_strike + spread_width
        return f"🟢 **{ticker} CALL SPREAD**\nPrice: ${price:.2f} | Buy ${long_strike}C / Sell ${short_strike}C"
    elif put_trigger:
        long_strike = round(price)
        short_strike = long_strike - spread_width
        return f"🔴 **{ticker} PUT SPREAD**\nPrice: ${price:.2f} | Buy ${long_strike}P / Sell ${short_strike}P"
        
    return None

def run_daily_swing_scan():
    """Runs the analysis on all tickers and sends the formatted alert."""
    print(f"Starting Multi-Day Swing Scan for {len(TICKERS)} tickers...")
    messages = [f"🔭 **Daily Swing Trade Scan** ({datetime.datetime.now().strftime('%b %d, %Y')})\n"]
    triggers = 0
    
    for ticker in TICKERS:
        signal = get_daily_signals(ticker)
        if signal:
            messages.append(signal)
            triggers += 1
            
    if triggers == 0:
        messages.append("⚪ No clear swing setups triggered today.")
        
    final_message = "\n\n".join(messages)
    print(final_message)
    send_telegram_message(final_message)

# --- SINGLE EXECUTION BLOCK ---
# Because GitHub Actions handles the cron scheduling via the YAML file, 
# this script just executes exactly once and shuts down immediately.
if __name__ == "__main__":
    run_daily_swing_scan()
