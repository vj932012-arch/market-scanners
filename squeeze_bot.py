import os
import datetime
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests

# --- SECRETS LOADED FROM GITHUB ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TICKERS = ["SPY", "QQQ", "GOOGL", "NVDA", "AMZN"] # Your mega-cap tech watchlist

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

def check_ttm_squeeze(ticker: str):
    """Pulls 5-minute data to detect active TTM Squeeze breakouts."""
    # Pulling 10 days of 5-minute data ensures enough warmup for pandas_ta to calculate the indicator accurately
    df = yf.download(ticker, period="10d", interval="5m", progress=False)
    if df.empty: return None
    
    # Flatten yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # 1. Calculate TTM Squeeze
    squeeze_df = df.ta.squeeze(lazybear=False, detailed=True)
    if squeeze_df is None: return None
    df = pd.concat([df, squeeze_df], axis=1)
    
    # Standardize column names
    sqz_on_col = [c for c in df.columns if "SQZ_ON" in c][0]
    sqz_off_col = [c for c in df.columns if "SQZ_OFF" in c][0]
    hist_col = [c for c in df.columns if "SQZ" in c and "ON" not in c and "OFF" not in c and "NO" not in c][0]
    
    df["SQZ_ON"] = df[sqz_on_col]
    df["SQZ_OFF"] = df[sqz_off_col]
    df["HISTOGRAM"] = df[hist_col]
    
    # 2. Calculate ADX for trend strength validation
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        df["ADX_14"] = 0.0

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["close"]
    
    # TRIGGER 1: Squeeze Firing (Red Dot transitions to Green Dot)
    squeeze_firing = (latest["SQZ_OFF"] == 1) and (prev["SQZ_ON"] == 1)
    
    # TRIGGER 2: Momentum Direction (Light Blue vs Red histogram expansion)
    hist_light_blue = (latest["HISTOGRAM"] > 0) and (latest["HISTOGRAM"] > prev["HISTOGRAM"])
    hist_red = (latest["HISTOGRAM"] < 0) and (latest["HISTOGRAM"] < prev["HISTOGRAM"])
    
    # TRIGGER 3: ADX Trend Strength Confirmation
    strong_trend = latest.get("ADX_14", 0) >= 25.0
    
    if squeeze_firing and hist_light_blue and strong_trend:
        return f"🟢 **{ticker} SQUEEZE FIRED: CALL SPREAD**\nPrice: ${price:.2f} | Momentum expanding UPWARD (Light Blue) with ADX >= 25."
    elif squeeze_firing and hist_red and strong_trend:
        return f"🔴 **{ticker} SQUEEZE FIRED: PUT SPREAD**\nPrice: ${price:.2f} | Momentum expanding DOWNWARD (Red) with ADX >= 25."
        
    return None

def run_squeeze_scan():
    """Runs the analysis on all tickers and only sends an alert if a squeeze is detected."""
    print(f"Starting Intraday TTM Squeeze Scan for {len(TICKERS)} mega-cap tickers...")
    messages = [f"💥 **Intraday TTM Squeeze Scan** ({datetime.datetime.now().strftime('%b %d, %H:%M ET')})\n"]
    triggers = 0
    
    for ticker in TICKERS:
        signal = check_ttm_squeeze(ticker)
        if signal:
            messages.append(signal)
            triggers += 1
            
    # Silent Mode: Only send a Telegram message if an actionable squeeze fired
    if triggers > 0:
        final_message = "\n\n".join(messages)
        print(final_message)
        send_telegram_message(final_message)
    else:
        print("No active squeeze breakouts right now. Remaining silent.")

if __name__ == "__main__":
    run_squeeze_scan()
