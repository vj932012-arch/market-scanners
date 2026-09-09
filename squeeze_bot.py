import os
import time
import datetime
import json
import pandas as pd
import pandas_ta as ta
import requests

# --- SECRETS LOADED FROM GITHUB ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

# --- LOAD CENTRAL CONFIGURATION ---
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)

config = load_config()["squeeze"]
TICKERS = config["tickers"]

def send_telegram_message(message: str):
    """Sends a push notification directly to your phone via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing Telegram credentials. Ensure secrets are set in GitHub.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("Telegram message sent successfully.")
    except Exception as e:
        print(f"Failed to send message: {e}")

def fetch_polygon_intraday(ticker_symbol: str) -> pd.DataFrame:
    """Fetches intraday 5-minute bars from Polygon.io."""
    if not POLYGON_API_KEY:
        print("Missing POLYGON_API_KEY environment variable.")
        return pd.DataFrame()

    # Request the last 14 calendar days to ensure enough warmup for pandas_ta indicators
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=14)

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

    # Convert JSON to DataFrame and normalize column names
    df = pd.DataFrame(data["results"])
    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "timestamp"
    })

    # Convert Unix ms timestamps to US/Eastern market time
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_convert("America/New_York")

    # Filter strictly for regular trading hours (9:30 AM to 4:00 PM ET)
    df = df.between_time("09:30", "16:00").copy()
    
    return df

def check_ttm_squeeze(ticker: str):
    """Pulls 5-minute data to detect active TTM Squeeze breakouts."""
    df = fetch_polygon_intraday(ticker)
    if df.empty or len(df) < 28: return None
    
    # Pull dynamic parameters from config
    adx_thresh = config["adx_threshold"]
    
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
    strong_trend = latest.get("ADX_14", 0) >= adx_thresh
    
    if squeeze_firing and hist_light_blue and strong_trend:
        return f"🟢 **{ticker} SQUEEZE FIRED: CALL SPREAD**\nPrice: ${price:.2f} | Momentum expanding UPWARD (Light Blue) with ADX >= {adx_thresh}."
    elif squeeze_firing and hist_red and strong_trend:
        return f"🔴 **{ticker} SQUEEZE FIRED: PUT SPREAD**\nPrice: ${price:.2f} | Momentum expanding DOWNWARD (Red) with ADX >= {adx_thresh}."
        
    return None

def run_squeeze_scan():
    """Runs the analysis on all tickers and only sends an alert if a squeeze is detected."""
    print(f"Starting Intraday TTM Squeeze Scan for {len(TICKERS)} mega-cap tickers via Polygon.io...")
    messages = [f"💥 **Intraday TTM Squeeze Scan** ({datetime.datetime.now().strftime('%b %d, %H:%M ET')})\n"]
    triggers = 0
    
    for ticker in TICKERS:
        signal = check_ttm_squeeze(ticker)
        if signal:
            messages.append(signal)
            triggers += 1
            
        # Polygon's free tier is limited to 5 API calls per minute
        # Sleeping for 12 seconds perfectly paces the tickers across a full minute
        time.sleep(12)
            
    # Silent Mode: Only send a Telegram message if an actionable squeeze fired
    if triggers > 0:
        final_message = "\n\n".join(messages)
        print(final_message)
        send_telegram_message(final_message)
    else:
        print("No active squeeze breakouts right now. Remaining silent.")

if __name__ == "__main__":
    run_squeeze_scan()
