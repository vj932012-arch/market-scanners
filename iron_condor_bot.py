"""
Iron Condor Bot with Telegram Alerts (Polygon.io Integration)
Monitors SPY for ideal setups and sends alerts via Telegram
"""

import os
import requests
import sys
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
from datetime import datetime, timedelta
from typing import Dict, Tuple

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TICKER = "SPY"
LOOKBACK_DAYS = 90  # Ensure 50+ trading days for SMA_50
PERIOD_SHORT = 14
PERIOD_LONG = 50

# Strike & Delta Configuration
DELTA_ATR_MULTIPLIER = 2.0  # Target Delta: ~16 Delta / 1 SD
WING_WIDTH = 2.0            # Defined-risk spread width in dollars

# Setup Parameters
IC_SETUP = {
    "volatility_min": 0.20,     # Re-adjusted to 20% floor
    "volatility_max": 0.85,
    "min_score": 70.0,             
}

# Secrets loaded from GitHub Actions
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ---------------------------------------------------------
# Telegram Functions
# ---------------------------------------------------------
def send_telegram_alert(message: str, parse_mode: str = "HTML") -> bool:
    """Send alert message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("✗ Missing Telegram credentials in environment variables.")
        return False

    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": parse_mode,
        }
        response = requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            print("✓ Telegram message sent successfully.")
            return True
        else:
            print(f"✗ Telegram error ({response.status_code}): {response.text}")
            return False

    except Exception as e:
        print(f"✗ Failed to send Telegram alert: {e}")
        return False

# ---------------------------------------------------------
# Data Analysis Functions
# ---------------------------------------------------------
def fetch_polygon_data(ticker_symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetches daily bars from Polygon.io."""
    if not POLYGON_API_KEY:
        print("Missing POLYGON_API_KEY environment variable.")
        return pd.DataFrame()

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker_symbol}/range/15/minute/{start_date}/{end_date}"
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
    
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df_calc = df.copy()

    # RSI
    df_calc['RSI'] = ta.rsi(df_calc['close'], length=PERIOD_SHORT)

    # Bollinger Bands
    bb = ta.bbands(df_calc['close'], length=20, std=2)
    if bb is not None:
        df_calc = pd.concat([df_calc, bb], axis=1)

    # ATR
    df_calc['ATR'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=14)

    # SMA
    df_calc['SMA_14'] = ta.sma(df_calc['close'], length=PERIOD_SHORT)
    df_calc['SMA_50'] = ta.sma(df_calc['close'], length=PERIOD_LONG)

    # Volatility (IV Rank Proxy)
    df_calc['Volatility'] = df_calc['close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    return df_calc

def identify_ic_setup(df: pd.DataFrame) -> Tuple[int, float, Dict]:
    """
    Identify neutral Iron Condor setup.
    signal: 2 = Neutral Iron Condor, 0 = No Setup
    """
    details = {
        "reasons": [],
        "rsi": None,
        "iv_rank": None,
        "atr": None,
        "price": None,
    }

    df = df.copy()
    signal = 0
    score = 0.0

    latest = df.iloc[-1]
    close_price = latest['close']
    rsi = latest.get('RSI', np.nan)
    atr = latest.get('ATR', 0)
    volatility = latest.get('Volatility', 0)

    details['price'] = close_price
    details['rsi'] = rsi
    details['atr'] = atr

    # IV Rank Proxy
    vol_min = df['Volatility'].min()
    vol_max = df['Volatility'].max()
    iv_rank = (volatility - vol_min) / (vol_max - vol_min) if vol_max > vol_min else 0.5
    details['iv_rank'] = iv_rank

    # 1. Volatility Check (30 points)
    if 0.20 <= iv_rank <= IC_SETUP['volatility_max']:
        score += 30
        details['reasons'].append(f"✓ Volatility elevated for premium collection (IVR: {iv_rank:.1%})")
    else:
        details['reasons'].append(f"✗ Volatility too low/high (IVR: {iv_rank:.1%})")

    # 2. Bollinger Bands Check (25 points)
    bb_upper_col = [c for c in df.columns if 'BBU' in c]
    bb_lower_col = [c for c in df.columns if 'BBL' in c]

    if bb_upper_col and bb_lower_col:
        bb_upper = df[bb_upper_col[0]].iloc[-1]
        bb_lower = df[bb_lower_col[0]].iloc[-1]
        bb_mid = (bb_upper + bb_lower) / 2
        bb_range = bb_upper - bb_lower
        distance_to_mid = abs(close_price - bb_mid)

        if bb_range > 0 and distance_to_mid < bb_range * 0.25:
            score += 25
            details['reasons'].append("✓ Price near middle of Bollinger Bands (Range-bound)")
        else:
            details['reasons'].append("✗ Price too close to BB edges (Trending)")

    # 3. RSI Neutrality Check (25 points)
    if not np.isnan(rsi):
        if 40 <= rsi <= 60:
            score += 25
            details['reasons'].append(f"✓ RSI is neutral ({rsi:.1f})")
        else:
            details['reasons'].append(f"✗ RSI indicates trend/momentum ({rsi:.1f})")

    # 4. ATR Normalization Check (20 points)
    atr_sma = df['ATR'].rolling(window=10).mean().iloc[-1]
    if atr_sma > 0:
        atr_ratio = atr / atr_sma
        if 0.8 <= atr_ratio <= 1.2:
            score += 20
            details['reasons'].append("✓ Volatility (ATR) is stable")
        else:
            details['reasons'].append("✗ Volatility (ATR) is expanding/contracting too fast")

    if score >= IC_SETUP.get('min_score', 70.0):
        signal = 2 

    return signal, score, details

def calculate_ic_levels(close_price: float, atr: float, signal: int = 0) -> Dict:
    """Calculate tradeable, rounded Iron Condor strike levels."""
    short_distance = atr * DELTA_ATR_MULTIPLIER

    # Asymmetric skewing based on trend signal
    if signal == 1:
        call_buffer = short_distance * 1.2
        put_buffer = short_distance * 0.8
    elif signal == -1:
        call_buffer = short_distance * 0.8
        put_buffer = short_distance * 1.2
    else:
        call_buffer = short_distance
        put_buffer = short_distance

    short_call = round(close_price + call_buffer)
    long_call = round(short_call + WING_WIDTH)

    short_put = round(close_price - put_buffer)
    long_put = round(short_put - WING_WIDTH)

    return {
        "short_call": short_call,
        "long_call": long_call,
        "short_put": short_put,
        "long_put": long_put,
        "wing_width": WING_WIDTH,
        "max_profit_range_low": short_put,
        "max_profit_range_high": short_call,
    }

def format_alert_message(signal: int, score: float, details: Dict, levels: Dict) -> str:
    """Format active alert message for Telegram"""
    now_et = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S ET')
    
    if signal == 2:
        signal_type = "🦅 NEUTRAL IRON CONDOR"
    elif signal == 1:
        signal_type = "🟢 CALL SPREAD"
    elif signal == -1:
        signal_type = "🔴 PUT SPREAD"
    else:
        signal_type = "⚪ UNKNOWN SIGNAL"

    message = f"""<b>{signal_type} SIGNAL - Setup Score: {score:.0f}/100</b>

<b>Current Price:</b> ${details['price']:.2f}
<b>RSI:</b> {details['rsi']:.1f}
<b>ATR:</b> ${details['atr']:.2f}
<b>IV Rank:</b> {details['iv_rank']:.1%}

<b>Setup Conditions:</b>"""

    for reason in details['reasons']:
        message += f"\n{reason}"

    if levels:
        message += f"""

<b>Suggested Strikes (16 Delta, ${levels['wing_width']}-Wide Wings):</b>
Short Call: ${levels['short_call']:.2f} / Long Call: ${levels['long_call']:.2f}
Short Put: ${levels['short_put']:.2f} / Long Put: ${levels['long_put']:.2f}

<b>Max Profit Zone:</b> ${levels['max_profit_range_low']:.2f} - ${levels['max_profit_range_high']:.2f}
<b>Defined Collateral:</b> ${levels['wing_width'] * 100:.2f}"""

    message += f"\n\n<b>Timestamp:</b> {now_et}"
    return message

def format_heartbeat_message(score: float, details: Dict) -> str:
    """Format heartbeat status message when no actionable setup is detected"""
    now_et = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S ET')
    
    rsi_val = details.get('rsi')
    rsi_str = f"{rsi_val:.1f}" if rsi_val is not None and not np.isnan(rsi_val) else "N/A"

    message = f"""<b>🦅 IRON CONDOR SCANNER (Heartbeat)</b>

<b>Status:</b> ⚪ Monitoring (No setup triggered)
<b>Score:</b> {score:.0f}/100 (Threshold: {IC_SETUP['min_score']:.0f})

<b>Current Price:</b> ${details['price']:.2f}
<b>RSI:</b> {rsi_str}
<b>ATR:</b> ${details['atr']:.2f}
<b>IV Rank:</b> {details['iv_rank']:.1%}

<b>Conditions Observed:</b>"""

    for reason in details['reasons']:
        message += f"\n{reason}"

    message += f"\n\n<b>Timestamp:</b> {now_et}"
    return message

# ---------------------------------------------------------
# Single-Run Execution for GitHub Actions
# ---------------------------------------------------------
def run_analysis_once():
    """Run analysis once and dispatch to Telegram unconditionally"""
    print(f"Running single analysis on {TICKER} via Polygon.io...")

    df = fetch_polygon_data(TICKER)
    if df is None or len(df) < 50:
        print("✗ Insufficient data fetched from Polygon.io.")
        now_et = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S ET')
        send_telegram_alert(f"⚠️ <b>Iron Condor Bot Heartbeat</b>\n\nCould not fetch sufficient data for {TICKER} via Polygon.\n\n<b>Timestamp:</b> {now_et}")
        return

    df = calculate_indicators(df)
    signal, score, details = identify_ic_setup(df)

    print(f"Price: ${details['price']:.2f} | RSI: {details['rsi']:.1f} | Score: {score:.0f}/100")

    if signal != 0 and score >= IC_SETUP['min_score']:
        levels = calculate_ic_levels(details['price'], details['atr'], signal)
        message = format_alert_message(signal, score, details, levels)
        print("📬 Actionable setup detected! Sending alert to Telegram...")
        send_telegram_alert(message)
    else:
        message = format_heartbeat_message(score, details)
        print("⚪ Score below threshold or no directional signal. Sending heartbeat to Telegram...")
        send_telegram_alert(message)

def test_telegram_connection():
    """Test Telegram bot connection"""
    print("Testing Telegram connection...")
    now_et = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S ET')
    message = f"🤖 <b>Iron Condor Bot Test</b>\n✓ GitHub Actions connection successful!\n\n<b>Timestamp:</b> {now_et}"
    send_telegram_alert(message)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_telegram_connection()
        else:
            print("Usage: python iron_condor_bot.py [test]")
    else:
        run_analysis_once()
