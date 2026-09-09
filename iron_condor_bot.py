"""
Iron Condor Bot with Telegram Alerts
Monitors SPY for ideal iron condor setups and sends alerts via Telegram
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import json
import os
import requests
import time
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TICKER = "SPY"
LOOKBACK_DAYS = 30
PERIOD_SHORT = 14
PERIOD_LONG = 50
CHECK_INTERVAL_MINUTES = 15  # Check every 15 minutes

# Iron Condor Parameters
IC_SETUP = {
    "rsi_sell_threshold": 70,      # Upper RSI for Put Spread
    "rsi_buy_threshold": 30,       # Lower RSI for Call Spread
    "atr_multiplier": 1.5,
    "bb_threshold": 2.0,
    "volatility_min": 0.01,
    "volatility_max": 0.85,
    "min_score": 70.0,             # Minimum score to trigger alert
}

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# State tracking (to avoid duplicate alerts)
LAST_ALERT = {
    "call_spread": None,
    "put_spread": None,
    "timestamp": None,
}
ALERT_COOLDOWN_MINUTES = 30  # Don't send same alert more often than this

# ---------------------------------------------------------
# Telegram Functions
# ---------------------------------------------------------
def send_telegram_alert(message: str, parse_mode: str = "HTML") -> bool:
    """
    Send alert message to Telegram
    
    Args:
        message: Alert message (supports HTML formatting)
        parse_mode: 'HTML' or 'Markdown'
    
    Returns:
        bool: True if sent successfully
    """
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
            print(f"✓ Telegram alert sent: {message[:100]}...")
            return True
        else:
            print(f"✗ Telegram error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to send Telegram alert: {e}")
        return False

def send_telegram_chart(chart_path: str) -> bool:
    """
    Send chart image to Telegram
    
    Args:
        chart_path: Path to local chart image
    
    Returns:
        bool: True if sent successfully
    """
    try:
        with open(chart_path, 'rb') as f:
            files = {'photo': f}
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
            }
            response = requests.post(
                f"{TELEGRAM_API_URL}/sendPhoto",
                data=payload,
                files=files,
                timeout=10
            )
        
        if response.status_code == 200:
            print(f"✓ Chart sent to Telegram")
            return True
        else:
            print(f"✗ Chart send error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to send chart: {e}")
        return False

# ---------------------------------------------------------
# Data Analysis Functions
# ---------------------------------------------------------
def fetch_spy_data(ticker=TICKER, days=LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch historical SPY data"""
    try:
        df = yf.download(ticker, period=f"{days}d", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        return df
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return None

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
    
    # Volatility
    df_calc['Volatility'] = df_calc['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    return df_calc

def identify_ic_setup(df: pd.DataFrame) -> Tuple[int, float, Dict]:
    """
    Identify iron condor setup
    
    Returns:
        signal: 1=Call Spread, -1=Put Spread, 0=No Setup
        score: Setup quality score (0-100)
        details: Dict with analysis details
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
    
    # IV Rank (normalized volatility)
    vol_min = df['Volatility'].min()
    vol_max = df['Volatility'].max()
    iv_rank = (volatility - vol_min) / (vol_max - vol_min) if vol_max > vol_min else 0.5
    details['iv_rank'] = iv_rank
    
    # Volatility check
    if IC_SETUP['volatility_min'] <= iv_rank <= IC_SETUP['volatility_max']:
        score += 30
        details['reasons'].append("✓ Volatility in ideal range")
    else:
        details['reasons'].append(f"✗ Volatility outside range")
    
    # Bollinger Bands position
    bb_upper_col = [c for c in df.columns if 'BBU' in c]
    bb_lower_col = [c for c in df.columns if 'BBL' in c]
    
    if bb_upper_col and bb_lower_col:
        bb_upper = df[bb_upper_col[0]].iloc[-1]
        bb_lower = df[bb_lower_col[0]].iloc[-1]
        bb_mid = (bb_upper + bb_lower) / 2
        
        distance_to_mid = abs(close_price - bb_mid)
        bb_range = bb_upper - bb_lower
        
        if bb_range > 0 and distance_to_mid < bb_range * 0.25:
            score += 25
            details['reasons'].append("✓ Price near middle bands")
    
    # RSI extremes
    if not np.isnan(rsi):
        if rsi > IC_SETUP['rsi_sell_threshold']:
            score += 25
            signal = -1
            details['reasons'].append(f"✓ High RSI {rsi:.1f} - Put Spread")
        elif rsi < IC_SETUP['rsi_buy_threshold']:
            score += 25
            signal = 1
            details['reasons'].append(f"✓ Low RSI {rsi:.1f} - Call Spread")
    
    # ATR normalization
    atr_sma = df['ATR'].rolling(window=10).mean().iloc[-1]
    if atr_sma > 0:
        atr_ratio = atr / atr_sma
        if 0.8 <= atr_ratio <= 1.2:
            score += 20
            details['reasons'].append("✓ ATR normalized")
    
    return signal, score, details

def calculate_ic_levels(close_price: float, atr: float, signal: int) -> Dict:
    """Calculate iron condor strike levels"""
    levels = {}
    strike_width = atr * IC_SETUP['atr_multiplier']
    
    if signal == 1:  # Call Spread
        levels['short_call'] = close_price + strike_width
        levels['long_call'] = close_price + (strike_width * 2)
        levels['short_put'] = close_price - (strike_width * 0.5)
        levels['long_put'] = close_price - (strike_width * 1.5)
        levels['max_profit_range_low'] = levels['long_put']
        levels['max_profit_range_high'] = levels['long_call']
    elif signal == -1:  # Put Spread
        levels['short_put'] = close_price - strike_width
        levels['long_put'] = close_price - (strike_width * 2)
        levels['short_call'] = close_price + (strike_width * 0.5)
        levels['long_call'] = close_price + (strike_width * 1.5)
        levels['max_profit_range_low'] = levels['long_put']
        levels['max_profit_range_high'] = levels['long_call']
    
    return levels

# ---------------------------------------------------------
# Alert Management
# ---------------------------------------------------------
def should_send_alert(signal: int, score: float) -> bool:
    """
    Check if alert should be sent (avoid duplicates)
    """
    if score < IC_SETUP['min_score']:
        return False
    
    now = datetime.now()
    signal_type = "call_spread" if signal == 1 else "put_spread"
    
    if LAST_ALERT['timestamp'] is None:
        return True
    
    # Check cooldown period
    time_since_last = (now - LAST_ALERT['timestamp']).total_seconds() / 60
    
    if signal_type == LAST_ALERT.get(signal_type):
        if time_since_last < ALERT_COOLDOWN_MINUTES:
            return False
    
    return True

def format_alert_message(signal: int, score: float, details: Dict, levels: Dict) -> str:
    """Format alert message for Telegram"""
    signal_type = "🟢 CALL SPREAD" if signal == 1 else "🔴 PUT SPREAD"
    
    message = f"""
<b>{signal_type} SIGNAL - Setup Score: {score:.0f}/100</b>

<b>Current Price:</b> ${details['price']:.2f}
<b>RSI:</b> {details['rsi']:.1f}
<b>ATR:</b> ${details['atr']:.2f}
<b>IV Rank:</b> {details['iv_rank']:.1%}

<b>Setup Conditions:</b>
"""
    
    for reason in details['reasons']:
        message += f"\n{reason}"
    
    if levels:
        message += f"""

<b>Suggested Strikes (ATM ±1.5 STD):</b>
Short Call: ${levels['short_call']:.2f}
Long Call: ${levels['long_call']:.2f}
Short Put: ${levels['short_put']:.2f}
Long Put: ${levels['long_put']:.2f}

<b>Max Profit Range:</b> ${levels['max_profit_range_low']:.2f} - ${levels['max_profit_range_high']:.2f}
"""
    
    message += f"\n<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    
    return message

# ---------------------------------------------------------
# Main Bot Loop
# ---------------------------------------------------------
def run_bot():
    """Main bot loop"""
    print(f"\n🤖 Iron Condor Bot Started")
    print(f"Monitoring {TICKER} every {CHECK_INTERVAL_MINUTES} minutes")
    print(f"Alert threshold: {IC_SETUP['min_score']}/100")
    print("-" * 60)
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{timestamp}] Check #{iteration}")
            
            # Fetch and analyze
            df = fetch_spy_data()
            if df is None or len(df) < 50:
                print("✗ Insufficient data")
                time.sleep(CHECK_INTERVAL_MINUTES * 60)
                continue
            
            df = calculate_indicators(df)
            signal, score, details = identify_ic_setup(df)
            
            print(f"  Price: ${details['price']:.2f} | RSI: {details['rsi']:.1f} | Score: {score:.0f}")
            
            # Check if alert needed
            if signal != 0 and should_send_alert(signal, score):
                levels = calculate_ic_levels(details['price'], details['atr'], signal)
                message = format_alert_message(signal, score, details, levels)
                
                print(f"  📬 Sending {('Call' if signal == 1 else 'Put')} Spread alert...")
                
                if send_telegram_alert(message):
                    LAST_ALERT[('call_spread' if signal == 1 else 'put_spread')] = score
                    LAST_ALERT['timestamp'] = datetime.now()
            else:
                if signal == 0:
                    print(f"  ⚪ No setup signal")
                else:
                    print(f"  🔇 Alert skipped (cooldown or low score)")
            
            # Wait for next check
            print(f"  ⏳ Next check in {CHECK_INTERVAL_MINUTES} minutes...")
            time.sleep(CHECK_INTERVAL_MINUTES * 60)
            
        except KeyboardInterrupt:
            print("\n\n✓ Bot stopped by user")
            break
        except Exception as e:
            print(f"✗ Error in bot loop: {e}")
            print(f"  Retrying in {CHECK_INTERVAL_MINUTES} minutes...")
            time.sleep(CHECK_INTERVAL_MINUTES * 60)

# ---------------------------------------------------------
# Test Functions
# ---------------------------------------------------------
def test_telegram_connection():
    """Test Telegram bot connection"""
    print("Testing Telegram connection...")
    message = "🤖 <b>Iron Condor Bot Test</b>\n✓ Connection successful!"
    success = send_telegram_alert(message)
    return success

def run_analysis_once():
    """Run analysis once and display results"""
    print("Running single analysis...")
    
    df = fetch_spy_data()
    if df is None:
        print("Failed to fetch data")
        return
    
    df = calculate_indicators(df)
    signal, score, details = identify_ic_setup(df)
    
    print(f"\n{'='*60}")
    print(f"SINGLE RUN ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Signal: {'Call Spread' if signal == 1 else 'Put Spread' if signal == -1 else 'No Setup'}")
    print(f"Score: {score:.0f}/100")
    print(f"Price: ${details['price']:.2f}")
    print(f"RSI: {details['rsi']:.1f}")
    print(f"ATR: ${details['atr']:.2f}")
    print(f"IV Rank: {details['iv_rank']:.1%}")
    print(f"\nReasons:")
    for reason in details['reasons']:
        print(f"  {reason}")
    
    if signal != 0:
        levels = calculate_ic_levels(details['price'], details['atr'], signal)
        print(f"\nStrike Levels:")
        print(f"  Short Call: ${levels['short_call']:.2f}")
        print(f"  Long Call: ${levels['long_call']:.2f}")
        print(f"  Short Put: ${levels['short_put']:.2f}")
        print(f"  Long Put: ${levels['long_put']:.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_telegram_connection()
        elif sys.argv[1] == "once":
            run_analysis_once()
        else:
            print("Usage: python iron_condor_bot.py [test|once]")
            print("  test: Test Telegram connection")
            print("  once: Run analysis once")
            print("  (no args): Run continuous monitoring")
    else:
        run_bot()