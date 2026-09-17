import os
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Configuration & Robinhood Execution Rules
# ---------------------------------------------------------
TICKER = "SPY"
LOOKBACK_DAYS = 90

IC_SETUP = {
    'volatility_min': 0.20,
    'volatility_max': 0.85,
    'rsi_sell_threshold': 65,  
    'rsi_buy_threshold': 35,
    'min_score': 70.0,
    'delta_atr_multiplier': 1.0, # Lowered from 1.5/2.0 to capture ~20 delta premium
    'wing_width': 5.0            # Expanded to $5 to improve credit ratios
}

def fetch_spy_data_polygon(ticker: str = TICKER, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key: return pd.DataFrame()

    end_date = datetime.now(pytz.timezone("America/New_York")).date()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "results" not in data: return pd.DataFrame()
            
        df = pd.DataFrame(data["results"])
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_convert("America/New_York")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['RSI'] = ta.rsi(df_calc['close'], length=14)
    bb = ta.bbands(df_calc['close'], length=20, std=2)
    if bb is not None: df_calc = pd.concat([df_calc, bb], axis=1)
    df_calc['ATR'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=14)
    df_calc['Volatility'] = df_calc['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    return df_calc

def identify_ic_setup(df: pd.DataFrame):
    details = {"reasons": [], "rsi": None, "iv_rank": None, "atr": None, "price": None}
    signal, score = 0, 0.0
    latest = df.iloc[-1]
    
    details['price'] = latest['close']
    details['rsi'] = latest.get('RSI', np.nan)
    details['atr'] = latest.get('ATR', 0)

    vol_min = df['Volatility'].min()
    vol_max = df['Volatility'].max()
    details['iv_rank'] = (latest.get('Volatility', 0) - vol_min) / (vol_max - vol_min) if vol_max > vol_min else 0.5

    if IC_SETUP['volatility_min'] <= details['iv_rank'] <= IC_SETUP['volatility_max']:
        score += 30
        details['reasons'].append("✓ Volatility in ideal range")

    bbu_col = [c for c in df.columns if 'BBU' in c]
    bbl_col = [c for c in df.columns if 'BBL' in c]
    if bbu_col and bbl_col:
        bb_mid = (latest[bbu_col[0]] + latest[bbl_col[0]]) / 2
        bb_range = latest[bbu_col[0]] - latest[bbl_col[0]]
        if bb_range > 0 and abs(latest['close'] - bb_mid) < bb_range * 0.25:
            score += 25
            details['reasons'].append("✓ Price oscillating near middle BB")

    if not np.isnan(details['rsi']):
        if 40 <= details['rsi'] <= 60:
            score += 25
            details['reasons'].append(f"✓ RSI Neutral ({details['rsi']:.1f})")

    atr_sma = df['ATR'].rolling(window=10).mean().iloc[-1]
    if atr_sma > 0 and 0.8 <= (details['atr'] / atr_sma) <= 1.2:
        score += 20
        details['reasons'].append("✓ ATR is stable")

    if score >= IC_SETUP['min_score']: signal = 2
    return signal, score, details

def calculate_ic_levels(close_price: float, atr: float):
    # Scale 1-day ATR to 5 trading days
    expected_move = atr * np.sqrt(5)
    short_dist = expected_move * IC_SETUP['delta_atr_multiplier']

    short_call = np.ceil(close_price + short_dist)
    short_put = np.floor(close_price - short_dist)
    
    long_call = short_call + IC_SETUP['wing_width']
    long_put = short_put - IC_SETUP['wing_width']
    
    return {"short_call": short_call, "long_call": long_call, "short_put": short_put, "long_put": long_put}

def send_telegram_alert(message: str):
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat_id:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": tg_chat_id, "text": message, "parse_mode": "HTML"})

def run_scan():
    df = fetch_spy_data_polygon()
    if df.empty: return
        
    df_calc = calculate_indicators(df)
    signal, score, details = identify_ic_setup(df_calc)
    now_et = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S ET')
    
    if signal == 2:
        strikes = calculate_ic_levels(details['price'], details['atr'])
        msg = f"""<b>🦅 7-DTE NEUTRAL IRON CONDOR</b>\n
<b>Current Price:</b> ${details['price']:.2f}
<b>Setup Score:</b> {score}/100\n
<b>Suggested Strikes (~20 Delta, ${IC_SETUP['wing_width']}-Wide):</b>
Short Call: ${strikes['short_call']:.0f}C / Long Call: ${strikes['long_call']:.0f}C
Short Put: ${strikes['short_put']:.0f}P / Long Put: ${strikes['long_put']:.0f}P\n
<b>Max Profit Zone:</b> ${strikes['short_put']:.0f} - ${strikes['short_call']:.0f}
<b>Timestamp:</b> {now_et}"""
        send_telegram_alert(msg)
    else:
        msg = f"<b>🦅 IRON CONDOR SCANNER (Heartbeat)</b>\nStatus: ⚪ Monitoring\nScore: {score:.0f}/100\nPrice: ${details['price']:.2f}\nTimestamp: {now_et}"
        send_telegram_alert(msg)

if __name__ == "__main__":
    run_scan()
