import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests

# --- SECRETS LOADED FROM GITHUB ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TICKERS = ["SPY", "QQQ"]

def send_telegram(msg: str):
    """Sends the breakout alert to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing Telegram credentials.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def fetch_intraday_data(ticker_symbol: str):
    """Fetches intraday 5-minute bars and computes cumulative day-anchored VWAP."""
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="5d", interval="5m")

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    df = df.between_time("09:30", "16:00").copy()

    df["date"] = df.index.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    df["cum_vp"] = (typical_price * df["volume"]).groupby(df["date"]).cumsum()
    df["cum_vol"] = (df["volume"]).groupby(df["date"]).cumsum()
    df["vwap"] = df["cum_vp"] / df["cum_vol"]
    df.drop(columns=["date", "cum_vp", "cum_vol"], inplace=True)

    return df

def generate_intraday_signals(
    df: pd.DataFrame, fast_ema=9, slow_ema=21, atr_period=14, 
    rvol_window=20, adx_period=14, adx_threshold=25.0
) -> pd.DataFrame:
    """Computes dynamic multi-factor entry thresholds for intraday directional debit spreads."""
    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(window=atr_period).mean()

    df["ema_spread_norm"] = (df["ema_fast"] - df["ema_slow"]) / df["atr"]
    df["vwap_dist_norm"] = (df["close"] - df["vwap"]) / df["atr"]

    df["vol_ma"] = df["volume"].rolling(window=rvol_window).mean()
    df["rvol"] = df["volume"] / df["vol_ma"]

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=adx_period)
    adx_col, dmp_col, dmn_col = f"ADX_{adx_period}", f"DMP_{adx_period}", f"DMN_{adx_period}"
    
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        df[adx_col], df[dmp_col], df[dmn_col] = 0.0, 0.0, 0.0

    squeeze_df = df.ta.squeeze(lazybear=False, detailed=True)
    if squeeze_df is not None:
        df = pd.concat([df, squeeze_df], axis=1)
        sqz_on_col = [c for c in df.columns if "SQZ_ON" in c][0]
        sqz_off_col = [c for c in df.columns if "SQZ_OFF" in c][0]
        hist_col = [c for c in df.columns if "SQZ" in c and "ON" not in c and "OFF" not in c and "NO" not in c][0]
        
        df["squeeze_firing"] = (df[sqz_off_col] == 1) & (df[sqz_on_col].shift(1) == 1)
        df["hist_light_blue"] = (df[hist_col] > 0) & (df[hist_col] > df[hist_col].shift(1))
        df["hist_red"] = (df[hist_col] < 0) & (df[hist_col] < df[hist_col].shift(1))
    else:
        df["squeeze_firing"], df["hist_light_blue"], df["hist_red"] = False, False, False

    time = df.index.time
    session_active = (
        ((time >= pd.to_datetime("09:50:00").time()) & (time <= pd.to_datetime("11:30:00").time())) |
        ((time >= pd.to_datetime("13:45:00").time()) & (time <= pd.to_datetime("15:15:00").time()))
    )

    call_spread_trigger = (
        session_active & (df["ema_spread_norm"] > 0.15) & (df["vwap_dist_norm"] >= 0.20) &
        (df["vwap_dist_norm"] <= 1.10) & (df["rvol"] >= 1.30) & (df["close"] > df["open"]) &
        (df[adx_col] >= adx_threshold) & (df[dmp_col] > df[dmn_col]) & 
        df["squeeze_firing"] & df["hist_light_blue"]
    )

    put_spread_trigger = (
        session_active & (df["ema_spread_norm"] < -0.15) & (df["vwap_dist_norm"] <= -0.20) &
        (df["vwap_dist_norm"] >= -1.10) & (df["rvol"] >= 1.30) & (df["close"] < df["open"]) &
        (df[adx_col] >= adx_threshold) & (df[dmn_col] > df[dmp_col]) & 
        df["squeeze_firing"] & df["hist_red"]
    )

    df["signal"] = 0
    df.loc[call_spread_trigger, "signal"] = 1
    df.loc[put_spread_trigger, "signal"] = -1
    df["entry_signal"] = np.where((df["signal"] != 0) & (df["signal"] != df["signal"].shift(1)), df["signal"], 0)

    return df

def run_intraday_scan():
    print("Running SPY & QQQ Intraday 0DTE Scan...")
    messages = [f"⚡ **0DTE Intraday Scanner** ({datetime.datetime.now().strftime('%H:%M ET')})\n"]
    triggers = 0
    
    for ticker in TICKERS:
        raw_df = fetch_intraday_data(ticker)
        if raw_df.empty:
            continue
            
        proc_df = generate_intraday_signals(raw_df)
        latest = proc_df.iloc[-1]
        
        sig_val = latest["entry_signal"]
        price = latest["close"]
        spread_width = 2.0
        
        if sig_val == 1:
            long_strike = np.floor(price)
            short_strike = long_strike + spread_width
            messages.append(f"🟢 **{ticker} 0DTE CALL SPREAD**\nPrice: ${price:.2f} | Momentum Confirmed (Squeeze + ADX + VWAP)\nBuy ${long_strike:.0f}C / Sell ${short_strike:.0f}C")
            triggers += 1
        elif sig_val == -1:
            long_strike = np.ceil(price)
            short_strike = long_strike - spread_width
            messages.append(f"🔴 **{ticker} 0DTE PUT SPREAD**\nPrice: ${price:.2f} | Momentum Confirmed (Squeeze + ADX + VWAP)\nBuy ${long_strike:.0f}P / Sell ${short_strike:.0f}P")
            triggers += 1
            
    if triggers > 0:
        send_telegram("\n\n".join(messages))
        print("Alerts triggered and sent.")
    else:
        print("No active 0DTE signals found. Remaining silent.")

if __name__ == "__main__":
    run_intraday_scan()
