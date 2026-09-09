import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import yfinance as yf

# =========================================================
# GLOBAL PAGE CONFIGURATION & CSS
# =========================================================
st.set_page_config(page_title="Master Trading Terminal", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
    .metric-card { border: 1px solid #30363d; border-radius: 8px; padding: 15px; background-color: rgba(255, 255, 255, 0.03); margin-bottom: 10px; }
    .banner-green { background-color: rgba(0, 230, 118, 0.15); border-left: 6px solid #00e676; padding: 15px 20px; border-radius: 6px; color: #00e676; font-weight: 700; font-size: 1.2rem; margin-bottom: 10px; }
    .banner-red { background-color: rgba(255, 82, 82, 0.15); border-left: 6px solid #ff5252; padding: 15px 20px; border-radius: 6px; color: #ff5252; font-weight: 700; font-size: 1.2rem; margin-bottom: 10px; }
    .banner-yellow { background-color: rgba(255, 235, 59, 0.15); border-left: 6px solid #ffeb3b; padding: 15px 20px; border-radius: 6px; color: #ffeb3b; font-weight: 700; font-size: 1.2rem; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODULE 1: 0DTE INTRADAY TRACKER (From "Investing" file)
# =========================================================
@st.cache_data(ttl=60)
def fetch_intraday_data(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="5d", interval="5m")
    if df.empty: return pd.DataFrame()
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

def generate_intraday_signals(df: pd.DataFrame, fast_ema=9, slow_ema=21, atr_period=14, rvol_window=20, adx_period=14, adx_threshold=25.0):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()
    
    tr1, tr2, tr3 = df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()
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
        df["histogram_col"] = df[hist_col] 
        df["SQZ_ON"] = df[sqz_on_col]
        df["SQZ_OFF"] = df[sqz_off_col]
    else:
        df["squeeze_firing"], df["hist_light_blue"], df["hist_red"], df["histogram_col"], df["SQZ_ON"], df["SQZ_OFF"] = False, False, False, 0.0, 0, 0

    time = df.index.time
    session_active = (((time >= pd.to_datetime("09:50:00").time()) & (time <= pd.to_datetime("11:30:00").time())) |
                      ((time >= pd.to_datetime("13:45:00").time()) & (time <= pd.to_datetime("15:15:00").time())))

    call_spread_trigger = (session_active & (df["ema_spread_norm"] > 0.15) & (df["vwap_dist_norm"] >= 0.20) &
                           (df["vwap_dist_norm"] <= 1.10) & (df["rvol"] >= 1.30) & (df["close"] > df["open"]) &
                           (df[adx_col] >= adx_threshold) & (df[dmp_col] > df[dmn_col]) & df["squeeze_firing"] & df["hist_light_blue"])

    put_spread_trigger = (session_active & (df["ema_spread_norm"] < -0.15) & (df["vwap_dist_norm"] <= -0.20) &
                          (df["vwap_dist_norm"] >= -1.10) & (df["rvol"] >= 1.30) & (df["close"] < df["open"]) &
                          (df[adx_col] >= adx_threshold) & (df[dmn_col] > df[dmp_col]) & df["squeeze_firing"] & df["hist_red"])

    df["signal"] = 0
    df.loc[call_spread_trigger, "signal"] = 1
    df.loc[put_spread_trigger, "signal"] = -1
    df["entry_signal"] = np.where((df["signal"] != 0) & (df["signal"] != df["signal"].shift(1)), df["signal"], 0)
    return df

def render_intraday_app():
    st.title("🎯 SPY & QQQ Intraday Spread Tracker")
    st.caption("Multi-factor signal scanner analyzing normalized EMA deltas, VWAP displacement, RVOL expansion, and ADX trend strength.")

    st.sidebar.header("⚙️ Strategy Parameters")
    fast_ema = st.sidebar.slider("Fast EMA", 5, 20, 9)
    slow_ema = st.sidebar.slider("Slow EMA", 15, 50, 21)
    atr_len = st.sidebar.slider("ATR Period", 7, 28, 14)
    rvol_win = st.sidebar.slider("RVOL Baseline Window", 10, 40, 20)
    adx_len = st.sidebar.slider("ADX Period", 7, 28, 14)
    adx_thresh = st.sidebar.slider("ADX Threshold", 15.0, 40.0, 25.0, step=1.0)
    spread_width = st.sidebar.selectbox("Spread Width ($)", [1.0, 2.0, 3.0, 5.0], index=1)

    if st.sidebar.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    raw_spy = fetch_intraday_data("SPY")
    raw_qqq = fetch_intraday_data("QQQ")
    if raw_spy.empty or raw_qqq.empty:
        st.error("Unable to retrieve intraday market data.")
        st.stop()

    proc_spy = generate_intraday_signals(raw_spy, fast_ema, slow_ema, atr_len, rvol_win, adx_len, adx_thresh)
    proc_qqq = generate_intraday_signals(raw_qqq, fast_ema, slow_ema, atr_len, rvol_win, adx_len, adx_thresh)

    latest_spy, latest_qqq = proc_spy.iloc[-1], proc_qqq.iloc[-1]
    
    def render_banner(ticker, sig_val, price):
        if sig_val == 1: return f'<div class="banner-green">🟢 {ticker} @ ${price:.2f} — CALL TRIGGERED</div>'
        elif sig_val == -1: return f'<div class="banner-red">🔴 {ticker} @ ${price:.2f} — PUT TRIGGERED</div>'
        else: return f'<div class="banner-yellow">🟡 {ticker} @ ${price:.2f} — MONITORING</div>'

    col_banner1, col_banner2 = st.columns(2)
    with col_banner1: st.markdown(render_banner("SPY", int(latest_spy["signal"]), latest_spy["close"]), unsafe_allow_html=True)
    with col_banner2: st.markdown(render_banner("QQQ", int(latest_qqq["signal"]), latest_qqq["close"]), unsafe_allow_html=True)

    tab_spy, tab_qqq = st.tabs(["🇺🇸 SPY Dashboard", "💻 QQQ Dashboard"])

    def render_dashboard(ticker, df, latest):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Last Price", f"${latest['close']:.2f}")
        col2.metric("VWAP", f"${latest['vwap']:.2f}")
        col3.metric("ATR", f"${latest['atr']:.2f}")
        col4.metric("EMA Spread", f"{latest['ema_spread_norm']:.2f}σ")
        col5.metric("RVOL", f"{latest['rvol']:.2f}x")
        col6.metric("ADX", f"{latest.get(f'ADX_{adx_len}', 0):.1f}")
        st.markdown("---")

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.45, 0.2, 0.15, 0.2], subplot_titles=(f"{ticker} Candlesticks", "TTM Squeeze Momentum", "Relative Volume (RVOL)", "ADX & DMI"))
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap"], line=dict(color="#ffa726", width=1.5), name="VWAP"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], line=dict(color="#29b6f6", width=1), name="Fast EMA"), row=1, col=1)
        
        if "histogram_col" in df.columns:
            colors = ["#00b0ff" if df['histogram_col'].iloc[i] > 0 and df['histogram_col'].iloc[i] > (df['histogram_col'].iloc[i-1] if i>0 else 0) else "#0d47a1" if df['histogram_col'].iloc[i] > 0 else "#ff5252" if df['histogram_col'].iloc[i] < (df['histogram_col'].iloc[i-1] if i>0 else 0) else "#ffeb3b" for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['histogram_col'], marker_color=colors, name="TTM Momentum"), row=2, col=1)
        
        fig.add_trace(go.Bar(x=df.index, y=df["rvol"], marker_color=np.where(df["rvol"] >= 1.3, "#00e676", "#78909c")), row=3, col=1)
        
        adx_col = f"ADX_{adx_len}"
        if adx_col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[adx_col], line=dict(color="#FFD700", width=2), name="ADX"), row=4, col=1)
            
        fig.update_layout(height=950, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode="pan", legend=dict(orientation="h", y=1.02))
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True, 'doubleClick': 'reset'})

    with tab_spy: render_dashboard("SPY", proc_spy, latest_spy)
    with tab_qqq: render_dashboard("QQQ", proc_qqq, latest_qqq)

# =========================================================
# MODULE 2: MULTI-DAY SPREAD TRACKER
# =========================================================
@st.cache_data(ttl=3600)
def fetch_daily_data(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="1y", interval="1d")
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df

def generate_swing_signals(df: pd.DataFrame):
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        df["ADX_14"], df["DMP_14"], df["DMN_14"] = 0.0, 0.0, 0.0

    call_trigger = (df["close"] > df["ema_50"]) & (df["ema_20"] > df["ema_50"]) & (df["rsi"].between(45, 70)) & (df["ADX_14"] >= 20) & (df["DMP_14"] > df["DMN_14"])
    put_trigger = (df["close"] < df["ema_50"]) & (df["ema_20"] < df["ema_50"]) & (df["rsi"].between(30, 55)) & (df["ADX_14"] >= 20) & (df["DMN_14"] > df["DMP_14"])
    
    df["signal"] = 0
    df.loc[call_trigger, "signal"] = 1
    df.loc[put_trigger, "signal"] = -1
    return df

def render_multiday_app():
    st.title("📅 Multi-Day Swing Spread Tracker")
    st.caption("Daily chart analysis using 20/50 EMAs, RSI, and ADX for multi-day debit spreads.")
    
    tickers = ["SPY", "QQQ", "NVDA", "GOOGL", "AAPL", "AMZN"]
    cols = st.columns(3)
    
    for i, ticker in enumerate(tickers):
        df = fetch_daily_data(ticker)
        if df.empty: continue
        
        proc_df = generate_swing_signals(df)
        latest = proc_df.iloc[-1]
        
        with cols[i % 3]:
            st.markdown(f"### {ticker}")
            st.metric("Last Price", f"${latest['close']:.2f}")
            
            sig_val = latest["signal"]
            spread_width = max(round(latest["atr"]), 1)
            
            if sig_val == 1:
                st.success(f"🟢 **CALL SPREAD**\nBuy ${round(latest['close'])}C / Sell ${round(latest['close']) + spread_width}C")
            elif sig_val == -1:
                st.error(f"🔴 **PUT SPREAD**\nBuy ${round(latest['close'])}P / Sell ${round(latest['close']) - spread_width}P")
            else:
                st.info("⚪ MONITORING")
            st.markdown("---")

# =========================================================
# MODULE 3: TTM SQUEEZE SCANNER
# =========================================================
@st.cache_data(ttl=300)
def fetch_and_calculate_squeeze(ticker: str):
    df = yf.download(ticker, period="10d", interval="5m", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    squeeze_df = df.ta.squeeze(lazybear=False, detailed=True)
    if squeeze_df is not None:
        df = pd.concat([df, squeeze_df], axis=1)
        sqz_on_col = [c for c in df.columns if "SQZ_ON" in c][0]
        sqz_off_col = [c for c in df.columns if "SQZ_OFF" in c][0]
        hist_col = [c for c in df.columns if "SQZ" in c and "ON" not in c and "OFF" not in c and "NO" not in c][0]
        
        df["SQZ_ON"], df["SQZ_OFF"], df["HISTOGRAM"] = df[sqz_on_col], df[sqz_off_col], df[hist_col]
        df["squeeze_firing"] = (df["SQZ_OFF"] == 1) & (df["SQZ_ON"].shift(1) == 1)
        df["hist_light_blue"] = (df["HISTOGRAM"] > 0) & (df["HISTOGRAM"] > df["HISTOGRAM"].shift(1))
        df["hist_red"] = (df["HISTOGRAM"] < 0) & (df["HISTOGRAM"] < df["HISTOGRAM"].shift(1))
        
        df["signal"] = 0
        df.loc[df["squeeze_firing"] & df["hist_light_blue"], "signal"] = 1
        df.loc[df["squeeze_firing"] & df["hist_red"], "signal"] = -1
    return df

def render_squeeze_app():
    st.title("💥 Mega-Cap TTM Squeeze Scanner")
    st.caption("Detects explosive volatility breakouts when Bollinger Bands narrow inside Keltner Channels.")
    if st.sidebar.button("🔄 Scan Market Now"): st.cache_data.clear()

    tickers = ["SPY", "QQQ", "GOOGL", "NVDA", "AMZN"]
    cols = st.columns(len(tickers))
    
    for col, ticker in zip(cols, tickers):
        df = fetch_and_calculate_squeeze(ticker)
        with col:
            st.subheader(ticker)
            if df is None or "HISTOGRAM" not in df.columns:
                st.error("Data error"); continue
                
            latest = df.iloc[-1]
            st.metric("Last Price", f"${latest['close']:.2f}")
            signal = latest.get("signal", 0)
            
            if signal == 1: st.success("🟢 SQUEEZE FIRED: CALL SPREAD")
            elif signal == -1: st.error("🔴 SQUEEZE FIRED: PUT SPREAD")
            elif latest["SQZ_ON"] == 1: st.warning("🟡 SQUEEZE COMPRESSING")
            else: st.info("⚪ NO ACTIVE SQUEEZE")

            colors = ["#00b0ff" if df['HISTOGRAM'].iloc[i] > 0 and df['HISTOGRAM'].iloc[i] > (df['HISTOGRAM'].iloc[i-1] if i>0 else 0) else "#0d47a1" if df['HISTOGRAM'].iloc[i] > 0 else "#ff5252" if df['HISTOGRAM'].iloc[i] < (df['HISTOGRAM'].iloc[i-1] if i>0 else 0) else "#ffeb3b" for i in range(len(df))]
            fig = go.Figure(go.Bar(x=df.index, y=df['HISTOGRAM'], marker_color=colors, name="Momentum"))
            fig.update_layout(height=250, margin=dict(l=5, r=5, t=30, b=5), template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# =========================================================
# MAIN APP ROUTER (SIDEBAR NAVIGATION)
# =========================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("Market Engine")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Select Dashboard:", ["⚡ 0DTE Intraday Tracker", "💥 TTM Squeeze Scanner", "📅 Multi-Day Option Spreads"])
st.sidebar.markdown("---")

if app_mode == "⚡ 0DTE Intraday Tracker": render_intraday_app()
elif app_mode == "💥 TTM Squeeze Scanner": render_squeeze_app()
elif app_mode == "📅 Multi-Day Option Spreads": render_multiday_app()
