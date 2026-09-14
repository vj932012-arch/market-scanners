import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="7-DTE Iron Condor Tracker", page_icon="🦅", layout="wide"
)

st.markdown(
    """
    <style>
    .metric-card { border: 1px solid #30363d; border-radius: 8px; padding: 15px; background-color: rgba(255, 255, 255, 0.03); margin-bottom: 10px; }
    .banner-green { background-color: rgba(0, 230, 118, 0.15); border-left: 6px solid #00e676; padding: 15px 20px; border-radius: 6px; color: #00e676; font-weight: 700; font-size: 1.2rem; margin-bottom: 10px; }
    .banner-yellow { background-color: rgba(255, 235, 59, 0.15); border-left: 6px solid #ffeb3b; padding: 15px 20px; border-radius: 6px; color: #ffeb3b; font-weight: 700; font-size: 1.2rem; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Market Data & Indicator Logic
# ---------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_daily_data(ticker_symbol: str, lookback_days: int = 180):
    """Fetches daily historical data for swing trade indicators."""
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=f"{lookback_days}d", interval="1d")

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    # Drop rows with NaN close prices
    df = df.dropna(subset=['close'])
    return df

def generate_ic_signals(df: pd.DataFrame, vol_min: float, vol_max: float) -> pd.DataFrame:
    """Calculates indicators and identifies neutral 7-DTE setups."""
    df = df.copy()

    # RSI & ATR
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Bollinger Bands (20, 2)
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)

    # Volatility & IV Rank Proxy
    df["volatility"] = df["close"].pct_change().rolling(window=20).std() * np.sqrt(252)
    vol_min_historical = df["volatility"].rolling(window=90).min()
    vol_max_historical = df["volatility"].rolling(window=90).max()
    df["iv_rank"] = (df["volatility"] - vol_min_historical) / (vol_max_historical - vol_min_historical)

    # Signal Generation (Neutral Setup)
    df["signal"] = 0
    df["score"] = 0

    for i in range(len(df)):
        score = 0
        
        # 1. Volatility check
        if pd.notna(df["iv_rank"].iloc[i]) and vol_min <= df["iv_rank"].iloc[i] <= vol_max:
            score += 30
            
        # 2. RSI Neutral check (40-60)
        if pd.notna(df["rsi"].iloc[i]) and 40 <= df["rsi"].iloc[i] <= 60:
            score += 25
            
        # 3. Bollinger Bands Mean Reversion check
        bbu_col = [c for c in df.columns if 'BBU' in c]
        bbl_col = [c for c in df.columns if 'BBL' in c]
        if bbu_col and bbl_col and pd.notna(df[bbu_col[0]].iloc[i]):
            bb_upper = df[bbu_col[0]].iloc[i]
            bb_lower = df[bbl_col[0]].iloc[i]
            bb_mid = (bb_upper + bb_lower) / 2
            bb_range = bb_upper - bb_lower
            if bb_range > 0 and abs(df["close"].iloc[i] - bb_mid) < (bb_range * 0.25):
                score += 25
                
        # 4. ATR Stability check
        if i >= 10:
            atr_sma = df["atr"].iloc[i-10:i].mean()
            if atr_sma > 0 and 0.8 <= (df["atr"].iloc[i] / atr_sma) <= 1.2:
                score += 20
                
        df.iloc[i, df.columns.get_loc("score")] = score
        if score >= 70:
            df.iloc[i, df.columns.get_loc("signal")] = 2

    return df

def calculate_7_dte_strikes(price: float, atr: float, delta_multiplier: float, wing_width: float):
    """Calculates whole-dollar strikes based on a 7-day expected move."""
    # Scale 1-day ATR to 5 trading days (7 calendar days)
    expected_7_day_move = atr * np.sqrt(5)
    short_distance = expected_7_day_move * delta_multiplier

    # Round inner strikes to nearest dollar for liquidity
    short_call = np.ceil(price + short_distance)
    short_put = np.floor(price - short_distance)
    
    # Fixed wing width for defined collateral
    long_call = short_call + wing_width
    long_put = short_put - wing_width
    
    return short_call, long_call, short_put, long_put

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🦅 7-DTE Iron Condor Tracker")
st.caption("Scans daily market data for neutral setups to harvest accelerating short-term Theta decay.")

st.sidebar.header("⚙️ Strategy Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", "SPY").upper()
delta_mult = st.sidebar.slider("Delta (ATR) Multiplier", 1.0, 3.0, 1.5, step=0.1)
wing_width = st.sidebar.selectbox("Wing Width ($)", [1.0, 2.0, 3.0, 5.0], index=1)
vol_min = st.sidebar.slider("Min IV Rank", 0.0, 0.5, 0.20, step=0.05)
vol_max = st.sidebar.slider("Max IV Rank", 0.5, 1.0, 0.85, step=0.05)

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.rerun()

raw_df = fetch_daily_data(ticker)
if raw_df.empty:
    st.error("Failed to fetch market data.")
    st.stop()

proc_df = generate_ic_signals(raw_df, vol_min, vol_max)
latest = proc_df.iloc[-1]

target_exp = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime('%b %d, %Y')

# ---------------------------------------------------------
# Dynamic Banner & Strikes
# ---------------------------------------------------------
if latest["signal"] == 2:
    st.markdown(f'<div class="banner-green">🦅 {ticker} @ ${latest["close"]:.2f} — NEUTRAL SETUP TRIGGERED (Score: {latest["score"]:.0f}/100)</div>', unsafe_allow_html=True)
    
    short_c, long_c, short_p, long_p = calculate_7_dte_strikes(latest["close"], latest["atr"], delta_mult, wing_width)
    
    st.markdown(f"### 📋 {ticker} Suggested 7-DTE Structure")
    st.markdown(f"**Target Expiration:** {target_exp}")
    
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Call Side (Credit):**\nSell ${short_c:.0f}C / Buy ${long_c:.0f}C")
    c2.info(f"**Put Side (Credit):**\nSell ${short_p:.0f}P / Buy ${long_p:.0f}P")
    c3.success(f"**Max Profit Zone:**\n${short_p:.0f} to ${short_c:.0f}\n\n**Collateral:** ${wing_width*100:.0f}")
else:
    st.markdown(f'<div class="banner-yellow">⚪ {ticker} @ ${latest["close"]:.2f} — MONITORING (Score: {latest["score"]:.0f}/100)</div>', unsafe_allow_html=True)
    st.write(f"Waiting for neutral trend confirmation. Requires IV Rank > {vol_min}, RSI between 40-60, and stable ATR.")

st.markdown("---")

# ---------------------------------------------------------
# Metric Cards
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Close Price", f"${latest['close']:.2f}")
col2.metric("RSI (Daily)", f"{latest['rsi']:.1f}")
col3.metric("ATR (Daily)", f"${latest['atr']:.2f}")
col4.metric("IV Rank", f"{latest['iv_rank']:.1%}")

# ---------------------------------------------------------
# Charting
# ---------------------------------------------------------
st.subheader(f"📊 {ticker} Daily Price Action & Bollinger Bands")

plot_df = proc_df.tail(90)
fig = go.Figure()

fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"], name="Price"))

bbu_col = [c for c in plot_df.columns if 'BBU' in c][0]
bbl_col = [c for c in plot_df.columns if 'BBL' in c][0]

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[bbu_col], line=dict(color="rgba(255, 255, 255, 0.2)", width=1, dash="dot"), name="Upper BB"))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[bbl_col], line=dict(color="rgba(255, 255, 255, 0.2)", width=1, dash="dot"), name="Lower BB", fill="tonexty", fillcolor="rgba(255, 255, 255, 0.05)"))

if latest["signal"] == 2:
    fig.add_hline(y=short_c, line_dash="solid", line_color="#ff5252", annotation_text=f"Short Call (${short_c:.0f})")
    fig.add_hline(y=short_p, line_dash="solid", line_color="#00e676", annotation_text=f"Short Put (${short_p:.0f})")

fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)
