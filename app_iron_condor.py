import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="7-DTE Iron Condor Tracker (QQQ & MSFT)",
    page_icon="🦅",
    layout="wide"
)

st.markdown(
    """
    <style>
    .metric-card { 
        border: 1px solid #30363d; 
        border-radius: 8px; 
        padding: 15px; 
        background-color: rgba(255, 255, 255, 0.03); 
        margin-bottom: 10px; 
    }
    .banner-green { 
        background-color: rgba(0, 230, 118, 0.15); 
        border-left: 6px solid #00e676; 
        padding: 12px 16px; 
        border-radius: 6px; 
        color: #00e676; 
        font-weight: 700; 
        font-size: 1.1rem; 
        margin-bottom: 10px; 
    }
    .banner-yellow { 
        background-color: rgba(255, 235, 59, 0.15); 
        border-left: 6px solid #ffeb3b; 
        padding: 12px 16px; 
        border-radius: 6px; 
        color: #ffeb3b; 
        font-weight: 700; 
        font-size: 1.1rem; 
        margin-bottom: 10px; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Market Data & Quantitative Logic
# ---------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_daily_data(ticker_symbol: str, lookback_days: int = 180) -> pd.DataFrame:
    """Fetches daily candlestick data and normalizes columns."""
    df = yf.Ticker(ticker_symbol).history(period=f"{lookback_days}d", interval="1d")
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna(subset=["close"])

def generate_ic_signals(df: pd.DataFrame, vol_min: float, vol_max: float) -> pd.DataFrame:
    """Evaluates 7-DTE neutral conditions: IV rank proxy, RSI, Bollinger Bands, and ATR."""
    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)

    # 20-day historical volatility annualized and 90-day IV Rank proxy
    df["volatility"] = df["close"].pct_change().rolling(window=20).std() * np.sqrt(252)
    vol_min_hist = df["volatility"].rolling(window=90).min()
    vol_max_hist = df["volatility"].rolling(window=90).max()
    df["iv_rank"] = (df["volatility"] - vol_min_hist) / (vol_max_hist - vol_min_hist)

    df["signal"], df["score"] = 0, 0

    bbu_cols = [c for c in df.columns if "BBU" in c]
    bbl_cols = [c for c in df.columns if "BBL" in c]

    for i in range(len(df)):
        score = 0
        # 1. IV Rank filter
        if pd.notna(df["iv_rank"].iloc[i]) and vol_min <= df["iv_rank"].iloc[i] <= vol_max:
            score += 30
        
        # 2. RSI Neutral Range (40 - 60)
        if pd.notna(df["rsi"].iloc[i]) and 40 <= df["rsi"].iloc[i] <= 60:
            score += 25

        # 3. Bollinger Band Mean Reversion check
        if bbu_cols and bbl_cols and pd.notna(df[bbu_cols[0]].iloc[i]):
            bbu = df[bbu_cols[0]].iloc[i]
            bbl = df[bbl_cols[0]].iloc[i]
            bb_mid = (bbu + bbl) / 2
            bb_range = bbu - bbl
            if bb_range > 0 and abs(df["close"].iloc[i] - bb_mid) < (bb_range * 0.25):
                score += 25

        # 4. ATR Stability check
        if i >= 10:
            atr_sma = df["atr"].iloc[i - 10 : i].mean()
            if atr_sma > 0 and 0.8 <= (df["atr"].iloc[i] / atr_sma) <= 1.2:
                score += 20

        df.iloc[i, df.columns.get_loc("score")] = score
        if score >= 70:
            df.iloc[i, df.columns.get_loc("signal")] = 2

    return df

def calculate_7_dte_strikes(price: float, atr: float, delta_multiplier: float, wing_width: float):
    """Calculates strikes by scaling 1-day ATR to 5 trading days."""
    expected_7_day_move = atr * np.sqrt(5)
    short_distance = expected_7_day_move * delta_multiplier

    short_call = np.ceil(price + short_distance)
    short_put = np.floor(price - short_distance)

    long_call = short_call + wing_width
    long_put = short_put - wing_width

    return short_call, long_call, short_put, long_put

# ---------------------------------------------------------
# UI & Dashboard Layout
# ---------------------------------------------------------
st.title("🦅 7-DTE Iron Condor Daily Screener")
st.caption("Daily neutral volatility and range scanner for QQQ and MSFT to capture short-term Theta decay.")

# Strategy Parameters
st.sidebar.header("⚙️ Strategy Parameters")
delta_mult = st.sidebar.slider("Delta (ATR) Multiplier", 0.5, 2.5, 1.0, step=0.1)
wing_width = st.sidebar.selectbox("Wing Width ($)", [2.0, 3.0, 5.0, 10.0, 15.0], index=2)
vol_min = st.sidebar.slider("Min IV Rank", 0.0, 0.5, 0.20, step=0.05)
vol_max = st.sidebar.slider("Max IV Rank", 0.5, 1.0, 0.85, step=0.05)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

TARGET_TICKERS = ["QQQ", "MSFT"]
target_exp = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%b %d, %Y")

# Process both tickers
processed_data = {}
for t in TARGET_TICKERS:
    raw_df = fetch_daily_data(t)
    if not raw_df.empty:
        processed_data[t] = generate_ic_signals(raw_df, vol_min, vol_max)

if len(processed_data) < len(TARGET_TICKERS):
    st.error("Could not load data for all tickers. Please try refreshing.")
    st.stop()

# Top Summary Banners
st.markdown("### 🚨 Daily Status Overview")
banner_cols = st.columns(len(TARGET_TICKERS))

for idx, t in enumerate(TARGET_TICKERS):
    latest_bar = processed_data[t].iloc[-1]
    with banner_cols[idx]:
        if latest_bar["signal"] == 2:
            st.markdown(
                f'<div class="banner-green">🦅 {t} @ ${latest_bar["close"]:.2f} — TRIGGERED ({latest_bar["score"]:.0f}/100)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="banner-yellow">⚪ {t} @ ${latest_bar["close"]:.2f} — MONITORING ({latest_bar["score"]:.0f}/100)</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")

# Tabbed Detail Views
tab_qqq, tab_msft = st.tabs(["💻 QQQ Dashboard", "🪟 MSFT Dashboard"])

def render_ticker_tab(ticker_symbol: str, df: pd.DataFrame):
    latest = df.iloc[-1]
    short_c, long_c, short_p, long_p = calculate_7_dte_strikes(
        latest["close"], latest["atr"], delta_mult, wing_width
    )

    if latest["signal"] == 2:
        st.markdown(f"#### 📋 {ticker_symbol} Suggested 7-DTE Structure")
        st.markdown(f"**Target Expiration:** {target_exp}")
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Call Wing:**\nSell ${short_c:.0f}C / Buy ${long_c:.0f}C")
        c2.info(f"**Put Wing:**\nSell ${short_p:.0f}P / Buy ${long_p:.0f}P")
        c3.success(f"**Max Profit Range:**\n${short_p:.0f} to ${short_c:.0f}\n\n**Collateral:** ${wing_width * 100:.0f}")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Close Price", f"${latest['close']:.2f}")
    m2.metric("RSI (14)", f"{latest['rsi']:.1f}")
    m3.metric("ATR (14)", f"${latest['atr']:.2f}")
    m4.metric("IV Rank Proxy", f"{latest['iv_rank']:.1%}")

    # Interactive Chart
    st.subheader(f"📊 {ticker_symbol} Daily Price Action & Bollinger Bands")
    plot_df = df.tail(90)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Price",
        )
    )

    bbu_cols = [c for c in plot_df.columns if "BBU" in c]
    bbl_cols = [c for c in plot_df.columns if "BBL" in c]
    if bbu_cols and bbl_cols:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[bbu_cols[0]],
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1, dash="dot"),
                name="Upper BB",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[bbl_cols[0]],
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1, dash="dot"),
                name="Lower BB",
                fill="tonexty",
                fillcolor="rgba(255, 255, 255, 0.05)",
            )
        )

    # Plot lines only if setup is triggered or monitoring near levels
    if latest["signal"] == 2:
        fig.add_hline(y=short_c, line_dash="solid", line_color="#ff5252", annotation_text=f"Short Call (${short_c:.0f})")
        fig.add_hline(y=short_p, line_dash="solid", line_color="#00e676", annotation_text=f"Short Put (${short_p:.0f})")

    fig.update_layout(
        height=550,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker_symbol}")

with tab_qqq:
    render_ticker_tab("QQQ", processed_data["QQQ"])

with tab_msft:
    render_ticker_tab("MSFT", processed_data["MSFT"])
