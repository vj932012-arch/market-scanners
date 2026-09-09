import os
import requests
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# App Configuration & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Day Spread Tracker & Workflow Trigger",
    page_icon="🔭",
    layout="wide"
)

st.markdown("""
<style>
.ticker-card {
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    background-color: rgba(255, 255, 255, 0.03);
    margin-bottom: 15px;
}
.status-bull { color: #00e676; font-weight: bold; font-size: 1.05rem; }
.status-bear { color: #ff5252; font-weight: bold; font-size: 1.05rem; }
.status-neutral { color: #b0bec5; font-weight: bold; font-size: 1.05rem; }
.action-text { font-size: 0.88rem; margin-top: 6px; color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

TICKERS = ["SPY", "QQQ", "NVDA", "GOOGL", "AAPL", "AMZN"]

# ---------------------------------------------------------
# Helper Functions: Secrets Retrieval
# ---------------------------------------------------------
def get_secret(key: str, default: str = "") -> str:
    """Retrieves credentials from Streamlit secrets or OS environment variables."""
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)

# ---------------------------------------------------------
# Data Fetching & Indicators
# ---------------------------------------------------------
# FIX: Reduced cache TTL to 5 minutes (300 seconds) so live data refreshes more frequently
@st.cache_data(ttl=300)
def fetch_daily_data(ticker: str) -> pd.DataFrame:
    """Fetches daily candlestick data and normalizes column headers."""
    df = yf.Ticker(ticker).history(period="1y", interval="1d")
    if df.empty:
        return df

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df

def generate_swing_signals(df: pd.DataFrame, adx_thresh: float = 20.0) -> pd.DataFrame:
    """Computes technical indicators and swing thresholds for multi-day debit spreads."""
    df = df.copy()

    # Moving averages
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()

    # Volatility & Momentum
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["rsi"] = ta.rsi(df["close"], length=14)

    # Trend Strength (ADX / DMI)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        df["ADX_14"], df["DMP_14"], df["DMN_14"] = 0.0, 0.0, 0.0

    # Entry condition filters
    call_trigger = (
        (df["close"] > df["ema_50"]) &
        (df["ema_20"] > df["ema_50"]) &
        (df["rsi"] >= 45) & (df["rsi"] <= 70) &
        (df["ADX_14"] >= adx_thresh) &
        (df["DMP_14"] > df["DMN_14"])
    )

    put_trigger = (
        (df["close"] < df["ema_50"]) &
        (df["ema_20"] < df["ema_50"]) &
        (df["rsi"] >= 30) & (df["rsi"] <= 55) &
        (df["ADX_14"] >= adx_thresh) &
        (df["DMN_14"] > df["DMP_14"])
    )

    df["signal"] = 0
    df.loc[call_trigger, "signal"] = 1
    df.loc[put_trigger, "signal"] = -1
    return df

def get_spread_recommendation(price: float, atr: float, signal: int):
    """Calculates spread width using ATR for dynamic strike selection."""
    spread_width = max(np.ceil(atr * 1.0), 1.0)

    if signal == 1:
        long_strike = np.floor(price)
        short_strike = long_strike + spread_width
        return {
            "status": "🟢 CALL SPREAD",
            "class": "status-bull",
            "action": f"Buy ${long_strike:.0f}C <br> Sell ${short_strike:.0f}C",
            "raw_text": f"🟢 CALL SPREAD: Buy ${long_strike:.0f}C / Sell ${short_strike:.0f}C"
        }
    elif signal == -1:
        long_strike = np.ceil(price)
        short_strike = long_strike - spread_width
        return {
            "status": "🔴 PUT SPREAD",
            "class": "status-bear",
            "action": f"Buy ${long_strike:.0f}P <br> Sell ${short_strike:.0f}P",
            "raw_text": f"🔴 PUT SPREAD: Buy ${long_strike:.0f}P / Sell ${short_strike:.0f}P"
        }
    else:
        return {
            "status": "⚪ MONITORING",
            "class": "status-neutral",
            "action": "Awaiting clear<br>daily trend.",
            "raw_text": "⚪ MONITORING: No signal."
        }

# ---------------------------------------------------------
# Sidebar: Automated Workflow Trigger Controls
# ---------------------------------------------------------
st.sidebar.header("⚡ Workflow & Bot Controls")

# FIX: Added a manual cache bust button to force a fresh data pull[span_2](start_span)[span_2](end_span)[span_3](start_span)[span_3](end_span)
if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

gh_token = get_secret("GH_PAT")
gh_repo = get_secret("GH_REPO")
tg_token = get_secret("TELEGRAM_BOT_TOKEN")
tg_chat_id = get_secret("TELEGRAM_CHAT_ID")

st.sidebar.subheader("1. GitHub Actions Trigger")
st.sidebar.caption("Manually dispatches `.github/workflows/swing_bot.yml`")

if st.sidebar.button("🚀 Run GitHub Action Now"):
    if not gh_token or not gh_repo:
        st.sidebar.error("Missing `GH_PAT` or `GH_REPO` in secrets.")
    else:
        url = f"https://api.github.com/repos/{gh_repo}/actions/workflows/swing_bot.yml/dispatches"
        headers = {
            "Authorization": f"Bearer {gh_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {"ref": "main"}
        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code == 204:
            st.sidebar.success("Workflow triggered on GitHub!")
        else:
            st.sidebar.error(f"Failed ({resp.status_code}): {resp.text}")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Direct Telegram Notification")
st.sidebar.caption("Sends the latest signals directly to your Telegram chat.")

# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
st.title("🔭 Multi-Day Debit Spread Screener")
st.caption("Target Expiration: 21 to 45 DTE. Scans daily bars for 20/50 EMA structure, RSI momentum, and ADX strength.")

# Process all tickers
all_data = {}
signals_summary = []

for ticker in TICKERS:
    raw = fetch_daily_data(ticker)
    if not raw.empty:
        processed = generate_swing_signals(raw)
        all_data[ticker] = processed
        latest = processed.iloc[-1]
        rec = get_spread_recommendation(latest["close"], latest["atr"], latest["signal"])
        signals_summary.append((ticker, latest["close"], rec))

# Sidebar direct Telegram send trigger
if st.sidebar.button("📲 Send Telegram Alert Now"):
    if not tg_token or not tg_chat_id:
        st.sidebar.error("Missing `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID`.")
    else:
        lines = [f"🔭 **Multi-Day Swing Status** ({datetime.datetime.now().strftime('%b %d, %Y')})\n"]
        for t, px, rec in signals_summary:
            lines.append(f"• **{t}** (${px:.2f}): {rec['raw_text']}")
        msg_payload = {
            "chat_id": tg_chat_id,
            "text": "\n".join(lines),
            "parse_mode": "Markdown"
        }
        tg_resp = requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", json=msg_payload)
        if tg_resp.status_code == 200:
            st.sidebar.success("Telegram notification sent!")
        else:
            st.sidebar.error(f"Telegram error ({tg_resp.status_code})")

# Top Summary Banner
cols = st.columns(len(TICKERS))
for i, (ticker, price, rec) in enumerate(signals_summary):
    with cols[i]:
        st.markdown(f"""
        <div class="ticker-card">
            <h3>{ticker}</h3>
            <h4>${price:.2f}</h4>
            <div class="{rec['class']}">{rec['status']}</div>
            <div class="action-text">{rec['action']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("📊 Individual Ticker Technical Charts")

# Ticker chart tabs
tabs = st.tabs(TICKERS)
for i, ticker in enumerate(TICKERS):
    with tabs[i]:
        if ticker not in all_data:
            st.error(f"Data unavailable for {ticker}")
            continue

        df = all_data[ticker].tail(120)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f"{ticker} Daily Candlesticks & Moving Averages", "RSI (14)", "ADX & Directional Movement (DMI)")
        )

        # Candlesticks + EMAs + SMA
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_20"], line=dict(color="#29b6f6", width=1.5), name="20 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_50"], line=dict(color="#ab47bc", width=1.5), name="50 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_200"], line=dict(color="#ffa726", width=2, dash="dot"), name="200 SMA"), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], line=dict(color="#00e676", width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ff5252", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00e676", row=2, col=1)

        # ADX / DMI
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX_14"], line=dict(color="#FFD700", width=2), name="ADX"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["DMP_14"], line=dict(color="#00e676", width=1), name="+DI"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["DMN_14"], line=dict(color="#ff5252", width=1), name="-DI"), row=3, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="#b0bec5", row=3, col=1, annotation_text="Trend Baseline (20)")

        fig.update_layout(
            height=750,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=15, r=15, t=30, b=15),
            dragmode="pan"
        )
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)

        # FIX: Added unique keys tracking the ticker string so Streamlit properly retains chart state across tabs[span_4](start_span)[span_4](end_span)[span_5](start_span)[span_5](end_span)
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            config={"scrollZoom": True, "displayModeBar": False},
            key=f"chart_{ticker}"
        )
