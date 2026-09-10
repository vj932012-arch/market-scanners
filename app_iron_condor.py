import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Iron Condor Tracker", page_icon="📊", layout="wide")
st.title("📊 SPY Iron Condor Tracker")
st.caption("Technical analysis for optimal neutral iron condor windows using Bollinger Bands, RSI, and ATR")

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TICKER = "SPY"
LOOKBACK_DAYS = 60
PERIOD_SHORT = 14
PERIOD_LONG = 50

# Strike & Delta Configuration
DELTA_ATR_MULTIPLIER = 2.0  # Target Delta: ~16 Delta / 1 SD
WING_WIDTH = 2.0            # Defined-risk spread width in dollars

# Iron Condor Parameters (Updated for Neutral Logic)
IC_SETUP = {
    "volatility_min": 0.20,        # Minimum IV Rank (20%)
    "volatility_max": 0.85,        # Maximum IV Rank
    "min_score": 70.0,             # Minimum score to trigger setup
}

# ---------------------------------------------------------
# Data Fetching & Analysis
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_spy_data(ticker=TICKER, days=LOOKBACK_DAYS):
    """Fetch historical SPY data"""
    df = yf.download(ticker, period=f"{days}d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    return df

def calculate_indicators(df):
    """Calculate technical indicators for iron condor setup"""
    df_calc = df.copy()
    
    # RSI (Relative Strength Index)
    df_calc['RSI'] = ta.rsi(df_calc['close'], length=PERIOD_SHORT)
    
    # Bollinger Bands
    bb = ta.bbands(df_calc['close'], length=20, std=2)
    if bb is not None:
        df_calc = pd.concat([df_calc, bb], axis=1)
    
    # ATR (Average True Range) for volatility
    df_calc['ATR'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=14)
    
    # SMA for trend
    df_calc['SMA_14'] = ta.sma(df_calc['close'], length=PERIOD_SHORT)
    df_calc['SMA_50'] = ta.sma(df_calc['close'], length=PERIOD_LONG)
    
    # MACD for momentum
    macd = ta.macd(df_calc['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df_calc = pd.concat([df_calc, macd], axis=1)
    
    # Volatility (Standard Deviation of returns)
    df_calc['Volatility'] = df_calc['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    return df_calc

def identify_ic_setup(df):
    """Identify ideal neutral iron condor setup windows"""
    df = df.copy()
    df['IC_Signal'] = 0  # 0=No Setup, 1=Call Spread, -1=Put Spread, 2=Neutral IC
    df['IC_Score'] = 0.0
    
    latest = df.iloc[-1]
    
    # Get values
    rsi = latest.get('RSI', np.nan)
    close_price = latest['close']
    atr = latest.get('ATR', 0)
    volatility = latest.get('Volatility', 0)
    
    # Calculate IV Rank (normalized volatility)
    vol_min = df['Volatility'].min()
    vol_max = df['Volatility'].max()
    iv_rank = (volatility - vol_min) / (vol_max - vol_min) if vol_max > vol_min else 0.5
    
    score = 0.0
    reasons = []
    
    # 1. Volatility is elevated for premium (30 points)
    if IC_SETUP['volatility_min'] <= iv_rank <= IC_SETUP['volatility_max']:
        score += 30
        reasons.append(f"✓ Volatility elevated for premium collection (IVR: {iv_rank:.1%})")
    else:
        reasons.append(f"✗ Volatility too low/high (IV Rank: {iv_rank:.1%})")
    
    # 2. Bollinger Bands Mean Reversion (25 points)
    bb_cols = [c for c in df.columns if 'BBL' in c or 'BBU' in c or 'BBM' in c]
    if bb_cols:
        bb_upper = df[[c for c in df.columns if 'BBU' in c][0]].iloc[-1] if any('BBU' in c for c in df.columns) else close_price + atr
        bb_lower = df[[c for c in df.columns if 'BBL' in c][0]].iloc[-1] if any('BBL' in c for c in df.columns) else close_price - atr
        bb_mid = (bb_upper + bb_lower) / 2
        
        distance_to_mid = abs(close_price - bb_mid)
        bb_range = bb_upper - bb_lower
        
        if bb_range > 0 and distance_to_mid < bb_range * 0.25:
            score += 25
            reasons.append("✓ Price near middle of Bollinger Bands (Range-bound)")
        else:
            reasons.append("✗ Price too close to BB edges (Trending)")
            
    # 3. RSI Neutrality Check (25 points)
    if not np.isnan(rsi):
        if 40 <= rsi <= 60:
            score += 25
            reasons.append(f"✓ RSI is neutral ({rsi:.1f})")
        elif rsi > 70:
            reasons.append(f"✗ RSI overbought ({rsi:.1f}) - Potential Put Spread instead")
            df.loc[df.index[-1], 'IC_Signal'] = -1 
        elif rsi < 30:
            reasons.append(f"✗ RSI oversold ({rsi:.1f}) - Potential Call Spread instead")
            df.loc[df.index[-1], 'IC_Signal'] = 1
        else:
            reasons.append(f"✗ RSI indicates mild trend/momentum ({rsi:.1f})")
    
    # 4. ATR Normalization Check (20 points)
    atr_sma = df['ATR'].rolling(window=10).mean().iloc[-1]
    if atr_sma > 0:
        atr_ratio = atr / atr_sma
        if 0.8 <= atr_ratio <= 1.2:
            score += 20
            reasons.append("✓ Volatility (ATR) is stable")
        else:
            reasons.append(f"✗ Volatility (ATR) is expanding/contracting too fast ({atr_ratio:.2f}x avg)")
    
    df.loc[df.index[-1], 'IC_Score'] = score
    
    # Set main signal if score is met
    if score >= IC_SETUP['min_score']:
        df.loc[df.index[-1], 'IC_Signal'] = 2
    
    return df, reasons, score, rsi, iv_rank

def calculate_ic_levels(close_price, atr, signal):
    """Calculate tradeable, rounded Iron Condor strike levels"""
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

# ---------------------------------------------------------
# Main App Logic
# ---------------------------------------------------------
# Fetch and analyze data
df = fetch_spy_data()
df = calculate_indicators(df)
df, reasons, score, rsi, iv_rank = identify_ic_setup(df)

# Get latest values
latest = df.iloc[-1]
close_price = latest['close']
atr = latest['ATR']
signal = int(latest['IC_Signal'])
sma_14 = latest['SMA_14']
sma_50 = latest['SMA_50']

# Calculate IC levels
ic_levels = calculate_ic_levels(close_price, atr, signal)

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("SPY Price", f"${close_price:.2f}")
with col2:
    st.metric("RSI (14)", f"{rsi:.1f}")
with col3:
    st.metric("ATR (14)", f"${atr:.2f}")
with col4:
    st.metric("IV Rank", f"{iv_rank:.1%}")

st.divider()

# Signal and Recommendations
col_signal, col_levels = st.columns(2)

with col_signal:
    st.subheader("🎯 Setup Analysis")
    st.metric("Setup Score", f"{score:.0f}/100")
    
    if signal == 2:
        st.success("🦅 NEUTRAL IRON CONDOR CANDIDATE")
        st.write("**Setup**: Range-bound price action, neutral RSI, and elevated volatility.")
    elif signal == 1:
        st.info("🟢 CALL SPREAD CANDIDATE (Skewed)")
        st.write("**Setup**: Price oversold. Directional skew suggested.")
    elif signal == -1:
        st.warning("🔴 PUT SPREAD CANDIDATE (Skewed)")
        st.write("**Setup**: Price overbought. Directional skew suggested.")
    else:
        st.info("⚪ NO CLEAR SETUP")
        st.write("**Wait for**: Range-bound consolidation and stable volatility.")
    
    st.markdown("### Conditions Matrix:")
    for reason in reasons:
        st.write(reason)

with col_levels:
    if signal != 0 and ic_levels:
        st.subheader("📍 Suggested Strike Levels")
        
        level_data = {
            "Strike Type": ["Short Call", "Long Call", "Short Put", "Long Put"],
            "Price": [
                f"${ic_levels.get('short_call'):.2f}",
                f"${ic_levels.get('long_call'):.2f}",
                f"${ic_levels.get('short_put'):.2f}",
                f"${ic_levels.get('long_put'):.2f}"
            ]
        }
        level_df = pd.DataFrame(level_data)
        st.dataframe(level_df, use_container_width=True, hide_index=True)
        
        st.write(f"**Max Profit Width (Wings)**: ${ic_levels['wing_width']:.2f}")
        st.write(f"**Collateral Required**: ${ic_levels['wing_width'] * 100:.2f} per contract")
        st.write(f"**Max Profit Region**: ${ic_levels['max_profit_range_low']:.2f} to ${ic_levels['max_profit_range_high']:.2f}")

st.divider()

# ---------------------------------------------------------
# Technical Chart
# ---------------------------------------------------------
st.subheader("📈 SPY Price & Technical Indicators")

fig = go.Figure()

# Candlestick-style price chart
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['close'],
    mode='lines',
    name='SPY Close',
    line=dict(color='#1f77b4', width=2),
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>'
))

# Bollinger Bands
bb_upper_col = [c for c in df.columns if 'BBU' in c][0] if any('BBU' in c for c in df.columns) else None
bb_lower_col = [c for c in df.columns if 'BBL' in c][0] if any('BBL' in c for c in df.columns) else None

if bb_upper_col:
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[bb_upper_col],
        mode='lines',
        name='BB Upper',
        line=dict(color='rgba(255, 100, 100, 0.3)'),
        hoverinfo='skip'
    ))

if bb_lower_col:
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[bb_lower_col],
        mode='lines',
        name='BB Lower',
        line=dict(color='rgba(100, 100, 255, 0.3)'),
        fill='tonexty',
        fillcolor='rgba(150, 150, 200, 0.1)',
        hoverinfo='skip'
    ))

# SMA Lines
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['SMA_14'],
    mode='lines',
    name='SMA 14',
    line=dict(color='#ff7f0e', width=1, dash='dash'),
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['SMA_50'],
    mode='lines',
    name='SMA 50',
    line=dict(color='#2ca02c', width=1, dash='dash'),
    hoverinfo='skip'
))

fig.update_layout(
    height=400,
    template='plotly_dark',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    hovermode='x unified',
    margin=dict(l=10, r=10, t=40, b=10)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# RSI Chart
# ---------------------------------------------------------
col_rsi, col_vol = st.columns(2)

with col_rsi:
    st.subheader("RSI (14)")
    fig_rsi = go.Figure()
    
    fig_rsi.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='#9467bd', width=2),
    ))
    
    # Add Neutral Zone shading
    fig_rsi.add_hrect(
        y0=40, y1=60, 
        line_width=0, 
        fillcolor="rgba(0, 255, 0, 0.15)", 
        annotation_text="Neutral Strategy Zone", 
        annotation_position="top left"
    )
    
    # Add overbought/oversold lines
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Oversold")
    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
    
    fig_rsi.update_layout(
        height=300,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='RSI',
        hovermode='x',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_vol:
    st.subheader("Volatility Trend (IV Rank Proxy)")
    fig_vol = go.Figure()
    
    # Normalize volatility for display purposes to match the IV Rank math
    vol_min = df['Volatility'].min()
    vol_max = df['Volatility'].max()
    historical_iv_rank = (df['Volatility'] - vol_min) / (vol_max - vol_min) * 100
    
    fig_vol.add_trace(go.Scatter(
        x=df.index,
        y=historical_iv_rank,
        mode='lines',
        name='IV Rank (%)',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.3)'
    ))
    
    # Add the 20% floor line
    fig_vol.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Minimum Premium Floor (20%)")
    
    fig_vol.update_layout(
        height=300,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='IV Rank (%)',
        hovermode='x',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Configuration Info
# ---------------------------------------------------------
with st.expander("⚙️ Configuration & Strategy"):
    st.write("""
    ### True Neutral Iron Condor Strategy
    
    **When to Trade:**
    - **Range-Bound Action:** Price must be hovering near the 20-day Bollinger Band midline, signifying the absence of a strong trend.
    - **Neutral Momentum:** RSI must sit squarely in the 40-60 zone. Extreme momentum (>70 or <30) suggests entering a skewed, directional spread instead.
    - **Elevated Volatility (IV Rank > 20%):** Iron Condors are short-premium strategies. Volatility must be elevated to ensure the collateral-to-premium ratio is worth the risk.
    - **Stable Volatility (ATR Check):** While elevated, the volatility should be stable (ATR within 20% of its 10-day average), avoiding erratic blowout moves.
    
    **Strike Selection:**
    - **Short strikes:** Placed symmetrically at ~2.0x ATR from the current price (targeting ~16 Delta / 1 Standard Deviation).
    - **Long wings:** Placed $2.00 further out to cap maximum loss and define buying power reduction.
    
    **Risk Management:**
    - Enter around 30-45 DTE (Days to Expiration).
    - Exit strategy: Close at 50% max profit.
    - Stop loss: Typically if the underlying breaches the short strike of either wing.
    """)
    
    st.json(IC_SETUP)
