import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import json

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Iron Condor Tracker", page_icon="📊", layout="wide")
st.title("📊 SPY Iron Condor Tracker")
st.caption("Technical analysis for optimal iron condor windows using Bollinger Bands, RSI, and ATR")

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TICKER = "SPY"
LOOKBACK_DAYS = 30
PERIOD_SHORT = 14
PERIOD_LONG = 50

# Iron Condor Parameters
IC_SETUP = {
    "rsi_sell_threshold": 70,      # Upper RSI for Put Spread
    "rsi_buy_threshold": 30,       # Lower RSI for Call Spread
    "atr_multiplier": 1.5,         # For breakout width
    "bb_threshold": 2.0,           # Standard deviations for Bollinger Bands
    "volatility_min": 0.01,        # Minimum IV percentile
    "volatility_max": 0.85,        # Maximum IV percentile (avoid too volatile)
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
    """Identify ideal iron condor setup windows"""
    df = df.copy()
    df['IC_Signal'] = 0  # 0=No Setup, 1=Call Spread, -1=Put Spread
    df['IC_Score'] = 0.0
    
    latest = df.iloc[-1]
    
    # Get RSI
    rsi = latest.get('RSI', np.nan)
    close_price = latest['close']
    atr = latest.get('ATR', 0)
    volatility = latest.get('Volatility', 0)
    
    # Calculate IV Rank (normalized volatility)
    vol_min = df['Volatility'].min()
    vol_max = df['Volatility'].max()
    iv_rank = (volatility - vol_min) / (vol_max - vol_min) if vol_max > vol_min else 0.5
    
    # Scoring system for iron condor setup
    score = 0.0
    reasons = []
    
    # Volatility is moderate (ideal for iron condor - not too high, not too low)
    if IC_SETUP['volatility_min'] <= iv_rank <= IC_SETUP['volatility_max']:
        score += 30
        reasons.append("✓ Volatility in ideal range")
    else:
        reasons.append(f"✗ Volatility outside range (IV Rank: {iv_rank:.2%})")
    
    # Check if price is near middle bands (mean reversion setup)
    bb_cols = [c for c in df.columns if 'BBL' in c or 'BBU' in c or 'BBM' in c]
    if bb_cols:
        bb_upper = df[[c for c in df.columns if 'BBU' in c][0]].iloc[-1] if any('BBU' in c for c in df.columns) else close_price + atr
        bb_lower = df[[c for c in df.columns if 'BBL' in c][0]].iloc[-1] if any('BBL' in c for c in df.columns) else close_price - atr
        bb_mid = (bb_upper + bb_lower) / 2
        
        distance_to_mid = abs(close_price - bb_mid)
        bb_range = bb_upper - bb_lower
        
        if bb_range > 0 and distance_to_mid < bb_range * 0.25:
            score += 25
            reasons.append("✓ Price near middle bands (mean reversion)")
    
    # PUT SPREAD conditions: High RSI (potential pullback)
    if not np.isnan(rsi):
        if rsi > IC_SETUP['rsi_sell_threshold']:
            score += 25
            df.loc[df.index[-1], 'IC_Signal'] = -1  # Put Spread
            reasons.append(f"✓ High RSI ({rsi:.1f}) - Put Spread candidate")
        elif rsi < IC_SETUP['rsi_buy_threshold']:
            score += 25
            df.loc[df.index[-1], 'IC_Signal'] = 1  # Call Spread
            reasons.append(f"✓ Low RSI ({rsi:.1f}) - Call Spread candidate")
        else:
            reasons.append(f"⚪ RSI neutral ({rsi:.1f})")
    
    # ATR check: Moderate but not extreme volatility
    atr_sma = df['ATR'].rolling(window=10).mean().iloc[-1]
    if atr_sma > 0:
        atr_ratio = atr / atr_sma
        if 0.8 <= atr_ratio <= 1.2:
            score += 20
            reasons.append("✓ ATR normalized (stable volatility)")
        else:
            reasons.append(f"⚠ ATR elevated ({atr_ratio:.2f}x avg)")
    
    df.loc[df.index[-1], 'IC_Score'] = score
    
    return df, reasons, score, rsi, iv_rank

def calculate_ic_levels(close_price, atr, signal):
    """Calculate iron condor strike levels"""
    levels = {}
    
    # Using ATR for strikes
    strike_width = atr * IC_SETUP['atr_multiplier']
    
    if signal == 1:  # Call Spread (Bullish)
        levels['short_call'] = close_price + strike_width
        levels['long_call'] = close_price + (strike_width * 2)
        levels['short_put'] = close_price - (strike_width * 0.5)
        levels['long_put'] = close_price - (strike_width * 1.5)
    elif signal == -1:  # Put Spread (Bearish)
        levels['short_put'] = close_price - strike_width
        levels['long_put'] = close_price - (strike_width * 2)
        levels['short_call'] = close_price + (strike_width * 0.5)
        levels['long_call'] = close_price + (strike_width * 1.5)
    
    return levels

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
    st.subheader("🎯 Iron Condor Setup")
    st.metric("Setup Score", f"{score:.0f}/100")
    
    if signal == 1:
        st.success("🟢 CALL SPREAD CANDIDATE")
        st.write("**Setup**: Price near support, low RSI, volatility normalized")
    elif signal == -1:
        st.error("🔴 PUT SPREAD CANDIDATE")
        st.write("**Setup**: Price near resistance, high RSI, volatility normalized")
    else:
        st.info("⚪ NO CLEAR SETUP")
        st.write("**Wait for**: RSI extreme or mean reversion signal")
    
    st.markdown("### Setup Conditions:")
    for reason in reasons:
        st.write(reason)

with col_levels:
    if signal != 0 and ic_levels:
        st.subheader("📍 Suggested Strike Levels")
        
        level_data = {
            "Strike Type": ["Short Call", "Long Call", "Short Put", "Long Put"],
            "Price": [
                ic_levels.get('short_call', 0),
                ic_levels.get('long_call', 0),
                ic_levels.get('short_put', 0),
                ic_levels.get('long_put', 0)
            ]
        }
        level_df = pd.DataFrame(level_data)
        st.dataframe(level_df, use_container_width=True, hide_index=True)
        
        # Max profit width
        call_width = ic_levels.get('short_call', 0) - ic_levels.get('long_call', 0)
        put_width = ic_levels.get('short_put', 0) - ic_levels.get('long_put', 0)
        max_width = max(abs(call_width), abs(put_width))
        
        st.write(f"**Max Profit Width**: ${max_width:.2f}")
        st.write(f"**Max Profit Region**: ${ic_levels.get('long_put', 0):.2f} - ${ic_levels.get('long_call', 0):.2f}")

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
bb_mid_col = [c for c in df.columns if 'BBM' in c][0] if any('BBM' in c for c in df.columns) else None

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
        fill='tozeroy',
        fillcolor='rgba(148, 103, 189, 0.3)'
    ))
    
    # Add overbought/oversold lines
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
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
    st.subheader("Volatility Trend")
    fig_vol = go.Figure()
    
    fig_vol.add_trace(go.Scatter(
        x=df.index,
        y=df['Volatility'] * 100,
        mode='lines',
        name='IV (%)',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.3)'
    ))
    
    fig_vol.update_layout(
        height=300,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        hovermode='x',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Data Table
# ---------------------------------------------------------
st.subheader("📊 Historical Data")
display_cols = ['close', 'RSI', 'ATR', 'Volatility', 'SMA_14', 'SMA_50']
display_df = df[display_cols].tail(20).copy()
display_df.columns = ['Close', 'RSI', 'ATR', 'Volatility', 'SMA 14', 'SMA 50']
display_df = display_df.round(2)

st.dataframe(display_df, use_container_width=True)

# ---------------------------------------------------------
# Configuration Info
# ---------------------------------------------------------
with st.expander("⚙️ Configuration & Strategy"):
    st.write("""
    ### Iron Condor Setup Strategy
    
    **When to Trade:**
    - RSI > 70: Put Spread candidate (potential pullback from resistance)
    - RSI < 30: Call Spread candidate (potential bounce from support)
    - IV Rank 15%-85%: Optimal premium selling range
    
    **Strike Selection:**
    - Short strikes at ±1.0 ATR from current price
    - Long protection at ±2.0 ATR from current price
    - Creates defined risk with ~25% max profit window
    
    **Risk Management:**
    - Position size based on account risk (1-2%)
    - Exit at 50% max profit
    - Stop loss if price breaks long strike
    """)
    
    st.json(IC_SETUP)