import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="TTM Squeeze Scanner", page_icon="💥", layout="wide")
st.title("💥 Mega-Cap TTM Squeeze Scanner")
st.caption("Detects explosive volatility breakouts when Bollinger Bands narrow inside Keltner Channels.")

# ---------------------------------------------------------
# Data Engine
# ---------------------------------------------------------
# Defaulting to high-beta/mega-cap tech and your core watchlist
DEFAULT_TICKERS = ["SPY", "QQQ", "GOOGL", "NVDA", "AMZN"]

@st.cache_data(ttl=300) # 5-minute refresh to avoid IP bans
def fetch_and_calculate_squeeze(ticker: str):
    df = yf.download(ticker, period="10d", interval="5m", progress=False)
    
    if df.empty:
        return None
        
    # Flatten multi-index columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # Calculate TTM Squeeze using pandas_ta
    # Returns Squeeze Status (1=On, 0=Off) and Momentum Histogram
    squeeze_df = df.ta.squeeze(lazybear=False, detailed=True)
    
    if squeeze_df is not None:
        df = pd.concat([df, squeeze_df], axis=1)
        
        # Standard pandas_ta column names for Squeeze
        sqz_on_col = [c for c in df.columns if "SQZ_ON" in c][0]
        sqz_off_col = [c for c in df.columns if "SQZ_OFF" in c][0]
        hist_col = [c for c in df.columns if "SQZ" in c and "ON" not in c and "OFF" not in c and "NO" not in c][0]
        
        # Standardize column names for UI mapping
        df["SQZ_ON"] = df[sqz_on_col]
        df["SQZ_OFF"] = df[sqz_off_col]
        df["HISTOGRAM"] = df[hist_col]
        
        # Strategy Logic: Find exactly when it transitions from ON to OFF (Red Dot -> Green Dot)
        df["squeeze_firing"] = (df["SQZ_OFF"] == 1) & (df["SQZ_ON"].shift(1) == 1)
        
        # Strict Thresholds: Check for momentum acceleration (Light Blue / Red)
        df["hist_light_blue"] = (df["HISTOGRAM"] > 0) & (df["HISTOGRAM"] > df["HISTOGRAM"].shift(1))
        df["hist_red"] = (df["HISTOGRAM"] < 0) & (df["HISTOGRAM"] < df["HISTOGRAM"].shift(1))
        
        # Determine Direction based strictly on accelerated momentum
        df["signal"] = 0
        df.loc[df["squeeze_firing"] & df["hist_light_blue"], "signal"] = 1  # Call Spread
        df.loc[df["squeeze_firing"] & df["hist_red"], "signal"] = -1 # Put Spread
        
    return df

# ---------------------------------------------------------
# UI & Display
# ---------------------------------------------------------
if st.button("🔄 Scan Market"):
    st.cache_data.clear()

cols = st.columns(len(DEFAULT_TICKERS))

for col, ticker in zip(cols, DEFAULT_TICKERS):
    df = fetch_and_calculate_squeeze(ticker)
    
    with col:
        st.subheader(ticker)
        if df is None or "HISTOGRAM" not in df.columns:
            st.error("Data error or insufficient history")
            continue
            
        latest = df.iloc[-1]
        current_price = latest["close"]
        signal = latest.get("signal", 0)
        
        st.metric("Last Price", f"${current_price:.2f}")
        
        if signal == 1:
             st.success("🟢 SQUEEZE FIRED: CALL SPREAD")
             st.write("Momentum is expanding upward (Light Blue).")
        elif signal == -1:
             st.error("🔴 SQUEEZE FIRED: PUT SPREAD")
             st.write("Momentum is expanding downward (Red).")
        else:
             # Check if it's currently compressing
             if latest["SQZ_ON"] == 1:
                 st.warning("🟡 SQUEEZE COMPRESSING")
                 st.write("Wait for the breakout.")
             else:
                 st.info("⚪ NO ACTIVE SQUEEZE")
                 st.write("Volatility is normal.")

        # --- Dynamic Plotly Chart ---
        st.markdown("---")
        
        # 1. Map the 4-Color Histogram
        histogram_colors = []
        for i in range(len(df)):
            val = df['HISTOGRAM'].iloc[i]
            prev_val = df['HISTOGRAM'].iloc[i-1] if i > 0 else 0
            
            if val > 0:
                color = "#00b0ff" if val > prev_val else "#0d47a1"  # Light Blue (Expansion) vs Dark Blue (Fade)
            else:
                color = "#ff5252" if val < prev_val else "#ffeb3b"  # Red (Expansion) vs Yellow (Fade)
            histogram_colors.append(color)

        fig = go.Figure()

        # 2. Add Histogram Bars
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['HISTOGRAM'], 
                marker_color=histogram_colors,
                name="Momentum"
            )
        )

        # 3. Add Volatility Dots (Zero Line)
        squeeze_on = df[df['SQZ_ON'] == 1]
        squeeze_fired = df[df['SQZ_OFF'] == 1]

        fig.add_trace(
            go.Scatter(
                x=squeeze_on.index, y=[0]*len(squeeze_on),
                mode="markers", marker=dict(color="#ff5252", size=4),
                name="Squeeze ON (Red)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=squeeze_fired.index, y=[0]*len(squeeze_fired),
                mode="markers", marker=dict(color="#00e676", size=6, symbol="diamond"),
                name="Squeeze FIRED (Green)"
            )
        )

        # 4. Format Chart for narrow columns
        fig.update_layout(
            height=250, 
            margin=dict(l=5, r=5, t=30, b=5), 
            xaxis_rangeslider_visible=False, 
            template="plotly_dark", 
            showlegend=False,
            title=dict(text="TTM Momentum", font=dict(size=13))
        )
        
        # Unlock axes to allow mobile panning/zooming if needed
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)

        st.plotly_chart(
            fig, 
            use_container_width=True, 
            config={'displayModeBar': False} # Hides the clunky top-right menu for clean UI
        )
