# 📊 Iron Condor Tracker & Bot

This module provides a comprehensive Iron Condor options strategy tracker with Streamlit visualization and automated Telegram alerts.

## 🎯 Features

- **Real-time SPY Analysis**: Monitors SPY price with technical indicators
- **Automated Setup Detection**: Identifies ideal windows for iron condor trades
- **Streamlit Dashboard**: Beautiful interactive visualizations
- **Telegram Alerts**: Instant notifications when setups form
- **GitHub Actions Automation**: Scheduled market scans during trading hours

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit Dashboard

```bash
streamlit run app_iron_condor.py
```

The dashboard will open at `http://localhost:8501` with:
- Current SPY price and technicals
- Setup score and signal
- Suggested strike levels
- Interactive price chart with Bollinger Bands
- RSI and volatility indicators

### 3. Setup Telegram Alerts (Local Bot)

#### Get Your Telegram Credentials

1. **Create Telegram Bot**:
   - Chat with [@BotFather](https://t.me/botfather) on Telegram
   - Send `/newbot`
   - Follow prompts to create bot
   - Save the **Bot Token** (example: `123456789:ABCdefGHIjklmnoPQRstuvWXYZ`)

2. **Get Your Chat ID**:
   - Send a message to your new bot
   - Visit `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your `chat.id` in the JSON response
   - Save your **Chat ID** (example: `987654321`)

#### Run Local Bot

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"

python iron_condor_bot.py
```

The bot will:
- Check SPY every 15 minutes
- Send Telegram alerts when setups form
- Display analysis in console

#### Test Telegram Connection

```bash
python iron_condor_bot.py test
```

### 4. Setup GitHub Actions Automation

For automatic scans every 15 minutes during market hours:

#### Add Repository Secrets

1. Go to: **Settings → Secrets and Variables → Actions**
2. Add these secrets:
   - `TELEGRAM_BOT_TOKEN`: Your bot token
   - `TELEGRAM_CHAT_ID`: Your chat ID

#### The Workflow

Defined in `.github/workflows/iron-condor-alerts.yml`:
- Triggers every 15 minutes (market hours only)
- Runs `iron_condor_bot.py once` for single analysis
- Sends Telegram alert if setup found
- Can be manually triggered via **Actions** tab

```yaml
# Runs 9:30 AM - 4:00 PM ET (14:00 - 20:00 UTC), Mon-Fri
- cron: '*/15 14-20 * * 1-5'
```

## 📊 Understanding the Strategy

### Iron Condor Setup Conditions

An ideal setup scores **70+/100** when:

| Condition | Call Spread | Put Spread |
|-----------|-------------|-----------|
| **RSI** | < 30 (Oversold) | > 70 (Overbought) |
| **Price** | Near lower BB | Near upper BB |
| **IV Rank** | 15% - 85% | 15% - 85% |
| **ATR** | Normalized | Normalized |
| **Trend** | Mean reversion setup | Mean reversion setup |

### Strike Level Calculation

Using **Average True Range (ATR)** for position sizing:

```
Strike Width = Current ATR × 1.5

Call Spread (Bullish):
  - Short Call: Price + 1.0 ATR
  - Long Call:  Price + 2.0 ATR
  
Put Spread (Bearish):
  - Short Put:  Price - 1.0 ATR
  - Long Put:   Price - 2.0 ATR

Max Profit Range: Short Put to Short Call (middle section)
```

### Example Alert

```
🔴 PUT SPREAD SIGNAL - Setup Score: 85/100

Current Price: $445.32
RSI: 74.5
ATR: $2.15
IV Rank: 62%

Setup Conditions:
✓ Volatility in ideal range
✓ Price near middle bands
✓ High RSI 74.5 - Put Spread
✓ ATR normalized

Suggested Strikes (ATM ±1.5 STD):
Short Call: $449.85
Long Call:  $454.38
Short Put:  $439.53
Long Put:   $432.98

Max Profit Range: $432.98 - $449.85
```

## 🔧 Configuration

Edit `iron_condor_bot.py`:

```python
IC_SETUP = {
    "rsi_sell_threshold": 70,      # RSI for put spreads
    "rsi_buy_threshold": 30,       # RSI for call spreads
    "atr_multiplier": 1.5,         # Strike width
    "volatility_min": 0.01,        # Min IV Rank
    "volatility_max": 0.85,        # Max IV Rank
    "min_score": 70.0,             # Alert threshold
}

CHECK_INTERVAL_MINUTES = 15        # Check frequency
ALERT_COOLDOWN_MINUTES = 30        # Duplicate alert cooldown
```

## 📈 Indicators Used

| Indicator | Period | Purpose |
|-----------|--------|---------|
| **RSI** | 14 | Overbought/oversold identification |
| **Bollinger Bands** | 20, 2 STD | Mean reversion reversal levels |
| **ATR** | 14 | Volatility-based position sizing |
| **SMA** | 14, 50 | Trend confirmation |
| **IV Rank** | 20-day | Volatility assessment |

## 💡 Trading Rules

### Entry Rules
1. **Score ≥ 70**: Indicates high-quality setup
2. **RSI Extreme**: Oversold (< 30) or Overbought (> 70)
3. **IV Rank**: Between 15% and 85% (avoid extremes)
4. **Price Position**: Within 25% of Bollinger Band middle

### Exit Rules
1. **Take Profit**: At 50% of max profit
2. **Stop Loss**: Price breaks long strike
3. **Time Decay**: Exit 3-5 days before expiration
4. **Max Profit**: When price enters max profit zone

## 🛡️ Risk Management

- **Position Size**: Risk 1-2% of account per trade
- **Width**: Typically $2-4 depending on SPY volatility
- **Margin**: Requires ~25% of max profit as margin
- **Defined Risk**: Both call and put sides protected

## 📱 Telegram Notifications

The bot sends alerts with:
- Signal type and confidence score
- Current technical levels
- Suggested strike prices
- Profit range for the trade
- Timestamp

Customize in `format_alert_message()` function.

## 🔄 Running Multiple Instances

For different markets or timeframes:

1. Create new bot file: `iron_condor_bot_nq.py`
2. Change `TICKER = "QQQ"`
3. Use different Telegram chat IDs
4. Run as separate service

## 📊 Monitoring & Debugging

### View Logs

**Local Bot:**
```bash
python iron_condor_bot.py 2>&1 | tee iron_condor.log
```

**GitHub Actions:**
- Go to repo **Actions** tab
- Click workflow run
- View logs under "Run Iron Condor Analysis"

### Run Single Analysis

```bash
python iron_condor_bot.py once
```

Outputs detailed setup analysis without alerts.

### Test Telegram Connection

```bash
python iron_condor_bot.py test
```

Sends test message to verify credentials.

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| No alerts received | Check bot token and chat ID |
| "Insufficient data" | Wait for market data to populate |
| High false positives | Increase `min_score` threshold |
| Missing indicators | Check yfinance data availability |

## 📚 Related Files

- `app_iron_condor.py` - Streamlit dashboard
- `iron_condor_bot.py` - Bot with Telegram alerts
- `.github/workflows/iron-condor-alerts.yml` - GitHub Actions automation
- `requirements.txt` - Python dependencies

## 🎓 Educational Notes

This tool implements **mean reversion** trading:
- Identifies extremes in RSI (overbought/oversold)
- Confirms with volatility and price position
- Sells premium with defined risk
- Profits from price returning to normal

Perfect for **range-bound markets** and **high IV environments**.

---

**Disclaimer**: This is an educational tool. Always do your own analysis. Iron condors carry defined but significant risk. Paper trade first, understand Greeks, and never risk more than you can afford to lose.
