# market-scanners
---

```markdown
# Market Scanners & Quantitative Options Suite

An automated, decoupled quantitative trading monorepo that combines high-frequency momentum screening, volatility compression breakouts, and multi-day trend following[cite: 1]. 

The repository hosts both **headless background bots** scheduled via GitHub Actions to deliver real-time push notifications through Telegram, and **interactive Streamlit web dashboards** with multi-pane Plotly charting for visual trade execution[cite: 1, 2].

---

## Architecture & Project Structure

The project employs a decoupled monorepo architecture[cite: 1]. Shared dependencies and environment secrets are managed centrally, while individual bots and dashboards operate independently to prevent cascading failures[cite: 1]:

```text
market-scanners/
├── .github/
│   └── workflows/
│       ├── intraday_0dte.yml      # Triggers 0DTE bot every 30 mins during market hours
│       ├── squeeze_bot.yml        # Triggers TTM Squeeze bot hourly during market hours
│       └── swing_bot.yml          # Triggers Swing bot daily at market close (4:30 PM ET)
├── app_intraday.py                # Streamlit UI for SPY/QQQ 0DTE momentum tracking
├── app_multiday.py                # Streamlit UI for multi-day swing debit spreads
├── app_squeeze.py                 # Streamlit UI for mega-cap TTM Squeeze breakouts
├── intraday_0dte_bot.py           # Headless Python bot for 0DTE intraday momentum
├── squeeze_bot.py                 # Headless Python bot for TTM Squeeze breakouts
├── swing_bot.py                   # Headless Python bot for daily closing swing scans
├── requirements.txt               # Central dependency manifest (Python 3.12 pinned)
└── README.md                      # Project documentation and setup guide

```

---

## Trading Engines & Strategy Logic

### 1. Intraday 0DTE Momentum Scanner

* **Files:** `intraday_0dte_bot.py` | `app_intraday.py` | `.github/workflows/intraday_0dte.yml`
* **Assets:** `SPY`, `QQQ`
* **Data Feed:** Polygon.io REST API (5-minute aggregate candles)
* **Core Logic:**
* **Session-Anchored VWAP:** Tracks cumulative intraday Volume Weighted Average Price reset each trading session.
* **Normalized EMA Spread:** Evaluates separation between 9 EMA and 21 EMA relative to the 14-period Average True Range (ATR):

$$\text{EMA Spread Norm} = \frac{\text{EMA}_9 - \text{EMA}_{21}}{\text{ATR}_{14}}$$


* **VWAP Displacement:** Ensures entries are breaking beyond VWAP without being overextended ($0.20 \le |\text{Dist}_{\text{VWAP}}| \le 1.10$).
* **Institutional Volume (RVOL):** Validates volume expansion with $\text{RVOL} \ge 1.30\times$ against a 20-period moving average baseline.
* **Trend Confirmation:** Enforces $\text{ADX}_{14} \ge 25.0$ with directional movement index alignment ($+\text{DI} > -\text{DI}$ for calls, $-\text{DI} > +\text{DI}$ for puts).


* **Suggested Structure:** Defined-risk 2-point wide directional vertical debit spreads (Buy ATM / Sell OTM).
* **Alert Mode:** **Silent Mode** — bypasses Telegram and remains quiet unless an actionable setup triggers.

---

### 2. Mega-Cap TTM Squeeze Volatility Scanner

* **Files:** `squeeze_bot.py` | `app_squeeze.py` | `.github/workflows/squeeze_bot.yml`
* **Assets:** `SPY`, `QQQ`, `GOOGL`, `NVDA`, `AMZN`
* **Data Feed:** Polygon.io REST API (5-minute bars with 14-day lookback for indicator warmup)
* **Core Logic:**
* **Volatility Compression:** Identifies consolidation when Bollinger Bands (20, 2.0) narrow completely inside Keltner Channels (20, 1.5 ATR) (Squeeze ON / Red Dot).
* **Breakout Trigger:** Alerts on the exact candle where Bollinger Bands expand outside the Keltner Channels (Squeeze FIRED / Green Dot).
* **4-Color Momentum Histogram:** Confirms expansion direction:
* 🟢 **Call Spread:** Squeeze fires with positive, expanding momentum (Light Blue histogram) and $\text{ADX}_{14} \ge 25.0$.
* 🔴 **Put Spread:** Squeeze fires with negative, deepening momentum (Red histogram) and $\text{ADX}_{14} \ge 25.0$.


* **API Pacing:** Incorporates a 12-second sleep between ticker evaluations to stay strictly within Polygon's free-tier rate limit (5 calls/min).


* **Alert Mode:** **Silent Mode** — only triggers when an active squeeze fires.

---

### 3. Daily Multi-Day Swing Screener

* **Files:** `swing_bot.py` | `app_multiday.py` | `.github/workflows/swing_bot.yml`
* **Assets:** `SPY`, `QQQ`, `NVDA`, `GOOGL`, `AAPL`, `AMZN`
* **Data Feed:** Daily candles (1-year history)
* **Core Logic:**
* **Macro Trend Alignment:** Evaluates price relative to the 50-day EMA and requires $20\text{ EMA} > 50\text{ EMA}$ for bullish setups ($20\text{ EMA} < 50\text{ EMA}$ for bearish setups).
* **RSI Range Filter:** Filters out overbought/oversold extremes ($45 \le \text{RSI}_{14} \le 70$ for calls; $30 \le \text{RSI}_{14} \le 55$ for puts).
* **Trend Velocity:** Confirms sustained direction with $\text{ADX}_{14} \ge 20.0$ and DMI dominance.
* **Dynamic ATR Strike Width:** Dynamically sizes vertical spread width based on rounded daily ATR ($\ge 1.0\text{ pt}$).


* **Alert Mode:** **Wrap-Up Mode** — executes unconditionally at 4:30 PM ET to provide a daily market status summary (delivers setups or confirms no triggers).

---

## Automation Schedule (GitHub Actions)

All headless workflows run on ephemeral Ubuntu runners via GitHub Actions. Workflows can also be manually dispatched at any time via the **Actions** tab on GitHub.

| Workflow | File | CRON Schedule | Trigger Time (Eastern) | Frequency |
| --- | --- | --- | --- | --- |
| **0DTE Intraday** | `intraday_0dte.yml` | `*/30 14-20 * * 1-5` | 10:00 AM – 4:00 PM ET | Every 30 minutes (M–F) |
| **TTM Squeeze** | `squeeze_bot.yml` | `0 14-20 * * 1-5` | 10:00 AM – 4:00 PM ET | Hourly at minute 0 (M–F) |
| **Daily Swing** | `swing_bot.yml` | `30 20 * * 1-5` | 4:30 PM ET | Once daily at market close (M–F) |

---

## Environment Secrets & Configuration

To enable automated Telegram notifications and API data ingestion, the following secrets must be registered under **Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions** in your GitHub repository:

| Secret Name | Description | Example / Format |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | HTTP API token generated via Telegram's `@BotFather` | `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ` |
| `TELEGRAM_CHAT_ID` | Numerical user or group chat ID receiving the alerts | `123456789` |
| `POLYGON_API_KEY` | REST API key from Polygon.io for 5-minute aggregations | `aBcDeFgHiJkLmNoPqRsTuVwXyZ123456` |

---

## Interactive Streamlit Dashboards

Each scanner features a dedicated Streamlit dashboard with interactive Plotly visualizations:

### Running Locally

1. **Clone the repository:**
```bash
git clone [https://github.com/vj932012-arch/market-scanners.git](https://github.com/vj932012-arch/market-scanners.git)
cd market-scanners

```


2. **Create and activate a Python 3.12 virtual environment:**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Launch any of the dashboards:**
```bash
# Launch 0DTE Intraday Dashboard
streamlit run app_intraday.py

# Launch TTM Squeeze Dashboard
streamlit run app_squeeze.py

# Launch Multi-Day Swing Dashboard
streamlit run app_multiday.py

```



### Deploying to Streamlit Cloud

Because this repository contains separate UI entry points, you can deploy multiple distinct web apps from the same repository by specifying the file path during setup:

* **App 1:** Point main file path to `app_intraday.py`
* **App 2:** Point main file path to `app_squeeze.py`
* **App 3:** Point main file path to `app_multiday.py`

---

## Technical Notes & Version Pinning

* **Python 3.12 Pinning:** Workflows and cloud environments must run on **Python 3.12** or **3.13**. The underlying dependencies of `pandas-ta` (`numba` and `llvmlite`) do not support Python 3.14 and will cause build failures on unpinned runners.
* **Package Naming:** In `requirements.txt`, the technical analysis library must be spelled with a hyphen (`pandas-ta`), while in Python scripts it is imported with an underscore (`import pandas_ta as ta`).
* **Timezone Localization:** The `tzdata` package is included in `requirements.txt` to eliminate `ZoneInfoNotFoundError` exceptions when localizing UTC server times to `America/New_York`.

---

## Disclaimer

*This software is developed strictly for educational and informational purposes. None of the notifications, formulas, or visualizations constitute financial or investment advice. Options trading involves substantial risk of loss and is not suitable for every investor.*

```

<ElicitationsGroup message="Next steps for your repository:">
  <Elicitation label="Add repo badges for GitHub Actions build status" query="How do I add live workflow status badges to the top of my README.md?"/>
  <Elicitation label="Add a license file (MIT)" query="Give me a standard MIT License file to include in my market-scanners repository."/>
</ElicitationsGroup>

```
