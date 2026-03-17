# Complete API & Setup Guide - Every Step Explained

This guide walks you through **every account signup, every API key, every install, and every configuration** needed to run the Oil Quantitative Trading Bot from scratch.

---

## Table of Contents

1. [Overview - What You Need](#1-overview---what-you-need)
2. [Step 1: Interactive Brokers Account](#2-step-1-interactive-brokers-account)
3. [Step 2: Twitter/X API Key](#3-step-2-twitterx-api-key)
4. [Step 3: NewsAPI Key](#4-step-3-newsapi-key)
5. [Step 4: EIA API Key](#5-step-4-eia-api-key)
6. [Step 5: Install Python & Dependencies](#6-step-5-install-python--dependencies)
7. [Step 6: Configure the .env File](#7-step-6-configure-the-env-file)
8. [Step 7: Set Up the Database](#8-step-7-set-up-the-database)
9. [Step 8: Install & Configure TWS/IB Gateway](#9-step-8-install--configure-twsib-gateway)
10. [Step 9: First Run - Paper Trading](#10-step-9-first-run---paper-trading)
11. [Step 10: Run the Dashboard](#11-step-10-run-the-dashboard)
12. [Step 11: Run a Backtest](#12-step-11-run-a-backtest)
13. [Step 12: Going Live (When Ready)](#13-step-12-going-live-when-ready)
14. [What Each API Does & Why You Need It](#14-what-each-api-does--why-you-need-it)
15. [Cost Summary](#15-cost-summary)
16. [FAQ](#16-faq)

---

## 1. Overview - What You Need

| # | Service | Cost | Required? | What It Does |
|---|---------|------|-----------|-------------|
| 1 | Interactive Brokers | Free (paper) / min $0 (live) | **YES** | Executes trades, provides market data |
| 2 | Twitter/X API | Free (Basic tier) | No (but recommended) | Oil sentiment from tweets |
| 3 | NewsAPI | Free (Developer tier) | No (but recommended) | Oil sentiment from news articles |
| 4 | EIA API | Free | No (but recommended) | Weekly petroleum inventory data |
| 5 | Python 3.11+ | Free | **YES** | Runs the bot |
| 6 | PostgreSQL | Free | No | Database (SQLite works by default) |

**Minimum to get started**: Just #1 (IBKR) and #5 (Python). Everything else enhances the bot but isn't required.

---

## 2. Step 1: Interactive Brokers Account

This is the **only required external account**. The bot trades through IBKR.

### 2.1 Create an Account

1. Go to **https://www.interactivebrokers.com**
2. Click **"Open Account"** (top right)
3. Choose **Individual Account**
4. Fill in your personal information:
   - Full legal name
   - Date of birth
   - Address
   - Social Security Number (US) or equivalent
   - Employment information
   - Financial information (net worth, income, trading experience)
5. For **Account Type**, choose:
   - **"IBKR Lite"** - $0 commissions on US stocks/ETFs, $0 account minimum
   - **"IBKR Pro"** - Lower futures commissions ($0.85/contract vs $2.25), best for this bot
6. Enable **futures trading** when asked about trading permissions
   - You need this for CL (WTI Crude) and BZ (Brent Crude) futures
   - They'll ask about your futures experience — answer honestly
7. Fund your account (for live trading later):
   - Minimum $0 for IBKR Lite
   - Recommended: $25,000+ for futures (margin requirements)
   - For **paper trading only**: no funding needed

### 2.2 Enable Paper Trading

Paper trading lets you test with fake money — **do this first**.

1. Log in at **https://www.interactivebrokers.com**
2. Go to **Settings > Account Settings**
3. Look for **"Paper Trading Account"** section
4. Click **"Create Paper Trading Account"**
5. You'll get separate credentials (username starts with a "D" for demo)
6. Write down your **paper trading username and password**

### 2.3 Download TWS or IB Gateway

You need one of these running on your computer for the bot to connect to:

**Option A - Trader Workstation (TWS)** — has a GUI, good for beginners:
1. Go to **https://www.interactivebrokers.com/en/trading/tws.php**
2. Download **"TWS Latest"** for your OS (Windows/Mac)
3. Install it
4. Log in with your **paper trading** credentials

**Option B - IB Gateway** — lightweight, no GUI, better for 24/7 operation:
1. Go to **https://www.interactivebrokers.com/en/trading/ibgateway-stable.php**
2. Download **"IB Gateway Stable"** for your OS
3. Install it
4. Log in with your **paper trading** credentials

### 2.4 Configure API Access

After logging in to TWS or IB Gateway:

1. **In TWS**: Go to **Edit > Global Configuration > API > Settings**
   **In IB Gateway**: Go to **Configure > Settings > API > Settings**

2. Check these boxes:
   - [x] **Enable ActiveX and Socket Clients**
   - [x] **Allow connections from localhost only** (for security)
   - [ ] **Read-Only API** — leave this **UNCHECKED** (bot needs to place orders)

3. Set **Socket port**: `7497` (this is the paper trading port)

4. Under **Trusted IPs**, add: `127.0.0.1`

5. Set **Master API client ID**: `1`

6. Click **Apply** then **OK**

7. **Leave TWS/IB Gateway running** — the bot connects to it

### 2.5 What You Get From IBKR

- Real-time market data for oil futures (CL, BZ)
- Historical price data for backtesting
- Order execution (buy/sell futures and ETFs)
- Portfolio and account information
- **No separate API key needed** — the bot connects via the local socket

---

## 3. Step 2: Twitter/X API Key

**Purpose**: The bot reads tweets about oil (OPEC, crude prices, energy crisis, etc.) and uses AI sentiment analysis (FinBERT) to gauge market sentiment.

### 3.1 Create a Developer Account

1. Go to **https://developer.x.com** (or https://developer.twitter.com)
2. Click **"Sign up for Free Account"** or **"Developer Portal"**
3. Log in with your Twitter/X account (create one if you don't have one)
4. You'll be asked to describe your use case:
   - Select **"Exploring the API"** or **"Academic Research"**
   - Description example: *"Building a quantitative trading research tool that analyzes public oil market sentiment from tweets for academic and personal investment research."*
5. Agree to the Developer Agreement
6. Your developer account is created

### 3.2 Create a Project and App

1. In the Developer Portal, go to **Dashboard**
2. Click **"+ Create Project"**
   - Project Name: `Oil Sentiment Analysis`
   - Use case: `Analyzing public conversation`
   - Description: `Analyzing oil market sentiment from public tweets`
3. Click **"Create App"** within the project
   - App Name: `OilQuantBot` (must be unique across all of Twitter)
4. You'll see your keys — **save the Bearer Token**

### 3.3 Get Your Bearer Token

1. In the Developer Portal, go to your app
2. Click **"Keys and tokens"** tab
3. Under **"Bearer Token"**, click **"Generate"** (or **"Regenerate"**)
4. **Copy the Bearer Token** — it looks like: `AAAAAAAAAAAAAAAAAAAAAxxxxxxxxxxxxxxxxxxxxxxxx`
5. **Save this somewhere safe** — you can't see it again (but you can regenerate)

### 3.4 API Tier Limits

| Tier | Cost | Tweets/month | Good Enough? |
|------|------|-------------|-------------|
| Free | $0 | 500K reads | Yes, for basic use |
| Basic | $100/month | 10K reads + search | Better for real-time |
| Pro | $5,000/month | 1M reads + full search | Overkill |

**Recommendation**: Start with **Free tier**. The bot only pulls tweets every 5 minutes.

### 3.5 If You Can't Get Twitter API Access

The bot works without it — Twitter sentiment just won't be included in the signal. News + EIA data still provide sentiment signals.

---

## 4. Step 3: NewsAPI Key

**Purpose**: The bot searches for oil-related news articles from Reuters, Bloomberg, AP, and hundreds of other sources, then runs FinBERT sentiment analysis on the headlines/descriptions.

### 4.1 Sign Up

1. Go to **https://newsapi.org**
2. Click **"Get API Key"** (top right)
3. Fill in:
   - Name: Your name
   - Email: Your email
   - Password: Choose a password
   - For the question "What are you using NewsAPI for?": Select **"I'm an individual"**
4. Click **"Submit"**
5. Your API key appears immediately on the next page
6. It looks like: `abc123def456ghi789jkl012mno345pq`
7. You'll also get an email with the key

### 4.2 API Limits

| Plan | Cost | Requests/day | Articles Returned | Good Enough? |
|------|------|-------------|------------------|-------------|
| Developer | $0 | 100 | Up to 100/request | Yes for starting out |
| Business | $449/month | 250,000 | Unlimited | For production |

**Recommendation**: **Free (Developer)** plan is fine. The bot makes ~288 requests/day at 5-minute intervals, which exceeds the free tier. Options:
- Increase the `ALT_DATA_INTERVAL_MINUTES` to 15 (reduces to ~96 requests/day) in settings.py
- Or upgrade when ready

### 4.3 What the Bot Searches For

The bot searches these keywords across all news sources:
- "crude oil", "OPEC", "oil price", "petroleum", "energy crisis"
- "oil supply", "$USO", "$XLE", "EIA report", "barrel"
- "oil inventory", "refinery", "pipeline", "sanctions", "SPR release", "Fed rate"

---

## 5. Step 4: EIA API Key

**Purpose**: The U.S. Energy Information Administration publishes weekly petroleum inventory data every Wednesday at 10:30 AM ET. This is one of the most market-moving reports for oil prices. The bot automatically downloads and analyzes:
- Crude oil stocks (how much oil is in storage)
- Crude oil production
- Crude oil imports
- Gasoline stocks
- Distillate stocks

### 5.1 Sign Up

1. Go to **https://www.eia.gov/opendata/register.php**
2. Fill in:
   - Email address
   - First and last name
   - Affiliation (optional): put your name or "Individual"
   - Reason: "Automated petroleum data retrieval for personal investment research"
3. Click **"Register"**
4. Check your email — you'll receive your API key within minutes
5. The key looks like: `aBcDeFgHiJkLmNoPqRsTuVwXyZ123456`

### 5.2 That's It

- **100% free**, no limits that matter for this bot
- The bot only calls it once per week (Wednesdays at 10:30 AM ET)
- No credit card, no paid tiers

---

## 6. Step 5: Install Python & Dependencies

### 6.1 Install Python

**Windows:**
1. Go to **https://www.python.org/downloads/**
2. Download **Python 3.11** or **3.12** (not 3.13 yet — some ML libraries may lag)
3. Run the installer
4. **IMPORTANT**: Check the box **"Add Python to PATH"** at the bottom of the installer
5. Click **"Install Now"**
6. Verify: open Command Prompt and type:
   ```cmd
   python --version
   ```
   Should show `Python 3.11.x` or `3.12.x`

**Mac:**
```bash
# Install Homebrew first (if not installed):
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python:
brew install python@3.11

# Verify:
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 --version
```

### 6.2 Clone the Repository

```bash
git clone https://github.com/FrankieBiz/Quantitative-Oil-Trading-Bot-Project.git
cd Quantitative-Oil-Trading-Bot-Project
```

### 6.3 Create a Virtual Environment

A virtual environment keeps this project's packages separate from your system Python.

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt. **Always activate this before running the bot.**

### 6.4 Install All Dependencies

```bash
pip install -r requirements.txt
```

This installs ~23 packages including XGBoost, PyTorch, Dash, and more. First install takes **10-20 minutes** (PyTorch and Transformers are large downloads, ~2-4 GB total).

**If PyTorch install fails on Windows** (common):
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**If you have an NVIDIA GPU** (optional, speeds up FinBERT sentiment):
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 6.5 Verify Installation

```bash
python -c "import xgboost, dash, transformers, ib_insync; print('All dependencies OK')"
```

If this prints `All dependencies OK`, you're good.

---

## 7. Step 6: Configure the .env File

### 7.1 Create the File

```bash
cp .env.example .env
```

Or on Windows:
```cmd
copy .env.example .env
```

### 7.2 Also Create It in the Config Directory

The bot's settings.py loads from `oil_quant_bot/config/.env`:

```bash
cp .env.example oil_quant_bot/config/.env
```

### 7.3 Edit with Your Keys

Open the `.env` file in any text editor (Notepad, VS Code, nano) and fill in your values:

```env
# Interactive Brokers
IBKR_HOST=127.0.0.1
IBKR_CLIENT_ID=1
LIVE_MODE=False

# Twitter/X API - paste your Bearer Token here
TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAxxxxxxxxxxxxxxxxxxxxxxxx

# NewsAPI - paste your API key here
NEWSAPI_KEY=abc123def456ghi789jkl012mno345pq

# EIA - paste your API key here
EIA_API_KEY=aBcDeFgHiJkLmNoPqRsTuVwXyZ123456

# Database (SQLite is fine for starting)
DATABASE_URL=sqlite:///oil_quant_bot.db

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050

# Risk-Free Rate (current US Treasury rate, update periodically)
RISK_FREE_RATE=0.05

# Logging
LOG_LEVEL=INFO
```

### 7.4 Security Reminder

- **NEVER commit your `.env` file to git** — it contains secrets
- The `.gitignore` already excludes it
- If you accidentally commit it, regenerate all your API keys immediately

---

## 8. Step 7: Set Up the Database

### Option A: SQLite (Easiest — Recommended for Starting)

**Nothing to do.** The bot automatically creates `oil_quant_bot.db` on first run. SQLite is a file-based database — no server needed.

### Option B: PostgreSQL (For Production / Heavy Use)

Only do this if you want a more robust database. Skip this for now if you're just getting started.

**Windows:**
1. Download from **https://www.postgresql.org/download/windows/**
2. Run the installer
3. Set a password for the `postgres` user (remember it!)
4. Keep the default port `5432`
5. Open **pgAdmin** (installed with PostgreSQL) or use Command Prompt:
   ```cmd
   psql -U postgres
   ```
6. Create the database:
   ```sql
   CREATE DATABASE oil_trading_db;
   \q
   ```
7. Update your `.env`:
   ```env
   DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/oil_trading_db
   ```
8. Install the Python driver:
   ```bash
   pip install psycopg2-binary
   ```

**Mac:**
```bash
brew install postgresql@15
brew services start postgresql@15
createdb oil_trading_db
```

**Linux:**
```bash
sudo apt install postgresql
sudo -u postgres createdb oil_trading_db
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'yourpassword';"
```

---

## 9. Step 8: Install & Configure TWS/IB Gateway

This was covered in Step 1, but here's the quick checklist:

- [ ] IBKR account created
- [ ] Paper trading account enabled
- [ ] TWS or IB Gateway downloaded and installed
- [ ] Logged in with **paper trading** credentials
- [ ] API enabled (Edit > Global Configuration > API > Settings)
- [ ] Socket port set to `7497`
- [ ] "Read-Only API" is **unchecked**
- [ ] `127.0.0.1` added to Trusted IPs
- [ ] TWS/IB Gateway is **running** (leave it open)

---

## 10. Step 9: First Run - Paper Trading

### 10.1 Pre-Flight Checklist

- [ ] TWS or IB Gateway is running and logged in
- [ ] Virtual environment is activated (`(.venv)` in terminal)
- [ ] `.env` file is configured with at least `IBKR_HOST` and `LIVE_MODE=False`

### 10.2 Start the Bot

```bash
python -m oil_quant_bot.main
```

### 10.3 What You Should See

```
INFO | AdaptiveLearner initialised | rolling_window=50 | retrain_threshold=30
INFO | BrokerExecutor connecting to IBKR at 127.0.0.1:7497...
INFO | Connected to IBKR (paper mode)
INFO | Fetching historical data for CL, BZ, USO, XLE...
INFO | Computing technical indicators...
INFO | Trading loop started (PAPER mode)
INFO | Waiting for market hours...
```

### 10.4 What Happens Next

1. The bot connects to IBKR and downloads historical oil price data
2. It computes ~80+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. If sentiment APIs are configured, it pulls tweets and news
4. The XGBoost model generates buy/sell/hold signals
5. The risk manager sizes positions using half-Kelly criterion
6. Orders are submitted to IBKR's paper trading engine
7. The adaptive learner monitors performance and adjusts parameters

### 10.5 Important Notes

- **The bot only trades during market hours** (oil futures: Sunday 6pm - Friday 5pm ET, with a daily break 5-6pm ET)
- First run may take a few minutes to download historical data
- The XGBoost model needs data to train — it may not trade immediately
- Check `oil_quant_bot/logs/` for detailed log files

---

## 11. Step 10: Run the Dashboard

Open a **second terminal** (keep the bot running in the first):

```bash
cd Quantitative-Oil-Trading-Bot-Project
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m oil_quant_bot.dashboard.serve
```

Then open **http://localhost:8050** in your browser.

See the [SETUP_GUIDE.md](SETUP_GUIDE.md) for accessing from Mac, iPhone, Android, and remotely.

---

## 12. Step 11: Run a Backtest

Before risking money, test the strategy on historical data:

```bash
python -m oil_quant_bot.backtest.engine
```

This runs a walk-forward backtest from 2019-2024 and shows:
- Total return, Sharpe ratio, max drawdown
- Win rate, profit factor
- Monte Carlo confidence intervals

---

## 13. Step 12: Going Live (When Ready)

**Only do this after at least 2-4 weeks of successful paper trading.**

1. Make sure your IBKR account is funded (recommended $25,000+ for futures margin)
2. Log into TWS/IB Gateway with your **real** (not paper) credentials
3. Change one line in your `.env`:
   ```env
   LIVE_MODE=True
   ```
4. Restart the bot:
   ```bash
   python -m oil_quant_bot.main
   ```
5. The bot will now use port `7496` (live) and trade with real money

**Warnings:**
- Start small — the `MAX_POSITION_FRACTION` is 10% by default
- The bot has risk guards (3% daily loss limit, 12% max drawdown)
- Monitor the dashboard closely for the first few days
- You can stop the bot at any time with `Ctrl+C`

---

## 14. What Each API Does & Why You Need It

### Interactive Brokers (Required)
- **What**: Connects to IBKR's trading platform via a local socket
- **Why**: This is how the bot buys and sells oil futures/ETFs
- **Without it**: The bot cannot trade at all
- **Data it provides**: Real-time prices, historical bars, account balance, positions

### Twitter/X API (Optional)
- **What**: Searches for oil-related tweets and analyzes sentiment using FinBERT AI
- **Why**: Social sentiment can predict short-term price moves (e.g., OPEC announcement tweets)
- **Without it**: Sentiment score loses 20% of its weight (see `SENTIMENT_WEIGHTS` in settings)
- **Data it provides**: Tweet text, author follower count, timestamp

### NewsAPI (Optional)
- **What**: Searches 80,000+ news sources for oil-related articles
- **Why**: Major news (Reuters, Bloomberg) sentiment has the highest weight (45%) in the model
- **Without it**: Sentiment score loses 65% of its weight (major + minor news combined)
- **Data it provides**: Article title, description, source, publication time

### EIA API (Optional)
- **What**: Downloads weekly U.S. petroleum inventory reports
- **Why**: The weekly EIA report is the single most market-moving event for oil prices
- **Without it**: Sentiment score loses 15% weight; the bot misses inventory surprise signals
- **Data it provides**: Crude stocks, production, imports, gasoline stocks, distillate stocks

### Database (Auto-configured)
- **What**: Stores trades, model predictions, performance metrics
- **Why**: Tracks everything for the adaptive learner and dashboard
- **Default**: SQLite (zero setup, file-based)
- **Upgrade**: PostgreSQL for better concurrent access and reliability

---

## 15. Cost Summary

### Free Tier (Everything Works)

| Service | Cost | Notes |
|---------|------|-------|
| IBKR Paper Trading | $0 | Fake money, full functionality |
| Twitter/X Basic API | $0 | 500K tweet reads/month |
| NewsAPI Developer | $0 | 100 requests/day |
| EIA API | $0 | Unlimited, no tiers |
| Python + Libraries | $0 | Open source |
| SQLite | $0 | Built into Python |
| **Total** | **$0** | |

### Live Trading (Minimum)

| Service | Cost | Notes |
|---------|------|-------|
| IBKR Pro Account | $0/month (waived with $100K+) or $10/month | Activity fees may be waived |
| IBKR Commissions | ~$0.85-$2.25/contract | Per futures trade |
| IBKR Market Data | $0-$10/month | Depends on bundle |
| Account Funding | $25,000+ recommended | Futures margin requirements |
| NewsAPI Business | $449/month (optional) | Only if free tier isn't enough |

---

## 16. FAQ

### Do I need all 4 API keys to run the bot?
**No.** Only IBKR is required. The bot gracefully handles missing sentiment APIs — it just won't have sentiment signals, relying on technical indicators and the ML model alone.

### Can I run this on a Mac?
**Yes.** TWS and IB Gateway both have Mac versions. Everything else is Python, which is cross-platform.

### Can I run this on a VPS/cloud server?
**Yes.** Use IB Gateway (headless) on a Linux VPS. You'll need to handle the IBKR login session — look into **IBC** (https://github.com/IbcAlpha/IBC) for automatic TWS/Gateway login.

### How much money do I need?
- **Paper trading**: $0 (fake money)
- **ETF trading (USO, XLE)**: $2,000+ recommended
- **Futures trading (CL, BZ)**: $25,000+ recommended (margin requirements for crude oil futures are ~$5,000-$12,000 per contract)

### Is this legal?
**Yes.** Algorithmic trading through a registered broker (IBKR) is legal in the US and most countries. You're responsible for taxes on any profits.

### What if the bot loses money?
The bot has multiple safety guards:
- **3% daily loss limit** — stops trading for the day
- **12% max drawdown** — halts all trading
- **Half-Kelly position sizing** — conservative bet sizes
- **Sharpe monitoring** — reduces/halts if performance degrades
- But **no guarantee of profits** — all trading carries risk

### How do I stop the bot?
- Press `Ctrl+C` in the terminal
- Or stop the Windows service if using NSSM
- Open orders will remain — check TWS to cancel them manually

### The bot says "No module named X"
You forgot to activate the virtual environment:
```bash
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### IBKR says "Not connected"
1. Is TWS/IB Gateway running?
2. Are you logged in?
3. Did you enable API access?
4. Is the port correct? (7497 for paper, 7496 for live)
5. Check `IBKR_HOST=127.0.0.1` in your `.env`
