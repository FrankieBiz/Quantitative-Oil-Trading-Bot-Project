# Oil Quant Bot - Complete Setup Guide

Step-by-step instructions for setting up the trading bot on your **Windows PC** (the main machine) and accessing the dashboard from your **Mac**, **iPhone**, or **Android** phone.

---

## Table of Contents

1. [Windows PC Setup (Main Machine)](#1-windows-pc-setup-main-machine)
2. [Interactive Brokers Setup](#2-interactive-brokers-setup)
3. [API Keys & Environment Variables](#3-api-keys--environment-variables)
4. [Database Setup](#4-database-setup)
5. [Running the Bot](#5-running-the-bot)
6. [Launching the Dashboard](#6-launching-the-dashboard)
7. [Accessing from Mac](#7-accessing-from-mac)
8. [Accessing from iPhone](#8-accessing-from-iphone)
9. [Accessing from Android](#9-accessing-from-android)
10. [Remote Access (Different Network)](#10-remote-access-different-network)
11. [Firewall & Network Configuration](#11-firewall--network-configuration)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Windows PC Setup (Main Machine)

This is the computer that runs the trading bot 24/7. Everything else just connects to its dashboard.

### 1.1 Install Python 3.11+

1. Download Python from https://www.python.org/downloads/
2. **Check "Add Python to PATH"** during installation
3. Verify in Command Prompt:
   ```cmd
   python --version
   ```
   Should show `Python 3.11.x` or higher.

### 1.2 Install Git

1. Download from https://git-scm.com/download/win
2. Install with default settings
3. Verify:
   ```cmd
   git --version
   ```

### 1.3 Clone the Repository

Open Command Prompt or PowerShell:

```cmd
cd C:\Users\YourName\Projects
git clone https://github.com/FrankieBiz/Quantitative-Oil-Trading-Bot-Project.git
cd Quantitative-Oil-Trading-Bot-Project
```

### 1.4 Create a Virtual Environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

> You should see `(.venv)` at the start of your command prompt. **Always activate this before running the bot.**

### 1.5 Install Dependencies

```cmd
pip install -r oil_quant_bot\requirements.txt
```

This installs everything: pandas, XGBoost, PyTorch, Dash, IBKR connector, etc. First install may take 10-15 minutes (PyTorch is large).

**If torch install fails** (common on Windows), install it separately first:
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r oil_quant_bot\requirements.txt
```

---

## 2. Interactive Brokers Setup

The bot trades through Interactive Brokers (IBKR). You need either TWS or IB Gateway running.

### 2.1 Install TWS or IB Gateway

- **TWS (Trader Workstation)**: https://www.interactivebrokers.com/en/trading/tws.php
- **IB Gateway** (lighter, headless): https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

> IB Gateway is recommended for 24/7 operation — uses less memory.

### 2.2 Configure API Access

In TWS or IB Gateway:

1. Go to **Edit > Global Configuration > API > Settings**
2. Check **"Enable ActiveX and Socket Clients"**
3. Set **Socket port**:
   - Paper trading: `7497` (default)
   - Live trading: `7496` (default)
4. Add `127.0.0.1` to **Trusted IPs**
5. Uncheck **"Read-Only API"** (the bot needs to place orders)

### 2.3 Paper Trading First

**Always start with paper trading:**

1. Log into TWS/Gateway with your **paper trading** account
2. The bot defaults to paper mode (`LIVE_MODE=False`)
3. Run for at least 2 weeks on paper before going live

---

## 3. API Keys & Environment Variables

### 3.1 Create the .env File

Create a file at `oil_quant_bot/config/.env`:

```env
# ============================================
# BROKER
# ============================================
IBKR_HOST=127.0.0.1
IBKR_CLIENT_ID=1
LIVE_MODE=False

# ============================================
# DATABASE
# ============================================
# SQLite (simple, good for single machine):
DATABASE_URL=sqlite:///oil_quant_bot.db

# PostgreSQL (optional, for production):
# DATABASE_URL=postgresql://user:password@localhost:5432/oil_quant_bot

# ============================================
# SENTIMENT DATA SOURCES
# ============================================
# Twitter API (get from https://developer.twitter.com)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# NewsAPI (get from https://newsapi.org)
NEWSAPI_KEY=your_newsapi_key_here

# ============================================
# DASHBOARD
# ============================================
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050

# ============================================
# RISK
# ============================================
RISK_FREE_RATE=0.05

# ============================================
# LOGGING
# ============================================
LOG_LEVEL=INFO
```

### 3.2 Get API Keys

| Service | URL | Free Tier |
|---------|-----|-----------|
| Twitter/X API | https://developer.twitter.com | Yes (Basic) |
| NewsAPI | https://newsapi.org | Yes (100 req/day) |
| IBKR | https://www.interactivebrokers.com | Paper trading is free |

> The bot works without Twitter/NewsAPI keys — sentiment features will just be disabled.

---

## 4. Database Setup

### Option A: SQLite (Recommended for Getting Started)

No setup needed. The bot creates `oil_quant_bot.db` automatically on first run.

### Option B: PostgreSQL (For Production)

1. Install PostgreSQL: https://www.postgresql.org/download/windows/
2. Create a database:
   ```sql
   CREATE DATABASE oil_quant_bot;
   ```
3. Update `.env`:
   ```env
   DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/oil_quant_bot
   ```

---

## 5. Running the Bot

### 5.1 Start the Trading Bot

```cmd
cd C:\Users\YourName\Projects\Quantitative-Oil-Trading-Bot-Project
.venv\Scripts\activate
python -m oil_quant_bot.main
```

The terminal will show:
```
INFO | AdaptiveLearner initialised | rolling_window=50 | retrain_threshold=30
INFO | BrokerExecutor connecting to IBKR at 127.0.0.1:7497...
INFO | Trading loop started (PAPER mode)
```

### 5.2 Run a Backtest (Optional)

Before live trading, validate the strategy on historical data:

```cmd
python -m oil_quant_bot.backtest.runner --start 2023-01-01 --end 2024-12-31
```

### 5.3 Run as a Background Service (Optional)

To keep the bot running 24/7 even when you close the terminal:

**Option 1 — NSSM (recommended for Windows):**
1. Download NSSM: https://nssm.cc/download
2. Install as a service:
   ```cmd
   nssm install OilQuantBot "C:\Users\YourName\Projects\Quantitative-Oil-Trading-Bot-Project\.venv\Scripts\python.exe" "-m oil_quant_bot.main"
   nssm set OilQuantBot AppDirectory "C:\Users\YourName\Projects\Quantitative-Oil-Trading-Bot-Project"
   nssm start OilQuantBot
   ```

**Option 2 — Task Scheduler:**
1. Open Task Scheduler
2. Create Basic Task > "OilQuantBot"
3. Trigger: "At startup"
4. Action: Start a program
   - Program: `C:\Users\YourName\Projects\...\\.venv\Scripts\python.exe`
   - Arguments: `-m oil_quant_bot.main`
   - Start in: `C:\Users\YourName\Projects\Quantitative-Oil-Trading-Bot-Project`

---

## 6. Launching the Dashboard

### 6.1 Start the Dashboard Server

Open a **second** Command Prompt window (keep the bot running in the first):

```cmd
cd C:\Users\YourName\Projects\Quantitative-Oil-Trading-Bot-Project
.venv\Scripts\activate
python -m oil_quant_bot.dashboard.serve
```

You'll see:
```
============================================================
  Oil Quant Bot Dashboard
============================================================

  Local:   http://localhost:8050
  Network: http://192.168.1.42:8050

  Access from your phone or Mac at the Network URL above.
  (Devices must be on the same WiFi network)

============================================================
```

### 6.2 Open on Your Windows PC

Open a browser and go to: **http://localhost:8050**

You'll see the 4-tab dashboard:
- **Overview** — equity curve, positions, trades, sentiment, alerts
- **Performance Analytics** — KPIs, P&L charts, drawdown, win/loss stats
- **Trade Log** — searchable trade history, CSV export
- **System Logs** — real-time log viewer

---

## 7. Accessing from Mac

**Requirement:** Mac and Windows PC must be on the **same WiFi network**.

### 7.1 Find Your Windows PC's IP Address

On your Windows PC, open Command Prompt:
```cmd
ipconfig
```
Look for **"IPv4 Address"** under your WiFi adapter. Example: `192.168.1.42`

### 7.2 Open the Dashboard

On your Mac, open Safari or Chrome and go to:

```
http://192.168.1.42:8050
```

Replace `192.168.1.42` with your actual Windows PC IP.

### 7.3 Bookmark It

- **Safari**: Cmd+D to bookmark
- **Chrome**: Click the star icon in the address bar

That's it. The dashboard auto-refreshes every 5 seconds.

---

## 8. Accessing from iPhone

**Requirement:** iPhone and Windows PC must be on the **same WiFi network**.

### 8.1 Open the Dashboard

1. Open **Safari** on your iPhone
2. Go to `http://192.168.1.42:8050` (use your PC's IP)
3. The dashboard loads in a mobile-optimized layout

### 8.2 Add to Home Screen (Recommended)

This makes it feel like a native app:

1. In Safari, tap the **Share button** (square with arrow)
2. Scroll down and tap **"Add to Home Screen"**
3. Name it **"OilBot"** (or anything you like)
4. Tap **Add**

Now you have an app icon on your home screen that opens the dashboard fullscreen — no browser bars, dark status bar, feels like a real app.

### 8.3 iPhone Tips

- The dashboard is touch-optimized with 44px minimum tap targets
- Tables scroll horizontally on small screens
- Rotate to landscape for better chart viewing
- Pull down to refresh (Safari)

---

## 9. Accessing from Android

**Requirement:** Phone and Windows PC must be on the **same WiFi network**.

### 9.1 Open the Dashboard

1. Open **Chrome** on your Android phone
2. Go to `http://192.168.1.42:8050` (use your PC's IP)

### 9.2 Add to Home Screen

1. Tap the **three-dot menu** (top right)
2. Tap **"Add to Home screen"**
3. Name it **"OilBot"**
4. Tap **Add**

Now it appears as an app icon on your home screen.

---

## 10. Remote Access (Different Network)

When you're away from home (different WiFi, cellular data), use a **Cloudflare Tunnel** for free, secure HTTPS access.

### 10.1 Install Cloudflared on Windows

Open PowerShell as Administrator:

```powershell
winget install Cloudflare.cloudflared
```

Or download directly: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/

### 10.2 Launch Dashboard with Tunnel

```cmd
cd C:\Users\YourName\Projects\Quantitative-Oil-Trading-Bot-Project
.venv\Scripts\activate
python -m oil_quant_bot.dashboard.serve --tunnel
```

The terminal will show something like:
```
============================================================
  Oil Quant Bot Dashboard
============================================================

  Local:   http://localhost:8050
  Network: http://192.168.1.42:8050

  Starting Cloudflare tunnel for remote access...

============================================================

  REMOTE URL: https://random-words-here.trycloudflare.com
```

### 10.3 Access from Anywhere

Open that `https://....trycloudflare.com` URL on **any device, anywhere**:
- Your Mac at a coffee shop
- Your iPhone on cellular data
- Any browser on any network

### 10.4 Important Notes

- The tunnel URL changes every time you restart — copy it fresh each time
- For a **permanent URL**, create a free Cloudflare account and set up a named tunnel
- The tunnel is HTTPS encrypted end-to-end
- No ports need to be opened on your router
- No account needed for quick tunnels

---

## 11. Firewall & Network Configuration

### 11.1 Windows Firewall (Required for Local Network Access)

If Mac/phone can't connect, allow Python through the firewall:

1. Open **Windows Defender Firewall**
2. Click **"Allow an app through firewall"**
3. Click **"Change settings"** then **"Allow another app"**
4. Browse to: `C:\Users\YourName\Projects\...\\.venv\Scripts\python.exe`
5. Check both **Private** and **Public**
6. Click **OK**

Or via PowerShell (run as Admin):
```powershell
New-NetFirewallRule -DisplayName "Oil Quant Bot Dashboard" -Direction Inbound -Protocol TCP -LocalPort 8050 -Action Allow
```

### 11.2 Router Settings

For **local network access** (same WiFi): No router changes needed.

For **remote access**: Use Cloudflare Tunnel (Section 10) — it's safer and easier than port forwarding.

---

## 12. Troubleshooting

### "Module not found" errors

Make sure you activated the virtual environment:
```cmd
.venv\Scripts\activate
```

### Dashboard won't load on phone/Mac

1. Check both devices are on the same WiFi
2. Verify the IP: run `ipconfig` on Windows
3. Check Windows Firewall (Section 11.1)
4. Try pinging from Mac: `ping 192.168.1.42`

### IBKR connection failed

1. Make sure TWS or IB Gateway is running
2. Check API settings are correct (Section 2.2)
3. Verify port: paper = `7497`, live = `7496`
4. Check `IBKR_HOST=127.0.0.1` in `.env`

### Database errors

If using SQLite, delete the DB and restart:
```cmd
del oil_quant_bot.db
python -m oil_quant_bot.main
```

### Bot crashes overnight

Use NSSM or Task Scheduler (Section 5.3) to auto-restart.

### Cloudflare tunnel not working

1. Verify `cloudflared` is installed: `cloudflared --version`
2. Check your internet connection
3. Try running the tunnel manually:
   ```cmd
   cloudflared tunnel --url http://localhost:8050
   ```

### Dashboard shows "No data"

This is normal when first starting. The bot needs to:
1. Connect to IBKR and fetch market data
2. Generate feature snapshots
3. Make predictions and open trades
4. Close trades to populate performance stats

Give it a few hours of market activity (or run a backtest first).

---

## Quick Reference Card

| Action | Command |
|--------|---------|
| Activate venv | `.venv\Scripts\activate` |
| Start bot | `python -m oil_quant_bot.main` |
| Start dashboard (local) | `python -m oil_quant_bot.dashboard.serve` |
| Start dashboard (remote) | `python -m oil_quant_bot.dashboard.serve --tunnel` |
| Run backtest | `python -m oil_quant_bot.backtest.runner --start 2023-01-01 --end 2024-12-31` |
| Run tests | `pytest oil_quant_bot/tests/ -v` |
| Check PC IP | `ipconfig` (look for IPv4 Address) |

| Device | URL |
|--------|-----|
| Windows PC (local) | `http://localhost:8050` |
| Mac / Phone (same WiFi) | `http://<PC-IP>:8050` |
| Anywhere (remote) | `https://<tunnel-url>.trycloudflare.com` |
