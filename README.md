# Quantitative Oil Trading Bot

A production-grade quantitative trading bot for crude oil (WTI/Brent) futures. It combines XGBoost signal generation, adaptive learning, risk management, and live execution via a broker API.

---

## Project Structure

```
Quantitative-Oil-Trading-Bot-Project/
└── oil_quant_bot/
    ├── main.py               # Entry point – starts the live trading loop
    ├── config/               # Settings (env vars, constants)
    ├── data/                 # Market data fetching & feature engineering
    ├── ml/                   # XGBoost model training & inference
    ├── signals/              # Signal generation pipeline
    ├── learning/
    │   └── adaptive.py       # Adaptive learning loop (see below)
    ├── risk/                 # Position sizing, stop-loss, Kelly criterion
    ├── execution/            # Order routing & broker integration
    ├── backtest/             # Walk-forward backtesting engine
    ├── dashboard/            # FastAPI dashboard & metrics endpoints
    ├── db/                   # SQLAlchemy models & session management
    └── tests/                # Full pytest test suite
```

---

## Prerequisites

- Python 3.11+
- PostgreSQL (or SQLite for local dev)
- A supported broker account (IBKR / Alpaca / simulated)

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/FrankieBiz/Quantitative-Oil-Trading-Bot-Project.git
cd Quantitative-Oil-Trading-Bot-Project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r oil_quant_bot/requirements.txt

# 4. Configure environment variables
cp .env.example .env           # edit .env with your credentials
```

### Key `.env` variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `BROKER_API_KEY` | Broker API key |
| `BROKER_API_SECRET` | Broker API secret |
| `CONFIDENCE_THRESHOLD` | Minimum signal confidence to trade (default `0.60`) |
| `KELLY_FRACTION` | Kelly fraction for position sizing (default `0.25`) |
| `KELLY_ROLLING_TRADES` | Rolling window size for statistics (default `50`) |
| `RETRAIN_TRADE_THRESHOLD` | Trades before model retraining (default `20`) |

---

## Running the Bot

```bash
# Live trading
python -m oil_quant_bot.main

# Backtest mode
python -m oil_quant_bot.backtest.runner --start 2023-01-01 --end 2024-01-01

# Dashboard (FastAPI)
uvicorn oil_quant_bot.dashboard.app:app --reload --port 8000
```

---

## Tests

```bash
pytest oil_quant_bot/tests/ -v
```

---

## Adaptive Learning Module (`learning/adaptive.py`)

The `AdaptiveLearner` class is the core of the continuous learning loop. It monitors live trade performance and dynamically adjusts the bot's behaviour.

### Features

#### 1. Rolling Statistics — `get_rolling_stats()`

Returns current in-memory performance metrics:

```python
stats = learner.get_rolling_stats()
# {
#   "win_rate": 0.54,
#   "avg_win": 320.0,
#   "avg_loss": 210.0,
#   "kelly_fraction": 0.12
# }
```

#### 2. Regime Detection with Hurst Smoothing & Hysteresis — `detect_regime(prices)`

Detects whether the market is **TRENDING**, **MEAN_REVERTING**, or **RANDOM_WALK** using the Hurst exponent and ADX.

- **Smoothing**: The last 5 raw Hurst values are stored in a `deque`; the **median** is used to prevent noisy flips.
- **Hysteresis**: A 0.05 buffer prevents rapid regime switching. For example, if currently `TRENDING` (H > 0.6), the regime won't switch until the smoothed Hurst drops below **0.55**.

```python
regime = learner.detect_regime(price_series)
# MarketRegime.TRENDING | MarketRegime.MEAN_REVERTING | MarketRegime.RANDOM_WALK
```

#### 3. Dynamic Confidence Threshold — `get_current_threshold()`

Fetches and stores the current dynamically adjusted confidence threshold:

```python
threshold = learner.get_current_threshold()
# e.g. 0.63  (adjusted up after a losing streak)
```

The threshold is updated by `adjust_confidence_threshold()` based on recent win rate vs. the target.

#### 4. OHLC-Based ADX — `_calculate_adx_from_ohlc(df, period=14)`

A more accurate ADX calculation that uses full OHLC data via `pandas_ta`:

```python
adx = learner._calculate_adx_from_ohlc(ohlc_df, period=14)
# Falls back to close-only _calculate_adx() if pandas_ta is unavailable
```

Input DataFrame must have columns: `high`, `low`, `close`.

### Full Usage Example

```python
from oil_quant_bot.learning.adaptive import AdaptiveLearner
import pandas as pd

learner = AdaptiveLearner()

# Log a completed trade
learner.log_trade(trade_record)

# Check rolling stats
print(learner.get_rolling_stats())

# Detect market regime
prices = pd.Series([...])  # recent close prices
regime = learner.detect_regime(prices)
print(f"Current regime: {regime}")

# Get dynamic confidence threshold
threshold = learner.get_current_threshold()
print(f"Signal threshold: {threshold:.2f}")

# Get regime-based position multipliers
multipliers = learner.get_regime_multipliers()
print(f"Position size multiplier: {multipliers['position_size']}")
```

---

## Architecture Overview

```
Market Data → Feature Engineering → XGBoost Model
                                          ↓
                                   Signal Generator
                                          ↓
                              AdaptiveLearner.detect_regime()
                                          ↓
                              Risk Manager (Kelly sizing)
                                          ↓
                                   Order Execution
                                          ↓
                              AdaptiveLearner.log_trade()
                                   (closes the loop)
```

---

## Branch

Active development is on: `claude/oil-trading-bot-HeYuw`
