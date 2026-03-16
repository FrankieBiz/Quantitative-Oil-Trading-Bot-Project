"""
Global configuration and constants for the Oil Quantitative Trading Bot.
All thresholds, parameters, and system constants are centralized here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ==============================================================================
# BROKER CONFIGURATION
# ==============================================================================
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT_PAPER = 7497
IBKR_PORT_LIVE = 7496
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))
LIVE_MODE = os.getenv("LIVE_MODE", "False").lower() == "true"
IBKR_PORT = IBKR_PORT_LIVE if LIVE_MODE else IBKR_PORT_PAPER

# Connection settings
IBKR_TIMEOUT = 30  # seconds
IBKR_RECONNECT_DELAY = 5  # seconds
IBKR_HEARTBEAT_INTERVAL = 30  # seconds (10s was too aggressive for IBKR)
IBKR_MAX_RECONNECT_ATTEMPTS = 10

# ==============================================================================
# INSTRUMENTS
# ==============================================================================
FUTURES_INSTRUMENTS = {
    "CL": {"exchange": "NYMEX", "sec_type": "FUT", "currency": "USD", "description": "WTI Crude Oil"},
    "BZ": {"exchange": "NYMEX", "sec_type": "FUT", "currency": "USD", "description": "Brent Crude Oil"},
}

ETF_INSTRUMENTS = {
    "USO": {"exchange": "ARCA", "sec_type": "STK", "currency": "USD", "description": "United States Oil Fund"},
    "XLE": {"exchange": "ARCA", "sec_type": "STK", "currency": "USD", "description": "Energy Select Sector SPDR"},
}

# Cross-asset instruments for correlation features
CROSS_ASSET_INSTRUMENTS = {
    "DX": {"exchange": "NYBOT", "sec_type": "FUT", "currency": "USD", "description": "US Dollar Index"},
    "ES": {"exchange": "CME", "sec_type": "FUT", "currency": "USD", "description": "E-mini S&P 500"},
}

# Crack spread components
CRACK_SPREAD_INSTRUMENTS = {
    "RB": {"exchange": "NYMEX", "sec_type": "FUT", "currency": "USD", "description": "RBOB Gasoline"},
    "HO": {"exchange": "NYMEX", "sec_type": "FUT", "currency": "USD", "description": "Heating Oil"},
}

ALL_INSTRUMENTS = {**FUTURES_INSTRUMENTS, **ETF_INSTRUMENTS}

# Correlation tracking
CORRELATED_GROUPS = [
    {"CL", "BZ"},  # Oil futures are correlated
]

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
BAR_SIZES = ["1 min", "5 mins", "15 mins", "1 hour", "1 day"]
WARMUP_BARS = 200
HISTORICAL_LOOKBACK = "30 D"  # for warmup (EMA-200 needs ~200 bars to converge)

# Alternative data refresh interval
ALT_DATA_INTERVAL_MINUTES = 5

# Twitter settings
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
TWITTER_KEYWORDS = [
    "crude oil", "OPEC", "oil price", "petroleum", "energy crisis",
    "oil supply", "$USO", "$XLE", "EIA report", "barrel",
]
TWITTER_MAX_RESULTS_PER_KEYWORD = 100
TWITTER_MIN_FOLLOWERS = 50

# News settings
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWS_KEYWORDS = [
    "crude oil", "OPEC", "oil price", "petroleum", "energy crisis",
    "oil supply", "$USO", "$XLE", "EIA report", "barrel",
    "oil inventory", "refinery", "pipeline", "sanctions", "SPR release", "Fed rate",
]

# RSS Feeds
RSS_FEEDS = {
    "reuters": "https://www.reuters.com/rssFeed/energy",
    "eia": "https://www.eia.gov/rss/todayinenergy.xml",
}

# EIA Weekly Report: Wednesday 10:30 AM ET
EIA_REPORT_DAY = "wednesday"
EIA_REPORT_HOUR = 10
EIA_REPORT_MINUTE = 30
EIA_REPORT_TZ = "US/Eastern"
EIA_API_URL = "https://api.eia.gov/v2/petroleum/sum/sndw/data/"

# ==============================================================================
# SENTIMENT ENGINE
# ==============================================================================
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512

# Source credibility weights
SENTIMENT_WEIGHTS = {
    "twitter": 0.20,
    "news_major": 0.45,  # Reuters, Bloomberg, AP
    "news_minor": 0.20,  # Other NewsAPI sources
    "eia": 0.15,
}

MAJOR_NEWS_SOURCES = {"reuters", "bloomberg", "associated press", "ap news", "reuters.com", "bloomberg.com"}

# Recency decay
RECENCY_DECAY_LAMBDA = 0.1  # λ for e^(-λ * age_in_hours)

# Sentiment confidence gating
MIN_SENTIMENT_VOLUME = 10  # minimum records in 48h window for CSS to be reliable
SENTIMENT_REGIME_SHIFT_THRESHOLD = 0.4  # CSS delta magnitude to flag regime shift

# ==============================================================================
# TECHNICAL INDICATORS
# ==============================================================================
# Trend
EMA_PERIODS = [9, 21, 50, 200]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14

# Momentum
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOCH_RSI_PERIOD = 14
STOCH_RSI_K = 3
STOCH_RSI_D = 3
ROC_PERIOD = 10

# Volatility
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
HIST_VOL_PERIOD = 20

# Volume
OBV_MA_PERIOD = 20
VOLUME_MA_PERIOD = 20

# Oil-specific
CORR_ROLLING_WINDOW = 20

# ==============================================================================
# ML SIGNAL MODEL
# ==============================================================================
# Target definition
FORWARD_BARS = 4
LONG_THRESHOLD = 0.004  # +0.4%
SHORT_THRESHOLD = -0.004  # -0.4%

# XGBoost params
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

# Training regime
WALK_FORWARD_TRAIN_DAYS = 60
WALK_FORWARD_TEST_DAYS = 10
RETRAIN_DAY = "sunday"
RETRAIN_HOUR = 0
TRAINING_LOOKBACK_DAYS = 180
MAX_MODEL_VERSIONS = 5

# Signal threshold
CONFIDENCE_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD_MIN = 0.52
CONFIDENCE_THRESHOLD_MAX = 0.70

# ==============================================================================
# REINFORCEMENT LEARNING
# ==============================================================================
RL_ACTION_THRESHOLD = 0.2
RL_RETRAIN_DAY = "sunday"
RL_RETRAIN_HOUR = 2
RL_LOOKBACK_DAYS = 90

PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

# Reward weights
REWARD_DRAWDOWN_PENALTY = 0.5
REWARD_TRANSACTION_COST = 0.1
REWARD_SHARPE_BONUS = 0.2
REWARD_MARGIN_CALL_PENALTY = 0.3

# ==============================================================================
# RISK MANAGEMENT
# ==============================================================================
# Kelly criterion
KELLY_FRACTION = 0.5  # half-Kelly
KELLY_ROLLING_TRADES = 50
MAX_POSITION_FRACTION = 0.10  # 10% of portfolio (conservative for oil futures)

# Stop-loss / Take-profit
STOP_LOSS_ATR_MULT = 2.0
TAKE_PROFIT_ATR_MULT = 3.5
MIN_RISK_REWARD = 1.75

# Trailing stop
TRAILING_ACTIVATION_ATR_MULT = 1.5
TRAILING_DISTANCE_ATR_MULT = 1.0

# Portfolio guards
MAX_DAILY_LOSS_PCT = 0.03  # 3%
MAX_DRAWDOWN_PCT = 0.12  # 12%
MAX_OPEN_POSITIONS = 4
MAX_CORRELATED_POSITIONS = 2
VAR_95_DAILY_LIMIT = 0.02  # 2%
VAR_SIMULATIONS = 10_000

# Sharpe monitoring
SHARPE_ROLLING_DAYS = 30
SHARPE_REDUCTION_THRESHOLD = 0.5
SHARPE_REDUCTION_CONSECUTIVE_DAYS = 5
SHARPE_HALT_THRESHOLD = 0.0
SHARPE_HALT_CONSECUTIVE_DAYS = 10
SHARPE_POSITION_REDUCTION = 0.5  # reduce by 50%
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))  # annualized; update via env

# ==============================================================================
# ORDER EXECUTION
# ==============================================================================
ORDER_THROTTLE_MAX_PER_MINUTE = 20
LIMIT_OFFSET_PCT = 0.0005  # ±0.05%
SLIPPAGE_PCT = 0.0005  # 0.05% for oil futures backtesting (realistic for CL)
COMMISSION_PER_CONTRACT = 2.25  # USD per contract for CL on IBKR

# ==============================================================================
# BACKTESTING
# ==============================================================================
BACKTEST_IN_SAMPLE_START = "2019-01-01"
BACKTEST_IN_SAMPLE_END = "2022-12-31"
BACKTEST_OUT_SAMPLE_START = "2023-01-01"
BACKTEST_OUT_SAMPLE_END = "2024-12-31"
WALK_FORWARD_FOLDS_MIN = 15  # reduced from 20 to fit 4-year in-sample window
MONTE_CARLO_SIMULATIONS = 10_000

# ==============================================================================
# CONCEPT DRIFT DETECTION
# ==============================================================================
DRIFT_PSI_THRESHOLD = 0.25  # Population Stability Index threshold for retraining
DRIFT_CHECK_INTERVAL_BARS = 100  # check drift every N bars
DRIFT_REFERENCE_WINDOW = 500  # number of training samples for reference distribution

# ==============================================================================
# HEALTH CHECKS
# ==============================================================================
HEALTH_CHECK_INTERVAL = 60  # seconds between health checks
MAX_DATA_STALE_SECONDS = 300  # 5 minutes max data staleness
MAX_CONSECUTIVE_ERRORS = 5  # circuit breaker threshold

# ==============================================================================
# ADAPTIVE LEARNING
# ==============================================================================
RETRAIN_TRADE_THRESHOLD = 30
CONFIDENCE_ADJUST_WINDOW = 10
CONFIDENCE_RAISE_THRESHOLD = 0.40  # win rate below this → raise
CONFIDENCE_LOWER_THRESHOLD = 0.65  # win rate above this → lower
CONFIDENCE_RAISE_STEP = 0.02
CONFIDENCE_LOWER_STEP = 0.01

# Regime detection
HURST_TRENDING_THRESHOLD = 0.6
HURST_MEAN_REVERTING_THRESHOLD = 0.4
CHOPPY_POSITION_REDUCTION = 0.4  # reduce by 40%
TRENDING_POSITION_MULTIPLIER = 1.2

# ==============================================================================
# DATABASE
# ==============================================================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///oil_quant_bot.db")

# ==============================================================================
# DASHBOARD
# ==============================================================================
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))
DASHBOARD_UPDATE_INTERVAL = 5_000  # milliseconds

# ==============================================================================
# LOGGING
# ==============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "30 days"

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data_store"
