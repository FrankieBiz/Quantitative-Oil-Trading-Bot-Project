"""
SQLAlchemy ORM models for trade logging, feature store, and system state.
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Boolean,
    Text, JSON, Enum as SAEnum, Index, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool
import enum
from loguru import logger

from config import settings

Base = declarative_base()


class TradeDirection(enum.Enum):
    """Trade direction enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(enum.Enum):
    """Trade lifecycle status."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class OrderType(enum.Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class MarketRegime(enum.Enum):
    """Market regime classification."""
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    CHOPPY = "CHOPPY"


class Trade(Base):
    """Records every trade from entry to exit."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), unique=True, nullable=False, index=True)
    instrument = Column(String(10), nullable=False, index=True)
    direction = Column(SAEnum(TradeDirection), nullable=False)
    status = Column(SAEnum(TradeStatus), default=TradeStatus.PENDING, nullable=False)

    # Entry
    entry_time = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=True)
    entry_quantity = Column(Float, nullable=True)
    entry_order_id = Column(String(64), nullable=True)

    # Exit
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_order_id = Column(String(64), nullable=True)

    # Stop/TP
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    trailing_stop_active = Column(Boolean, default=False)
    trailing_stop_price = Column(Float, nullable=True)

    # Signal info at entry
    signal_strength = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    css_score = Column(Float, nullable=True)
    regime = Column(SAEnum(MarketRegime), nullable=True)

    # Features snapshot at entry (JSON blob)
    features_at_entry = Column(JSON, nullable=True)

    # PnL
    realized_pnl = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_trades_instrument_status", "instrument", "status"),
        Index("idx_trades_entry_time", "entry_time"),
    )

    def net_pnl(self) -> float:
        """Calculate net PnL after commissions and slippage."""
        if self.realized_pnl is None:
            return 0.0
        return self.realized_pnl - (self.commission or 0.0) - (self.slippage or 0.0)


class Order(Base):
    """Records every order submitted to the broker."""
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(64), unique=True, nullable=False, index=True)
    trade_id = Column(String(64), nullable=True, index=True)
    instrument = Column(String(10), nullable=False)
    order_type = Column(SAEnum(OrderType), nullable=False)
    side = Column(String(4), nullable=False)  # BUY or SELL
    quantity = Column(Float, nullable=False)
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)

    # Fill info
    fill_price = Column(Float, nullable=True)
    fill_quantity = Column(Float, nullable=True)
    commission = Column(Float, nullable=True)
    filled = Column(Boolean, default=False)

    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    status = Column(String(20), default="SUBMITTED")

    __table_args__ = (
        Index("idx_orders_submitted", "submitted_at"),
    )


class BarData(Base):
    """OHLCV bar data storage."""
    __tablename__ = "bar_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument = Column(String(10), nullable=False)
    bar_size = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    vwap = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_bar_instrument_size_ts", "instrument", "bar_size", "timestamp", unique=True),
    )


class FeatureSnapshot(Base):
    """Stores computed feature vectors for model training and analysis."""
    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    features = Column(JSON, nullable=False)
    signal_strength = Column(Float, nullable=True)
    prediction = Column(Integer, nullable=True)  # -1, 0, 1

    __table_args__ = (
        Index("idx_features_instrument_ts", "instrument", "timestamp"),
    )


class SentimentRecord(Base):
    """Stores individual sentiment analysis results."""
    __tablename__ = "sentiment_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(20), nullable=False)  # twitter, news_major, news_minor, eia
    source_id = Column(String(256), nullable=True, unique=True)  # dedup key
    text = Column(Text, nullable=False)
    sentiment_score = Column(Float, nullable=False)  # p - n
    positive = Column(Float, nullable=False)
    negative = Column(Float, nullable=False)
    neutral = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    author = Column(String(256), nullable=True)
    url = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_sentiment_source_ts", "source", "timestamp"),
    )


class CompositeSentiment(Base):
    """Stores computed CSS values over time."""
    __tablename__ = "composite_sentiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    css_score = Column(Float, nullable=False)
    css_momentum = Column(Float, nullable=True)
    css_volume = Column(Integer, default=0)  # item count

    __table_args__ = (
        Index("idx_css_ts", "timestamp"),
    )


class ModelVersion(Base):
    """Tracks ML model versions and their performance metrics."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(20), nullable=False)  # xgboost, rl
    version = Column(String(64), nullable=False, unique=True)
    file_path = Column(Text, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    training_samples = Column(Integer, nullable=True)

    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision_long = Column(Float, nullable=True)
    recall_long = Column(Float, nullable=True)
    f1_long = Column(Float, nullable=True)
    precision_short = Column(Float, nullable=True)
    recall_short = Column(Float, nullable=True)
    f1_short = Column(Float, nullable=True)
    feature_importances = Column(JSON, nullable=True)

    is_active = Column(Boolean, default=True)


class PortfolioState(Base):
    """Tracks daily portfolio snapshots."""
    __tablename__ = "portfolio_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl_day = Column(Float, default=0.0)
    open_positions = Column(Integer, default=0)
    daily_return = Column(Float, nullable=True)
    drawdown = Column(Float, default=0.0)
    sharpe_30d = Column(Float, nullable=True)
    regime = Column(SAEnum(MarketRegime), nullable=True)


class SystemAlert(Base):
    """Stores system alerts for the dashboard."""
    __tablename__ = "system_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(10), nullable=False)  # INFO, WARNING, CRITICAL
    category = Column(String(30), nullable=False)  # drawdown, regime, model, connection
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False)


# ==============================================================================
# Database Engine & Session Factory
# ==============================================================================

_engine = None
_SessionFactory = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        Base.metadata.create_all(_engine)
        logger.info(f"Database engine created: {settings.DATABASE_URL}")
    return _engine


def get_session() -> Session:
    """Get a new database session."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


def init_db():
    """Initialize the database, creating all tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database initialized successfully.")
    return engine
