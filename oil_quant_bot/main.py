"""
Oil Quantitative Trading Bot - Main Orchestrator

Startup sequence:
1. Load config, connect to IBKR (paper mode)
2. Warm up market data (fetch 200 bars history per instrument)
3. Load latest saved ML models (or train fresh if none exist)
4. Start async loops:
   - Market data stream (continuous)
   - Sentiment data pull (every 5 min)
   - Signal generation (every bar close)
   - Risk check (before every order)
   - Dashboard server
5. Run walk-forward backtest on startup, log results
6. Enter live loop
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from db.models import (
    init_db, get_session, Trade, TradeDirection, TradeStatus,
    PortfolioState, SystemAlert, MarketRegime,
)
from data.market_feed import MarketDataFeed
from data.twitter_feed import TwitterFeed
from data.news_feed import NewsFeed
from data.reddit_feed import RedditFeed
from data.eia_scraper import EIAScraper
from signals.sentiment import SentimentEngine
from signals.technicals import TechnicalEngine
from ml.features import FeatureEngineer
from ml.signal_model import SignalModel
from ml.rl_agent import RLPositionSizer
from risk.manager import RiskManager
from execution.broker import BrokerExecutor
from backtest.engine import BacktestEngine
from learning.adaptive import AdaptiveLearner
from dashboard.app import DashboardApp


class OilQuantBot:
    """Main orchestrator for the Oil Quantitative Trading Bot."""

    def __init__(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("OIL QUANTITATIVE TRADING BOT - INITIALIZING")
        logger.info(f"Mode: {'LIVE' if settings.LIVE_MODE else 'PAPER'}")
        logger.info(f"IBKR Port: {settings.IBKR_PORT}")
        logger.info("=" * 60)

        # Initialize database
        init_db()

        # Initialize components
        self.market_feed = MarketDataFeed()
        self.twitter_feed = TwitterFeed()
        self.reddit_feed = RedditFeed()
        self.news_feed = NewsFeed()
        self.eia_scraper = EIAScraper()
        self.sentiment_engine = SentimentEngine()
        self.technical_engine = TechnicalEngine()
        self.feature_engineer = FeatureEngineer()
        self.signal_model = SignalModel()
        self.rl_sizer = RLPositionSizer()
        self.risk_manager = RiskManager(portfolio_value=100_000.0)  # default; updated on connect
        self.broker = BrokerExecutor()
        self.backtest_engine = BacktestEngine()
        self.adaptive_learner = AdaptiveLearner()
        self.dashboard = DashboardApp()

        # Scheduler
        self.scheduler = AsyncIOScheduler()

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure loguru logging."""
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            settings.LOG_DIR / "bot_{time}.log",
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            level=settings.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        )
        logger.info("Logging configured.")

    async def startup(self):
        """Execute the startup sequence."""
        logger.info("Starting startup sequence...")

        # Step 1: Connect to IBKR
        logger.info("[1/6] Connecting to IBKR...")
        try:
            await self.broker.connect()
            logger.info("IBKR connection established.")
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            logger.warning("Continuing in offline mode - backtest only.")

        # Update risk manager portfolio value from broker
        if self.broker.connected:
            try:
                pv = self.broker.get_portfolio_value()
                if pv > 0:
                    self.risk_manager.update_portfolio_value(pv)
                    logger.info(f"Portfolio value: ${pv:,.2f}")
            except Exception as e:
                logger.warning(f"Could not fetch portfolio value: {e}")

        # Step 2: Warm up market data
        logger.info("[2/6] Warming up market data...")
        try:
            await self._warmup_market_data()
            logger.info("Market data warmup complete.")
        except Exception as e:
            logger.error(f"Market data warmup failed: {e}")

        # Step 3: Load ML models
        logger.info("[3/6] Loading ML models...")
        await self._load_or_train_models()

        # Step 4: Set up async loops and scheduler
        logger.info("[4/6] Setting up async loops...")
        self._setup_scheduler()

        # Step 5: Run startup backtest
        logger.info("[5/6] Running startup backtest...")
        try:
            await self._run_startup_backtest()
        except Exception as e:
            logger.error(f"Startup backtest failed: {e}")

        # Step 6: Start dashboard
        logger.info("[6/6] Starting dashboard...")
        self._start_dashboard()

        logger.info("Startup sequence complete. Entering live loop.")

    async def _warmup_market_data(self):
        """Fetch historical data for all instruments including cross-asset."""
        try:
            await self.market_feed.fetch_historical_bars()
            logger.info("Historical bars fetched for all instruments.")
        except Exception as e:
            logger.error(f"Failed to fetch historical bars: {e}")

        # Fetch cross-asset data for correlation features
        try:
            await self.market_feed.fetch_cross_asset_bars()
            logger.info("Cross-asset bars fetched (DXY, SPX, crack spread).")
        except Exception as e:
            logger.warning(f"Cross-asset data fetch failed: {e}")

    async def _load_or_train_models(self):
        """Load latest saved models or train fresh ones."""
        # Try loading XGBoost model
        if self.signal_model.load_latest_model():
            logger.info("Loaded latest XGBoost signal model.")
        else:
            logger.info("No saved XGBoost model found. Will train on first available data.")

        # Try loading RL agents (per instrument)
        any_rl_loaded = False
        for inst in self.rl_sizer.instruments:
            if self.rl_sizer.load_model(inst):
                any_rl_loaded = True
        if any_rl_loaded:
            logger.info("Loaded RL position sizer model(s).")
        else:
            logger.info("No saved RL models found. Will train on first available data.")

        # Feature scaler is auto-loaded in FeatureEngineer.__init__
        if self.feature_engineer.scaler is not None:
            logger.info("Feature scaler loaded.")
        else:
            logger.info("No saved scaler found. Will fit on first available data.")

    def _setup_scheduler(self):
        """Configure APScheduler jobs."""
        # Sentiment data pull every 5 minutes
        self.scheduler.add_job(
            self._sentiment_loop,
            IntervalTrigger(minutes=settings.ALT_DATA_INTERVAL_MINUTES),
            id="sentiment_pull",
            name="Sentiment Data Pull",
            max_instances=1,
        )

        # EIA report on Wednesday 10:30 AM ET
        self.scheduler.add_job(
            self._eia_report_check,
            CronTrigger(
                day_of_week="wed",
                hour=settings.EIA_REPORT_HOUR,
                minute=settings.EIA_REPORT_MINUTE,
                timezone=settings.EIA_REPORT_TZ,
            ),
            id="eia_report",
            name="EIA Weekly Report",
        )

        # Weekly model retrain (Sunday midnight)
        self.scheduler.add_job(
            self._weekly_retrain,
            CronTrigger(
                day_of_week="sun",
                hour=settings.RETRAIN_HOUR,
                minute=0,
            ),
            id="weekly_retrain",
            name="Weekly Model Retrain",
        )

        # Weekly RL retrain (Sunday 2 AM)
        self.scheduler.add_job(
            self._weekly_rl_retrain,
            CronTrigger(
                day_of_week="sun",
                hour=settings.RL_RETRAIN_HOUR,
                minute=0,
            ),
            id="weekly_rl_retrain",
            name="Weekly RL Retrain",
        )

        # Weekly scaler refit
        self.scheduler.add_job(
            self._weekly_scaler_refit,
            CronTrigger(day_of_week="sun", hour=1, minute=0),
            id="weekly_scaler_refit",
            name="Weekly Scaler Refit",
        )

        # Portfolio state snapshot every hour
        self.scheduler.add_job(
            self._snapshot_portfolio,
            IntervalTrigger(hours=1),
            id="portfolio_snapshot",
            name="Portfolio Snapshot",
        )

        self.scheduler.start()
        logger.info("Scheduler started with all jobs configured.")

    def _start_dashboard(self):
        """Start the Dash monitoring dashboard in a separate thread."""
        try:
            self.dashboard.start_in_thread()
            logger.info(
                f"Dashboard running at http://{settings.DASHBOARD_HOST}:{settings.DASHBOARD_PORT}"
            )
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")

    async def _run_startup_backtest(self):
        """Run walk-forward backtest on startup and log results."""
        session = get_session()
        try:
            from db.models import BarData
            bars = session.query(BarData).filter(
                BarData.instrument == "CL",
                BarData.bar_size == "1 hour",
            ).order_by(BarData.timestamp).all()

            if len(bars) < 500:
                logger.warning("Insufficient data for startup backtest. Skipping.")
                return

            df = pd.DataFrame([{
                "timestamp": b.timestamp, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "volume": b.volume,
            } for b in bars]).set_index("timestamp")

            results = self.backtest_engine.run_backtest(
                prices_df=df, signals_df=None, params={}
            )
            logger.info(f"Startup backtest results: {results}")
        except Exception as e:
            logger.error(f"Startup backtest error: {e}")
        finally:
            session.close()

    # ==========================================================================
    # MAIN TRADING LOOP
    # ==========================================================================

    async def run(self):
        """Main event loop - runs continuously."""
        self._running = True
        await self.startup()

        logger.info("Entering main trading loop...")

        try:
            # Start market data streaming
            stream_task = asyncio.create_task(self._market_stream_loop())

            # Main signal processing loop
            while self._running:
                try:
                    await self._process_signals()
                except Exception as e:
                    logger.error(f"Error in signal processing: {e}")

                # Wait for next bar close or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=60.0
                    )
                    break  # shutdown requested
                except asyncio.TimeoutError:
                    continue  # normal loop iteration

        except asyncio.CancelledError:
            logger.info("Main loop cancelled.")
        finally:
            await self.shutdown()

    async def _market_stream_loop(self):
        """Continuously stream market data."""
        while self._running:
            try:
                await self.market_feed.stream_realtime_bars()
            except Exception as e:
                logger.error(f"Market stream error: {e}")
                await asyncio.sleep(5)

    async def _process_signals(self):
        """Process signals for all instruments and execute trades."""
        for symbol in settings.ALL_INSTRUMENTS:
            try:
                await self._process_instrument(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    async def _process_instrument(self, symbol: str):
        """Generate signals and potentially trade for a single instrument."""
        # Get latest bar data
        bars_df = self.market_feed.get_enriched_dataframe(symbol, bar_size="5 mins")
        if bars_df is None or len(bars_df) < settings.WARMUP_BARS:
            return

        # Compute technical indicators
        technicals = self.technical_engine.compute_all(bars_df, symbol)
        if technicals is None:
            return

        # Get sentiment signal
        sentiment = self.sentiment_engine.get_sentiment_signal()

        # Build feature vector
        features = self.feature_engineer.build_feature_vector(
            technicals=technicals,
            sentiment=sentiment,
            bar_timestamp=bars_df.index[-1],
        )

        # Check feature staleness
        staleness = self.feature_engineer.check_feature_staleness(features)
        if staleness["is_stale"]:
            logger.warning(f"Stale features detected for {symbol}, skipping signal generation")
            return

        # Scale features
        scaled_features = self.feature_engineer.transform(features)

        # Get ML signal
        prediction = self.signal_model.predict(scaled_features)
        p_long = prediction["P_long"]
        p_flat = prediction["P_flat"]
        p_short = prediction["P_short"]
        signal_strength = prediction["signal_strength"]

        # Check confidence threshold (adaptive)
        current_threshold = self.adaptive_learner.adjust_confidence_threshold(
            settings.CONFIDENCE_THRESHOLD
        )
        if max(p_long, p_short) < current_threshold:
            return  # Not confident enough

        # Determine direction
        if p_long > p_short and signal_strength > 0:
            direction = "LONG"
        elif p_short > p_long and signal_strength < 0:
            direction = "SHORT"
        else:
            return

        # Get RL-based position sizing
        rl_state = self._build_rl_state(scaled_features, symbol)
        rl_action = self.rl_sizer.predict(symbol, rl_state)

        # Check if RL agrees with direction
        if direction == TradeDirection.LONG and rl_action < settings.RL_ACTION_THRESHOLD:
            return
        if direction == TradeDirection.SHORT and rl_action > -settings.RL_ACTION_THRESHOLD:
            return

        # Get ATR for stops
        atr = technicals.get("atr14", 0)
        if atr <= 0:
            return

        entry_price = bars_df["close"].iloc[-1]

        # Calculate stop-loss and take-profit
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price, direction, atr
        )
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price, direction, atr
        )

        # Detect regime and apply adjustments
        regime = self.adaptive_learner.detect_regime(bars_df["close"])
        regime_adj = self.adaptive_learner.get_regime_adjustments(regime)

        # Calculate position size (Kelly) from adaptive learner's rolling stats
        kelly_fraction = self.risk_manager.calculate_kelly_fraction(
            self.adaptive_learner.win_rate,
            self.adaptive_learner.avg_win,
            self.adaptive_learner.avg_loss,
        )

        # Apply RL scaling and regime adjustments
        position_size = abs(rl_action) * kelly_fraction * regime_adj["position_size_mult"]

        # Portfolio value for sizing
        portfolio_value = self.broker.get_portfolio_value()
        self.risk_manager.update_portfolio_value(portfolio_value)
        quantity = self._calculate_quantity(
            symbol, entry_price, position_size, portfolio_value
        )

        if quantity <= 0:
            return

        # Risk management gate
        open_positions = self.broker.get_positions()
        trade_params = {
            "instrument": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "atr": atr,
            "portfolio": {"equity": portfolio_value, "equity_start_of_day": portfolio_value},
            "peak_equity": portfolio_value,
            "open_positions": open_positions,
            "requested_size": quantity * entry_price,
            "signal_strength": signal_strength,
            "confidence": max(p_long, p_short),
            "regime": regime,
            "features": scaled_features.tolist() if hasattr(scaled_features, "tolist") else list(scaled_features),
            "css_score": sentiment.get("css_score", 0),
        }

        approved, reason, adjusted_size = self.risk_manager.evaluate_trade(
            trade_params
        )

        if not approved:
            logger.info(f"Trade rejected for {symbol}: {reason}")
            return

        # Adjust quantity if risk manager modified size (adjusted_size is dollar notional)
        adjusted_qty = self._calculate_quantity(symbol, entry_price, adjusted_size / max(portfolio_value, 1), portfolio_value)
        if adjusted_qty <= 0:
            return
        quantity = adjusted_qty

        # Execute trade
        logger.info(
            f"TRADE SIGNAL: {direction} {symbol} qty={quantity} "
            f"entry={entry_price:.2f} SL={stop_loss:.2f} TP={take_profit:.2f} "
            f"signal={signal_strength:.3f} regime={regime.value}"
        )

        try:
            trade = await self.broker.submit_bracket_order(
                instrument=symbol,
                direction=direction,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            # Log trade to DB
            trade_params["quantity"] = quantity
            self._log_trade(trade, trade_params)

        except Exception as e:
            logger.error(f"Order execution failed for {symbol}: {e}")

    def _build_rl_state(self, features: np.ndarray, symbol: str) -> np.ndarray:
        """Build the state vector for the RL agent."""
        # Get current position info (synchronous calls)
        positions = self.broker.get_positions()
        current_pos = 0
        unrealized_pnl = 0.0
        for pos in positions:
            if pos.get("instrument") == symbol:
                current_pos = pos.get("quantity", 0)
                unrealized_pnl = pos.get("unrealized_pnl", 0.0) or 0.0

        portfolio_value = self.broker.get_portfolio_value()
        portfolio_heat = sum(
            abs(p.get("market_value", 0) or 0) for p in positions
        ) / max(portfolio_value, 1)

        extra_state = np.array([current_pos, unrealized_pnl, portfolio_heat])
        return np.concatenate([features.flatten(), extra_state])

    def _calculate_quantity(
        self, symbol: str, price: float, position_fraction: float,
        portfolio_value: float
    ) -> int:
        """Calculate the number of contracts/shares to trade."""
        dollar_amount = portfolio_value * min(
            position_fraction, settings.MAX_POSITION_FRACTION
        )
        if symbol in settings.FUTURES_INSTRUMENTS:
            # Futures: 1 contract of CL = 1000 barrels
            contract_value = price * 1000
            return max(1, int(dollar_amount / contract_value))
        else:
            # ETF shares
            return max(1, int(dollar_amount / price))

    def _log_trade(self, trade_result: dict, params: dict):
        """Log a trade to the database."""
        session = get_session()
        try:
            import uuid
            trade_dir = TradeDirection.LONG if params["direction"] == "LONG" else TradeDirection.SHORT
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                instrument=params["instrument"],
                direction=trade_dir,
                status=TradeStatus.OPEN,
                entry_time=datetime.utcnow(),
                entry_price=params["entry_price"],
                entry_quantity=params.get("quantity", 0),
                stop_loss_price=params["stop_loss"],
                take_profit_price=params["take_profit"],
                signal_strength=params["signal_strength"],
                confidence=params["confidence"],
                css_score=params.get("css_score"),
                regime=params.get("regime"),
                features_at_entry=params.get("features"),
            )
            session.add(trade)
            session.commit()
            logger.info(f"Trade logged: {trade.trade_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log trade: {e}")
        finally:
            session.close()

    # ==========================================================================
    # SCHEDULED JOBS
    # ==========================================================================

    async def _sentiment_loop(self):
        """Pull sentiment data from all sources."""
        logger.debug("Running sentiment data pull...")
        try:
            # Fetch from all sources concurrently
            twitter_task = asyncio.create_task(self.twitter_feed.fetch_tweets())
            reddit_task = asyncio.create_task(self.reddit_feed.fetch_posts())
            news_task = asyncio.create_task(self.news_feed.fetch_all())
            twitter_items, reddit_items, news_items = await asyncio.gather(
                twitter_task, reddit_task, news_task, return_exceptions=True
            )

            all_items = []
            if isinstance(twitter_items, list):
                all_items.extend(twitter_items)
            if isinstance(reddit_items, list):
                all_items.extend(reddit_items)
            if isinstance(news_items, list):
                all_items.extend(news_items)

            # Process through sentiment engine
            if all_items:
                self.sentiment_engine.process_batch(all_items)
                self.sentiment_engine.compute_css()
                logger.info(f"Processed {len(all_items)} sentiment items.")

        except Exception as e:
            logger.error(f"Sentiment loop error: {e}")

    async def _eia_report_check(self):
        """Check for EIA weekly report."""
        try:
            report = await self.eia_scraper.fetch_weekly_report()
            if report:
                self.sentiment_engine.process_batch([report])
                logger.info("EIA weekly report processed.")
        except Exception as e:
            logger.error(f"EIA report check error: {e}")

    async def _weekly_retrain(self):
        """Retrain XGBoost signal model weekly."""
        logger.info("Starting weekly XGBoost retrain...")
        try:
            session = get_session()
            from db.models import FeatureSnapshot, BarData

            # Get training data
            cutoff = datetime.utcnow() - timedelta(days=settings.TRAINING_LOOKBACK_DAYS)
            features = session.query(FeatureSnapshot).filter(
                FeatureSnapshot.timestamp >= cutoff
            ).order_by(FeatureSnapshot.timestamp).all()

            if len(features) < 100:
                logger.warning("Insufficient data for retraining. Skipping.")
                return

            X = np.array([f.features for f in features])
            bars = session.query(BarData).filter(
                BarData.timestamp >= cutoff,
                BarData.bar_size == "5 mins",
                BarData.instrument == "CL",
            ).order_by(BarData.timestamp).all()

            if len(bars) < len(features):
                logger.warning("Bar/feature mismatch. Skipping retrain.")
                return

            prices = pd.Series([b.close for b in bars[:len(features)]])
            y = self.signal_model.create_labels(prices)

            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X, y = X[:min_len], y[:min_len]

            self.signal_model.train(X, y)
            self.signal_model.save_model()
            logger.info("Weekly XGBoost retrain complete.")

            session.close()
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")

    async def _weekly_rl_retrain(self):
        """Retrain RL position sizer weekly."""
        logger.info("Starting weekly RL retrain...")
        try:
            session = get_session()
            from db.models import FeatureSnapshot, BarData

            cutoff = datetime.utcnow() - timedelta(days=settings.TRAINING_LOOKBACK_DAYS)

            for symbol in ["CL", "BZ"]:
                features = session.query(FeatureSnapshot).filter(
                    FeatureSnapshot.timestamp >= cutoff
                ).order_by(FeatureSnapshot.timestamp).all()

                bars = session.query(BarData).filter(
                    BarData.timestamp >= cutoff,
                    BarData.instrument == symbol,
                ).order_by(BarData.timestamp).all()

                if len(features) < 500 or len(bars) < 500:
                    logger.warning(f"Insufficient data for RL retrain on {symbol}. Skipping.")
                    continue

                min_len = min(len(features), len(bars))
                historical_features = np.array([f.features for f in features[:min_len]])
                historical_prices = np.array([b.close for b in bars[:min_len]])

                try:
                    historical_features = self.feature_engineer.transform(historical_features)
                except Exception:
                    logger.warning(f"Scaler not fitted for {symbol}. Using raw features for RL retrain.")

                self.rl_sizer.train(symbol, historical_features, historical_prices)
                logger.info(f"RL retrain complete for {symbol}")

            session.close()
        except Exception as e:
            logger.error(f"RL retrain failed: {e}")

    async def _weekly_scaler_refit(self):
        """Refit the feature scaler on rolling 90-day window."""
        logger.info("Refitting feature scaler...")
        try:
            session = get_session()
            from db.models import FeatureSnapshot
            cutoff = datetime.utcnow() - timedelta(days=90)
            features = session.query(FeatureSnapshot).filter(
                FeatureSnapshot.timestamp >= cutoff
            ).all()

            if len(features) > 50:
                X = np.array([f.features for f in features])
                self.feature_engineer.fit_scaler(X)
                self.feature_engineer.save_scaler()
                logger.info(f"Scaler refitted on {len(features)} samples.")

            session.close()
        except Exception as e:
            logger.error(f"Scaler refit failed: {e}")

    async def _snapshot_portfolio(self):
        """Take a portfolio state snapshot."""
        try:
            account = self.broker.get_account_summary()
            if not account:
                return

            session = get_session()
            try:
                state = PortfolioState(
                    equity=account.get("equity", 0),
                    cash=account.get("cash", 0),
                    unrealized_pnl=account.get("unrealized_pnl", 0),
                    realized_pnl_day=account.get("realized_pnl_day", 0),
                    open_positions=account.get("open_positions", 0),
                )
                session.add(state)
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Portfolio snapshot error: {e}")

    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================

    async def shutdown(self):
        """Gracefully shut down all components."""
        logger.info("Shutting down Oil Quant Bot...")
        self._running = False
        self._shutdown_event.set()

        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

        # Disconnect from IBKR
        try:
            await self.broker.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting broker: {e}")

        # Save models
        try:
            if self.signal_model.model is not None:
                self.signal_model.save_model()
            for inst in self.rl_sizer.instruments:
                if inst in self.rl_sizer._models:
                    self.rl_sizer.save_model(inst)
            self.feature_engineer.save_scaler()
        except Exception as e:
            logger.error(f"Error saving models: {e}")

        logger.info("Shutdown complete.")


def main():
    """Entry point for the Oil Quantitative Trading Bot."""
    bot = OilQuantBot()

    # Handle signals for graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}. Initiating shutdown...")
        loop.create_task(bot.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()
        logger.info("Bot terminated.")


if __name__ == "__main__":
    main()
