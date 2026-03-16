"""
Continuous Learning Loop for the Oil Quantitative Trading Bot.

Provides adaptive mechanisms that monitor live trading performance and
dynamically adjust model confidence thresholds, position sizing, and
retraining schedules based on rolling trade statistics and detected
market regimes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from db.models import (
    MarketRegime,
    Trade,
    TradeStatus,
    get_session,
)


class AdaptiveLearner:
    """Continuous learning loop that adapts trading behaviour to recent performance.

    Responsibilities:
        - Log closed trades with full feature snapshots.
        - Maintain rolling performance statistics (win rate, avg win/loss, Kelly).
        - Trigger incremental XGBoost retraining when enough new trades accumulate.
        - Dynamically adjust the signal confidence threshold based on recent results.
        - Detect the current market regime via Hurst exponent and ADX.
        - Provide regime-based position-size and stop-width multipliers.

    Attributes:
        trade_log: In-memory list of closed-trade records used for rolling stats.
        rolling_window: Number of trades used for rolling statistics.
        trades_since_retrain: Counter of trades logged since the last model retrain.
        win_rate: Current rolling win rate.
        avg_win: Average net PnL of winning trades in the rolling window.
        avg_loss: Average net PnL of losing trades in the rolling window (positive value).
        kelly_fraction: Current Kelly fraction (half-Kelly applied).
    """

    def __init__(self) -> None:
        """Initialise the AdaptiveLearner with default state."""
        self.rolling_window: int = settings.KELLY_ROLLING_TRADES  # 50
        self.trade_log: List[Dict[str, Any]] = []
        self.trades_since_retrain: int = 0

        # Rolling statistics – initialised to conservative defaults.
        self.win_rate: float = 0.50
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0
        self.kelly_fraction: float = settings.KELLY_FRACTION

        logger.info(
            "AdaptiveLearner initialised | rolling_window={} | retrain_threshold={}",
            self.rolling_window,
            settings.RETRAIN_TRADE_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # 1. Trade logging
    # ------------------------------------------------------------------

    def log_closed_trade(self, trade: Trade) -> None:
        """Record a closed trade with entry/exit details, features, and outcome.

        The trade object is expected to be a fully populated ``Trade`` ORM
        instance with status ``CLOSED``.

        Args:
            trade: A closed ``Trade`` ORM object.
        """
        net_pnl = trade.net_pnl()
        record: Dict[str, Any] = {
            "trade_id": trade.trade_id,
            "instrument": trade.instrument,
            "direction": trade.direction.value if trade.direction else None,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "exit_time": trade.exit_time,
            "exit_price": trade.exit_price,
            "signal_strength": trade.signal_strength,
            "confidence": trade.confidence,
            "features_at_entry": trade.features_at_entry,
            "realized_pnl": trade.realized_pnl,
            "commission": trade.commission,
            "slippage": trade.slippage,
            "net_pnl": net_pnl,
            "is_win": net_pnl > 0.0,
            "logged_at": datetime.utcnow(),
        }

        self.trade_log.append(record)
        self.trades_since_retrain += 1

        logger.info(
            "Trade logged | id={} | instrument={} | net_pnl={:.2f} | wins_since_retrain={}",
            trade.trade_id,
            trade.instrument,
            net_pnl,
            self.trades_since_retrain,
        )

        # Recompute rolling stats after every new trade.
        self.update_rolling_stats()

    # ------------------------------------------------------------------
    # 2. Rolling statistics update
    # ------------------------------------------------------------------

    def update_rolling_stats(self) -> None:
        """Recalculate win rate, average win, and average loss on a rolling window.

        Uses the most recent ``self.rolling_window`` trades (default 50).
        After updating these statistics the Kelly fraction is recalculated.
        """
        window = self.trade_log[-self.rolling_window :]
        if not window:
            return

        wins = [t for t in window if t["is_win"]]
        losses = [t for t in window if not t["is_win"]]

        total = len(window)
        self.win_rate = len(wins) / total if total > 0 else 0.0

        self.avg_win = (
            float(np.mean([t["net_pnl"] for t in wins])) if wins else 0.0
        )
        self.avg_loss = (
            float(np.abs(np.mean([t["net_pnl"] for t in losses])))
            if losses
            else 0.0
        )

        self._recalculate_kelly()

        logger.debug(
            "Rolling stats updated | trades={} | win_rate={:.2%} | avg_win={:.2f} | avg_loss={:.2f} | kelly={:.4f}",
            total,
            self.win_rate,
            self.avg_win,
            self.avg_loss,
            self.kelly_fraction,
        )

    def _recalculate_kelly(self) -> None:
        """Recalculate the Kelly fraction from current rolling statistics.

        Uses the formula:
            f* = (win_rate / avg_loss_ratio) - ((1 - win_rate) / avg_win_ratio)

        Where avg_loss_ratio and avg_win_ratio are relative to the average
        win/loss magnitudes.  A simpler equivalent when the payoff odds are
        expressed as ``b = avg_win / avg_loss``:

            f* = win_rate - (1 - win_rate) / b

        The result is then multiplied by 0.5 (half-Kelly) and clamped to
        [0, MAX_POSITION_FRACTION].
        """
        if self.avg_loss <= 0.0 or self.avg_win <= 0.0:
            self.kelly_fraction = settings.KELLY_FRACTION
            return

        b = self.avg_win / self.avg_loss  # payoff ratio
        full_kelly = self.win_rate - (1.0 - self.win_rate) / b

        # Half-Kelly for safety; clamp to sane range.
        half_kelly = full_kelly * 0.5
        self.kelly_fraction = float(
            np.clip(half_kelly, 0.0, settings.MAX_POSITION_FRACTION)
        )

    # ------------------------------------------------------------------
    # 3. Incremental retraining trigger
    # ------------------------------------------------------------------

    def check_retrain_needed(self) -> bool:
        """Check whether enough new trades have accumulated to trigger retraining.

        Returns:
            ``True`` if the number of trades since the last retrain exceeds
            ``settings.RETRAIN_TRADE_THRESHOLD`` (30), ``False`` otherwise.
        """
        needed = self.trades_since_retrain >= settings.RETRAIN_TRADE_THRESHOLD
        if needed:
            logger.info(
                "Retrain triggered | trades_since_retrain={} >= threshold={}",
                self.trades_since_retrain,
                settings.RETRAIN_TRADE_THRESHOLD,
            )
        return needed

    def trigger_incremental_retrain(
        self,
        signal_model: Any,
        recent_X: pd.DataFrame,
        recent_y: pd.Series,
    ) -> Any:
        """Perform an incremental (warm-start) XGBoost retrain.

        Fits additional boosting rounds on top of the existing model using
        recent labelled data.  The caller is responsible for providing the
        feature matrix ``recent_X`` and labels ``recent_y`` corresponding
        to the trades accumulated since the last retrain.

        Args:
            signal_model: An XGBoost classifier instance that supports
                ``fit()`` with ``xgb_model`` for warm-starting.
            recent_X: Feature matrix for the new samples.
            recent_y: Target labels for the new samples.

        Returns:
            The updated signal model after incremental fitting.
        """
        logger.info(
            "Starting incremental retrain | samples={} | features={}",
            len(recent_X),
            recent_X.shape[1] if hasattr(recent_X, "shape") else "?",
        )

        try:
            # XGBoost supports warm-start via the xgb_model parameter.
            # We obtain the underlying booster for warm-starting.
            booster = signal_model.get_booster()
            signal_model.fit(
                recent_X,
                recent_y,
                xgb_model=booster,
            )
            self.trades_since_retrain = 0
            logger.info("Incremental retrain completed successfully.")
        except Exception as exc:
            logger.error("Incremental retrain failed: {}", exc)
            raise

        return signal_model

    # ------------------------------------------------------------------
    # 4. Dynamic confidence threshold
    # ------------------------------------------------------------------

    def adjust_confidence_threshold(self, current_threshold: float) -> float:
        """Dynamically adjust the signal confidence threshold based on recent results.

        Rules (applied to the last ``settings.CONFIDENCE_ADJUST_WINDOW``
        trades, default 10):
            - If recent win rate < 40 %  -> raise threshold by 0.02 (max 0.70).
            - If recent win rate > 65 %  -> lower threshold by 0.01 (min 0.52).
            - Otherwise                  -> no change.

        Args:
            current_threshold: The current confidence threshold.

        Returns:
            The (possibly adjusted) confidence threshold.
        """
        window_size = settings.CONFIDENCE_ADJUST_WINDOW
        recent = self.trade_log[-window_size:]

        if len(recent) < window_size:
            # Not enough trades to make a meaningful adjustment.
            return current_threshold

        recent_wins = sum(1 for t in recent if t["is_win"])
        recent_win_rate = recent_wins / len(recent)

        new_threshold = current_threshold

        if recent_win_rate < settings.CONFIDENCE_RAISE_THRESHOLD:
            new_threshold = min(
                current_threshold + settings.CONFIDENCE_RAISE_STEP,
                settings.CONFIDENCE_THRESHOLD_MAX,
            )
            logger.warning(
                "Confidence threshold RAISED | recent_wr={:.2%} | {:.4f} -> {:.4f}",
                recent_win_rate,
                current_threshold,
                new_threshold,
            )
        elif recent_win_rate > settings.CONFIDENCE_LOWER_THRESHOLD:
            new_threshold = max(
                current_threshold - settings.CONFIDENCE_LOWER_STEP,
                settings.CONFIDENCE_THRESHOLD_MIN,
            )
            logger.info(
                "Confidence threshold LOWERED | recent_wr={:.2%} | {:.4f} -> {:.4f}",
                recent_win_rate,
                current_threshold,
                new_threshold,
            )
        else:
            logger.debug(
                "Confidence threshold unchanged | recent_wr={:.2%} | threshold={:.4f}",
                recent_win_rate,
                current_threshold,
            )

        return new_threshold

    # ------------------------------------------------------------------
    # 5. Regime detection
    # ------------------------------------------------------------------

    def detect_regime(self, prices: pd.Series) -> MarketRegime:
        """Detect the current market regime from a price series.

        Combines two indicators:
            1. **Hurst exponent** (rescaled-range analysis):
               - H > 0.6  -> TRENDING
               - H < 0.4  -> MEAN_REVERTING
               - otherwise -> CHOPPY (random walk)
            2. **ADX** (Average Directional Index, 14-period):
               - ADX > 25 -> TRENDING
               - ADX < 20 -> CHOPPY

        When the two indicators disagree the Hurst exponent takes precedence
        because it captures longer-horizon persistence more robustly.

        Args:
            prices: A ``pd.Series`` of closing prices, ordered chronologically.
                Must contain at least ``settings.ADX_PERIOD * 2`` bars.

        Returns:
            A ``MarketRegime`` enum value.
        """
        if len(prices) < settings.ADX_PERIOD * 2:
            logger.warning(
                "Insufficient price data for regime detection ({} bars); defaulting to CHOPPY.",
                len(prices),
            )
            return MarketRegime.CHOPPY

        hurst = self.calculate_hurst(prices)
        adx = self._calculate_adx(prices, period=settings.ADX_PERIOD)

        # Primary: Hurst exponent
        if hurst > settings.HURST_TRENDING_THRESHOLD:
            hurst_regime = MarketRegime.TRENDING
        elif hurst < settings.HURST_MEAN_REVERTING_THRESHOLD:
            hurst_regime = MarketRegime.MEAN_REVERTING
        else:
            hurst_regime = MarketRegime.CHOPPY

        # Secondary: ADX
        if adx > 25.0:
            adx_regime = MarketRegime.TRENDING
        elif adx < 20.0:
            adx_regime = MarketRegime.CHOPPY
        else:
            adx_regime = MarketRegime.CHOPPY

        # Hurst takes precedence; ADX can reinforce or be overridden.
        if hurst_regime == adx_regime:
            regime = hurst_regime
        else:
            regime = hurst_regime

        logger.info(
            "Regime detected | hurst={:.4f} ({}) | adx={:.2f} ({}) | final={}",
            hurst,
            hurst_regime.value,
            adx,
            adx_regime.value,
            regime.value,
        )
        return regime

    @staticmethod
    def _calculate_adx(prices: pd.Series, period: int = 14) -> float:
        """Calculate the latest ADX value from a price series.

        This is a simplified ADX that derives the directional movement from
        close prices alone (using price changes as proxies for high/low
        movements).  For production use with full OHLC bars, consider
        replacing with a TA-Lib or pandas-ta implementation.

        Args:
            prices: Close price series.
            period: Smoothing period (default 14).

        Returns:
            The most recent ADX value as a float.
        """
        prices = prices.astype(float)
        diff = prices.diff()

        # Proxy directional movements from close-to-close changes.
        plus_dm = diff.clip(lower=0.0)
        minus_dm = (-diff).clip(lower=0.0)

        # True range proxy (close-to-close absolute change).
        tr = diff.abs()

        # Smoothed using exponential moving average.
        atr = tr.ewm(span=period, min_periods=period).mean()
        plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period).mean() / atr)
        minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period).mean() / atr)

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
        adx = dx.ewm(span=period, min_periods=period).mean()

        latest = adx.iloc[-1]
        return float(latest) if pd.notna(latest) else 0.0

    # ------------------------------------------------------------------
    # 6. Regime-based adjustments
    # ------------------------------------------------------------------

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """Return position-size and stop-width multipliers for the given regime.

        Adjustments:
            - **TRENDING**: position_size_mult = 1.2, stop_width_mult = 1.0
            - **MEAN_REVERTING**: position_size_mult = 1.0, stop_width_mult = 0.8
            - **CHOPPY**: position_size_mult = 0.6 (40 % reduction), stop_width_mult = 1.3

        Args:
            regime: The current ``MarketRegime``.

        Returns:
            A dict with keys ``position_size_mult`` and ``stop_width_mult``.
        """
        adjustments: Dict[str, Dict[str, float]] = {
            MarketRegime.TRENDING: {
                "position_size_mult": settings.TRENDING_POSITION_MULTIPLIER,  # 1.2
                "stop_width_mult": 1.0,
            },
            MarketRegime.MEAN_REVERTING: {
                "position_size_mult": 1.0,
                "stop_width_mult": 0.8,
            },
            MarketRegime.CHOPPY: {
                "position_size_mult": 1.0 - settings.CHOPPY_POSITION_REDUCTION,  # 0.6
                "stop_width_mult": 1.3,
            },
        }

        result = adjustments.get(
            regime,
            {"position_size_mult": 1.0, "stop_width_mult": 1.0},
        )

        logger.debug(
            "Regime adjustments | regime={} | pos_mult={:.2f} | stop_mult={:.2f}",
            regime.value,
            result["position_size_mult"],
            result["stop_width_mult"],
        )
        return result

    # ------------------------------------------------------------------
    # 7. Hurst exponent calculation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_hurst(prices: pd.Series, max_lag: int = 100) -> float:
        """Calculate the Hurst exponent using rescaled-range (R/S) analysis.

        The R/S analysis works by:
            1. Converting prices to a log-return series.
            2. For each sub-sample size *n* (from 10 to ``max_lag``):
               a. Split the return series into non-overlapping blocks of size *n*.
               b. For each block compute the cumulative deviation from the block mean.
               c. Compute *R* = max(cumulative) - min(cumulative) and *S* = std(block).
               d. Record R/S = R / S (skip blocks where S is near zero).
            3. Regress log(mean R/S) on log(n).  The slope is the Hurst exponent.

        Args:
            prices: A chronologically ordered ``pd.Series`` of closing prices.
            max_lag: Maximum sub-sample size to consider.

        Returns:
            The estimated Hurst exponent as a float in approximately [0, 1].
            Returns 0.5 (random walk) if the calculation fails.
        """
        try:
            prices_arr = np.asarray(prices, dtype=np.float64)
            if len(prices_arr) < 20:
                logger.warning("Too few prices for Hurst calculation; returning 0.5")
                return 0.5

            # Log returns
            log_prices = np.log(prices_arr[prices_arr > 0])
            if len(log_prices) < 20:
                return 0.5
            returns = np.diff(log_prices)

            n_obs = len(returns)
            max_lag = min(max_lag, n_obs // 2)
            if max_lag < 10:
                return 0.5

            # Range of sub-sample sizes (use logarithmically spaced integers).
            lag_sizes = np.unique(
                np.logspace(np.log10(10), np.log10(max_lag), num=30).astype(int)
            )
            lag_sizes = lag_sizes[lag_sizes >= 10]

            log_lags: List[float] = []
            log_rs: List[float] = []

            for lag in lag_sizes:
                n_blocks = n_obs // lag
                if n_blocks < 1:
                    continue

                rs_values: List[float] = []
                for i in range(n_blocks):
                    block = returns[i * lag : (i + 1) * lag]
                    block_mean = np.mean(block)
                    cumulative_dev = np.cumsum(block - block_mean)
                    r = np.max(cumulative_dev) - np.min(cumulative_dev)
                    s = np.std(block, ddof=1)
                    if s > 1e-12:
                        rs_values.append(r / s)

                if rs_values:
                    log_lags.append(np.log(float(lag)))
                    log_rs.append(np.log(np.mean(rs_values)))

            if len(log_lags) < 3:
                logger.warning("Insufficient R/S data points; returning 0.5")
                return 0.5

            # OLS regression: log(R/S) = H * log(n) + c
            log_lags_arr = np.array(log_lags)
            log_rs_arr = np.array(log_rs)
            poly = np.polyfit(log_lags_arr, log_rs_arr, 1)
            hurst = float(poly[0])

            # Clamp to [0, 1] for safety.
            hurst = float(np.clip(hurst, 0.0, 1.0))

            logger.debug("Hurst exponent calculated: {:.4f}", hurst)
            return hurst

        except Exception as exc:
            logger.error("Hurst calculation failed: {}; returning 0.5", exc)
            return 0.5
