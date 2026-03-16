"""
Feature engineering pipeline for the Oil Quantitative Trading Bot.

Constructs a 27-element feature vector from technical indicators, sentiment
data, and calendar signals, applies RobustScaler normalization, and persists
snapshots to the database.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from loguru import logger
from sklearn.preprocessing import RobustScaler

from config import settings
from db.models import FeatureSnapshot, get_session


class FeatureEngineer:
    """Builds, scales, and stores feature vectors for the ML signal model."""

    FEATURE_NAMES: List[str] = [
        # Trend (5)
        "ema9",
        "ema21",
        "ema50",
        "ema200",
        "ema9_cross_ema21",
        # MACD (2)
        "macd_hist",
        "macd_signal",
        # Momentum (3)
        "rsi14",
        "stoch_rsi",
        "roc10",
        # Volatility (4)
        "atr14",
        "bb_pct_b",
        "bb_width",
        "hist_vol20",
        # Volume (3)
        "obv_zscore",
        "volume_zscore",
        "vwap_distance",
        # Oil-specific (6)
        "wti_brent_spread",
        "oil_dxy_corr20",
        "oil_spx_corr20",
        "term_structure_spread",
        "crack_spread",
        "contango_flag",
        # Sentiment (4)
        "css_score",
        "css_momentum",
        "css_volume_zscore",
        "css_confidence",
        # Calendar (4)
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        # EIA (1)
        "eia_surprise",
    ]

    NUM_FEATURES: int = len(FEATURE_NAMES)

    # ------------------------------------------------------------------ init
    def __init__(self, instrument: str = "CL") -> None:
        self.instrument = instrument
        self.scaler: Optional[RobustScaler] = None
        self._scaler_path = settings.MODELS_DIR / f"scaler_{instrument}.joblib"
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Attempt to load an existing scaler from disk
        if self._scaler_path.exists():
            self.load_scaler()

    # ------------------------------------------------- build_feature_vector
    def build_feature_vector(
        self,
        technicals: dict,
        sentiment: dict,
        bar_timestamp: datetime,
    ) -> np.ndarray:
        """Assemble a raw (unscaled) feature vector from indicator dicts.

        Parameters
        ----------
        technicals : dict
            Keys should include every technical / oil-specific feature name
            (e.g. ``ema9``, ``atr14``, ``obv_zscore``, ``wti_brent_spread``
            etc.).  Missing keys default to ``0.0``.
        sentiment : dict
            Expected keys: ``css_score``, ``css_momentum``,
            ``css_volume_zscore``, ``eia_surprise``.
        bar_timestamp : datetime
            Timestamp of the current bar, used for cyclical time encoding.

        Returns
        -------
        np.ndarray
            Shape ``(NUM_FEATURES,)`` float64 vector.
        """
        # --- Trend -----------------------------------------------------------
        ema9 = float(technicals.get("ema9", 0.0))
        ema21 = float(technicals.get("ema21", 0.0))
        ema50 = float(technicals.get("ema50", 0.0))
        ema200 = float(technicals.get("ema200", 0.0))
        ema9_cross_ema21 = 1.0 if ema9 > ema21 else 0.0

        # --- MACD -------------------------------------------------------------
        macd_hist = float(technicals.get("macd_hist", 0.0))
        macd_signal = float(technicals.get("macd_signal", 0.0))

        # --- Momentum ---------------------------------------------------------
        rsi14 = float(technicals.get("rsi14", 0.0))
        stoch_rsi = float(technicals.get("stoch_rsi", 0.0))
        roc10 = float(technicals.get("roc10", 0.0))

        # --- Volatility -------------------------------------------------------
        atr14 = float(technicals.get("atr14", 0.0))
        bb_pct_b = float(technicals.get("bb_pct_b", 0.0))
        bb_width = float(technicals.get("bb_width", 0.0))
        hist_vol20 = float(technicals.get("hist_vol20", 0.0))

        # --- Volume -----------------------------------------------------------
        obv_zscore = float(technicals.get("obv_zscore", 0.0))
        volume_zscore = float(technicals.get("volume_zscore", 0.0))

        # vwap_distance = (price - vwap) / atr
        price = float(technicals.get("close", technicals.get("price", 0.0)))
        vwap = float(technicals.get("vwap", price))
        vwap_distance = (price - vwap) / atr14 if atr14 != 0.0 else 0.0

        # --- Oil-specific -----------------------------------------------------
        wti_brent_spread = float(technicals.get("wti_brent_spread", 0.0))
        oil_dxy_corr20 = float(technicals.get("oil_dxy_corr20", 0.0))
        oil_spx_corr20 = float(technicals.get("oil_spx_corr20", 0.0))
        term_structure_spread = float(technicals.get("term_structure_spread", 0.0))
        crack_spread = float(technicals.get("crack_spread", 0.0))
        contango_flag = float(technicals.get("contango_flag", 0.0))

        # --- Sentiment --------------------------------------------------------
        css_score = float(sentiment.get("css_score", 0.0))
        css_momentum = float(sentiment.get("css_momentum", 0.0))
        css_volume_zscore = float(sentiment.get("css_volume_zscore", 0.0))
        css_confidence = float(sentiment.get("css_confidence", 1.0))

        # --- Calendar (cyclical encoding) -------------------------------------
        hour_of_day = bar_timestamp.hour + bar_timestamp.minute / 60.0
        day_of_week = bar_timestamp.weekday()  # 0=Mon … 6=Sun

        hour_sin = math.sin(2 * math.pi * hour_of_day / 24.0)
        hour_cos = math.cos(2 * math.pi * hour_of_day / 24.0)
        dow_sin = math.sin(2 * math.pi * day_of_week / 7.0)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7.0)

        # --- EIA surprise -----------------------------------------------------
        eia_surprise = float(sentiment.get("eia_surprise", 0.0))

        # --- Assemble vector --------------------------------------------------
        vector = np.array(
            [
                ema9, ema21, ema50, ema200, ema9_cross_ema21,
                macd_hist, macd_signal,
                rsi14, stoch_rsi, roc10,
                atr14, bb_pct_b, bb_width, hist_vol20,
                obv_zscore, volume_zscore, vwap_distance,
                wti_brent_spread, oil_dxy_corr20, oil_spx_corr20,
                term_structure_spread, crack_spread, contango_flag,
                css_score, css_momentum, css_volume_zscore, css_confidence,
                hour_sin, hour_cos, dow_sin, dow_cos,
                eia_surprise,
            ],
            dtype=np.float64,
        )

        assert vector.shape == (self.NUM_FEATURES,), (
            f"Feature vector length mismatch: {vector.shape[0]} vs {self.NUM_FEATURES}"
        )
        return vector

    # ---------------------------------------------------- scaler: fit / transform
    def fit_scaler(self, historical_features: np.ndarray) -> None:
        """Fit a RobustScaler on a matrix of historical feature vectors.

        Parameters
        ----------
        historical_features : np.ndarray
            Shape ``(n_samples, NUM_FEATURES)``.
        """
        if historical_features.ndim == 1:
            historical_features = historical_features.reshape(1, -1)

        self.scaler = RobustScaler()
        self.scaler.fit(historical_features)
        logger.info(
            "RobustScaler fitted on {} samples for instrument={}",
            historical_features.shape[0],
            self.instrument,
        )
        self.save_scaler()

    def fit_scaler_rolling(self, historical_features: np.ndarray, window_days: int = 90) -> None:
        """Fit scaler on only the most recent *window_days* worth of rows.

        Designed to be called weekly so the scaler tracks recent market
        conditions on a rolling 90-day window.

        Parameters
        ----------
        historical_features : np.ndarray
            Full historical matrix (n_samples, NUM_FEATURES).  Only the last
            ``window_days`` rows (assuming 1 row per bar) are used.  If fewer
            rows exist, all rows are used.
        window_days : int
            Rolling window in days.  Default 90.
        """
        # Approximate number of 15-min bars in *window_days* trading days
        # ~6.5 trading hours * 4 bars/hr * window_days
        bars_per_day = 26
        n_rows = min(len(historical_features), bars_per_day * window_days)
        subset = historical_features[-n_rows:]
        self.fit_scaler(subset)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Scale a feature vector (or matrix) using the fitted scaler.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(NUM_FEATURES,)`` or ``(n, NUM_FEATURES)``.

        Returns
        -------
        np.ndarray
            Scaled features with same shape as input.
        """
        if self.scaler is None:
            logger.warning("Scaler not fitted; returning raw features for instrument={}", self.instrument)
            return features

        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        scaled = self.scaler.transform(features)

        if single:
            scaled = scaled.squeeze(0)
        return scaled

    # ---------------------------------------------------- persistence
    def save_scaler(self) -> None:
        """Persist the fitted RobustScaler to disk via joblib."""
        if self.scaler is None:
            logger.warning("No scaler to save for instrument={}", self.instrument)
            return
        self._scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self._scaler_path)
        logger.info("Scaler saved to {}", self._scaler_path)

    def load_scaler(self) -> None:
        """Load a previously saved RobustScaler from disk."""
        if not self._scaler_path.exists():
            logger.warning("Scaler file not found at {}", self._scaler_path)
            return
        self.scaler = joblib.load(self._scaler_path)
        logger.info("Scaler loaded from {}", self._scaler_path)

    # ---------------------------------------------------- DB snapshot
    def store_snapshot(
        self,
        bar_timestamp: datetime,
        features: np.ndarray,
        signal_strength: Optional[float] = None,
        prediction: Optional[int] = None,
    ) -> None:
        """Write a FeatureSnapshot row to the database.

        Parameters
        ----------
        bar_timestamp : datetime
            Bar timestamp associated with this feature vector.
        features : np.ndarray
            The (possibly scaled) feature vector.
        signal_strength : float, optional
            Model signal strength at this bar.
        prediction : int, optional
            Model prediction label (-1, 0, 1).
        """
        feature_dict = {
            name: float(val) for name, val in zip(self.FEATURE_NAMES, features)
        }
        session = get_session()
        try:
            snapshot = FeatureSnapshot(
                instrument=self.instrument,
                timestamp=bar_timestamp,
                features=feature_dict,
                signal_strength=signal_strength,
                prediction=prediction,
            )
            session.add(snapshot)
            session.commit()
            logger.debug(
                "FeatureSnapshot stored: instrument={} ts={}",
                self.instrument,
                bar_timestamp,
            )
        except Exception:
            session.rollback()
            logger.exception("Failed to store FeatureSnapshot for instrument={}", self.instrument)
        finally:
            session.close()

    # ---------------------------------------------------- staleness detection
    def check_feature_staleness(self, features: np.ndarray) -> dict:
        """Check for potentially stale features (stuck at 0.0).

        Returns a dict with staleness info. Features that are expected to be
        non-zero during market hours are checked.
        """
        # Features that should rarely be exactly 0.0 during market hours
        expected_nonzero = [
            "ema9", "ema21", "ema50", "atr14", "rsi14", "hist_vol20",
        ]
        stale_features = []
        for name in expected_nonzero:
            if name in self.FEATURE_NAMES:
                idx = self.FEATURE_NAMES.index(name)
                if idx < len(features) and features[idx] == 0.0:
                    stale_features.append(name)

        stale_ratio = len(stale_features) / len(expected_nonzero) if expected_nonzero else 0.0
        if stale_ratio > 0.5:
            logger.warning(
                "Feature staleness detected: {}/{} features are zero: {}",
                len(stale_features), len(expected_nonzero), stale_features,
            )

        return {
            "stale_features": stale_features,
            "stale_ratio": stale_ratio,
            "is_stale": stale_ratio > 0.5,
        }

    # ---------------------------------------------------- convenience
    def build_and_transform(
        self,
        technicals: dict,
        sentiment: dict,
        bar_timestamp: datetime,
        *,
        store: bool = True,
    ) -> np.ndarray:
        """Build, scale, optionally store, and return the feature vector.

        This is the primary entry-point called on every new bar.
        """
        raw = self.build_feature_vector(technicals, sentiment, bar_timestamp)
        scaled = self.transform(raw)

        if store:
            self.store_snapshot(bar_timestamp, scaled)

        return scaled
