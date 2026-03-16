"""
Technical Indicators Engine for the Oil Quantitative Trading Bot.

Computes trend, momentum, volatility, volume, and oil-specific indicators
using pandas_ta and manual pandas/numpy calculations.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from config import settings


class TechnicalEngine:
    """Computes the full suite of technical indicators for a given OHLCV DataFrame."""

    def __init__(self) -> None:
        logger.info("TechnicalEngine initialised.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def compute_all(self, df: pd.DataFrame, instrument: str) -> Dict[str, Optional[float]]:
        """Compute all technical indicators on *df* and return latest-bar values.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.  Expected columns: open, high, low, close, volume.
            For oil-specific indicators the caller may also supply:
              - close_brent / close_wti  (or the DataFrame is for a single instrument)
              - close_dxy   (Dollar Index)
              - close_spx   (S&P 500)
            An optional ``vwap`` column can be pre-supplied for intraday VWAP.
        instrument : str
            Instrument symbol (e.g. ``"CL"``).

        Returns
        -------
        dict[str, float | None]
            Flat dictionary of indicator name -> latest value.
        """
        if df is None or df.empty or len(df) < 2:
            logger.warning("Insufficient data for technical computation (len={})", 0 if df is None else len(df))
            return {}

        out: Dict[str, Optional[float]] = {}

        # Ensure standard column names (lowercase)
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        # ---- Trend ---------------------------------------------------
        self._compute_trend(df, out)

        # ---- Momentum ------------------------------------------------
        self._compute_momentum(df, out)

        # ---- Volatility ----------------------------------------------
        self._compute_volatility(df, out)

        # ---- Volume --------------------------------------------------
        self._compute_volume(df, out)

        # ---- Oil-specific --------------------------------------------
        self._compute_oil_specific(df, out, instrument)

        # ---- EMA crossover -------------------------------------------
        self._compute_ema_crossover(df, out)

        logger.debug("Computed {} indicators for {}", len(out), instrument)
        return out

    # ==================================================================
    # Trend indicators
    # ==================================================================
    def _compute_trend(self, df: pd.DataFrame, out: Dict[str, Optional[float]]) -> None:
        # EMAs
        for period in settings.EMA_PERIODS:
            col = f"ema_{period}"
            df[col] = ta.ema(df["close"], length=period)
            out[col] = self._last(df[col])

        # MACD
        macd_df = ta.macd(
            df["close"],
            fast=settings.MACD_FAST,
            slow=settings.MACD_SLOW,
            signal=settings.MACD_SIGNAL,
        )
        if macd_df is not None and not macd_df.empty:
            macd_cols = macd_df.columns.tolist()
            # pandas_ta returns columns like MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
            df["macd_line"] = macd_df.iloc[:, 0].values
            df["macd_signal"] = macd_df.iloc[:, 1].values
            df["macd_histogram"] = macd_df.iloc[:, 2].values
            out["macd_line"] = self._last(df["macd_line"])
            out["macd_signal"] = self._last(df["macd_signal"])
            out["macd_histogram"] = self._last(df["macd_histogram"])
        else:
            out["macd_line"] = None
            out["macd_signal"] = None
            out["macd_histogram"] = None

        # ADX
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=settings.ADX_PERIOD)
        if adx_df is not None and not adx_df.empty:
            # First column is ADX, second +DI, third -DI
            df["adx"] = adx_df.iloc[:, 0].values
            df["plus_di"] = adx_df.iloc[:, 1].values
            df["minus_di"] = adx_df.iloc[:, 2].values
            out["adx"] = self._last(df["adx"])
            out["plus_di"] = self._last(df["plus_di"])
            out["minus_di"] = self._last(df["minus_di"])
        else:
            out["adx"] = None
            out["plus_di"] = None
            out["minus_di"] = None

    # ==================================================================
    # Momentum indicators
    # ==================================================================
    def _compute_momentum(self, df: pd.DataFrame, out: Dict[str, Optional[float]]) -> None:
        # RSI
        rsi = ta.rsi(df["close"], length=settings.RSI_PERIOD)
        if rsi is not None:
            df["rsi"] = rsi.values
            out["rsi"] = self._last(df["rsi"])
        else:
            out["rsi"] = None

        # Stochastic RSI
        stoch_rsi = ta.stochrsi(
            df["close"],
            length=settings.STOCH_RSI_PERIOD,
            rsi_length=settings.STOCH_RSI_PERIOD,
            k=settings.STOCH_RSI_K,
            d=settings.STOCH_RSI_D,
        )
        if stoch_rsi is not None and not stoch_rsi.empty:
            df["stochrsi_k"] = stoch_rsi.iloc[:, 0].values
            df["stochrsi_d"] = stoch_rsi.iloc[:, 1].values
            out["stochrsi_k"] = self._last(df["stochrsi_k"])
            out["stochrsi_d"] = self._last(df["stochrsi_d"])
        else:
            out["stochrsi_k"] = None
            out["stochrsi_d"] = None

        # Rate of Change
        roc = ta.roc(df["close"], length=settings.ROC_PERIOD)
        if roc is not None:
            df["roc"] = roc.values
            out["roc"] = self._last(df["roc"])
        else:
            out["roc"] = None

    # ==================================================================
    # Volatility indicators
    # ==================================================================
    def _compute_volatility(self, df: pd.DataFrame, out: Dict[str, Optional[float]]) -> None:
        # ATR
        atr = ta.atr(df["high"], df["low"], df["close"], length=settings.ATR_PERIOD)
        if atr is not None:
            df["atr"] = atr.values
            out["atr"] = self._last(df["atr"])
        else:
            out["atr"] = None

        # Bollinger Bands
        bb = ta.bbands(df["close"], length=settings.BB_PERIOD, std=settings.BB_STD)
        if bb is not None and not bb.empty:
            # pandas_ta returns: BBL, BBM, BBU, BBB, BBP
            df["bb_lower"] = bb.iloc[:, 0].values
            df["bb_mid"] = bb.iloc[:, 1].values
            df["bb_upper"] = bb.iloc[:, 2].values
            df["bb_bandwidth"] = bb.iloc[:, 3].values
            df["bb_percent_b"] = bb.iloc[:, 4].values
            out["bb_lower"] = self._last(df["bb_lower"])
            out["bb_mid"] = self._last(df["bb_mid"])
            out["bb_upper"] = self._last(df["bb_upper"])
            out["bb_bandwidth"] = self._last(df["bb_bandwidth"])
            out["bb_percent_b"] = self._last(df["bb_percent_b"])
        else:
            for k in ("bb_lower", "bb_mid", "bb_upper", "bb_bandwidth", "bb_percent_b"):
                out[k] = None

        # Historical Volatility = std(log returns) * sqrt(252)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        hv = log_ret.rolling(window=settings.HIST_VOL_PERIOD).std() * np.sqrt(252)
        df["hist_vol"] = hv
        out["hist_vol"] = self._last(df["hist_vol"])

    # ==================================================================
    # Volume indicators
    # ==================================================================
    def _compute_volume(self, df: pd.DataFrame, out: Dict[str, Optional[float]]) -> None:
        if "volume" not in df.columns or df["volume"].isna().all():
            for k in ("obv", "vwap", "volume_zscore"):
                out[k] = None
            return

        # OBV
        obv = ta.obv(df["close"], df["volume"])
        if obv is not None:
            df["obv"] = obv.values
            out["obv"] = self._last(df["obv"])
        else:
            out["obv"] = None

        # VWAP - use pre-supplied column or compute intraday approximation
        if "vwap" in df.columns and not df["vwap"].isna().all():
            out["vwap"] = self._last(df["vwap"])
        else:
            # Intraday VWAP approximation: cumsum(typical_price * volume) / cumsum(volume)
            typical = (df["high"] + df["low"] + df["close"]) / 3.0
            cum_tv = (typical * df["volume"]).cumsum()
            cum_vol = df["volume"].cumsum()
            df["vwap"] = cum_tv / cum_vol.replace(0, np.nan)
            out["vwap"] = self._last(df["vwap"])

        # Volume Z-score = (vol - vol_ma20) / vol_std20
        vol_ma = df["volume"].rolling(window=settings.VOLUME_MA_PERIOD).mean()
        vol_std = df["volume"].rolling(window=settings.VOLUME_MA_PERIOD).std()
        vol_zscore = (df["volume"] - vol_ma) / vol_std.replace(0, np.nan)
        df["volume_zscore"] = vol_zscore
        out["volume_zscore"] = self._last(df["volume_zscore"])

    # ==================================================================
    # Oil-specific indicators
    # ==================================================================
    def _compute_oil_specific(
        self, df: pd.DataFrame, out: Dict[str, Optional[float]], instrument: str
    ) -> None:
        # WTI / Brent spread
        if "close_wti" in df.columns and "close_brent" in df.columns:
            df["wti_brent_spread"] = df["close_wti"] - df["close_brent"]
            out["wti_brent_spread"] = self._last(df["wti_brent_spread"])
        else:
            out["wti_brent_spread"] = None

        # Oil / DXY rolling correlation
        if "close_dxy" in df.columns:
            oil_ret = df["close"].pct_change()
            dxy_ret = df["close_dxy"].pct_change()
            corr = oil_ret.rolling(window=settings.CORR_ROLLING_WINDOW).corr(dxy_ret)
            df["oil_dxy_corr"] = corr
            out["oil_dxy_corr"] = self._last(df["oil_dxy_corr"])
        else:
            out["oil_dxy_corr"] = None

        # Oil / SPX rolling correlation
        if "close_spx" in df.columns:
            oil_ret = df["close"].pct_change()
            spx_ret = df["close_spx"].pct_change()
            corr = oil_ret.rolling(window=settings.CORR_ROLLING_WINDOW).corr(spx_ret)
            df["oil_spx_corr"] = corr
            out["oil_spx_corr"] = self._last(df["oil_spx_corr"])
        else:
            out["oil_spx_corr"] = None

    # ==================================================================
    # EMA crossover detection
    # ==================================================================
    def _compute_ema_crossover(self, df: pd.DataFrame, out: Dict[str, Optional[float]]) -> None:
        """Detect EMA-9 crossing above/below EMA-21 on the latest bar.

        Signal:
            +1  if EMA9 crosses above EMA21 (bullish)
            -1  if EMA9 crosses below EMA21 (bearish)
             0  no crossover
        """
        if "ema_9" not in df.columns or "ema_21" not in df.columns:
            out["ema9_cross_ema21"] = None
            return

        if len(df) < 2:
            out["ema9_cross_ema21"] = 0
            return

        prev_diff = df["ema_9"].iloc[-2] - df["ema_21"].iloc[-2]
        curr_diff = df["ema_9"].iloc[-1] - df["ema_21"].iloc[-1]

        if pd.isna(prev_diff) or pd.isna(curr_diff):
            out["ema9_cross_ema21"] = 0
            return

        if prev_diff <= 0 and curr_diff > 0:
            out["ema9_cross_ema21"] = 1
        elif prev_diff >= 0 and curr_diff < 0:
            out["ema9_cross_ema21"] = -1
        else:
            out["ema9_cross_ema21"] = 0

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _last(series: pd.Series) -> Optional[float]:
        """Return the last non-NaN value in a Series, or None."""
        if series is None or series.empty:
            return None
        val = series.iloc[-1]
        if pd.isna(val):
            return None
        return float(val)
