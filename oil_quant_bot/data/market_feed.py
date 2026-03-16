"""
IBKR Market Data Feed via ib_insync.

Provides real-time and historical bar data for oil futures (CL, BZ) and
energy ETFs (USO, XLE).  Bars are persisted to the database (BarData model)
and cached in per-instrument, per-bar-size pandas DataFrames for fast
indicator computation.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from ib_insync import IB, Contract, Future, Stock, BarData as IBBarData, util
from loguru import logger

from config import settings
from db.models import BarData, get_session


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------

def _build_contracts() -> Dict[str, Contract]:
    """Build ib_insync Contract objects for every configured instrument."""
    contracts: Dict[str, Contract] = {}
    for symbol, meta in settings.FUTURES_INSTRUMENTS.items():
        contracts[symbol] = Future(
            symbol=symbol,
            exchange=meta["exchange"],
            currency=meta["currency"],
        )
    for symbol, meta in settings.ETF_INSTRUMENTS.items():
        contracts[symbol] = Stock(
            symbol=symbol,
            exchange=meta["exchange"],
            currency=meta["currency"],
        )
    return contracts


def _build_cross_asset_contracts() -> Dict[str, Contract]:
    """Build contracts for cross-asset correlation instruments (DXY, SPX)."""
    contracts: Dict[str, Contract] = {}
    for symbol, meta in settings.CROSS_ASSET_INSTRUMENTS.items():
        contracts[symbol] = Future(
            symbol=symbol,
            exchange=meta["exchange"],
            currency=meta["currency"],
        )
    return contracts


def _build_crack_spread_contracts() -> Dict[str, Contract]:
    """Build contracts for crack spread components (RBOB, Heating Oil)."""
    contracts: Dict[str, Contract] = {}
    for symbol, meta in settings.CRACK_SPREAD_INSTRUMENTS.items():
        contracts[symbol] = Future(
            symbol=symbol,
            exchange=meta["exchange"],
            currency=meta["currency"],
        )
    return contracts


# ---------------------------------------------------------------------------
# Bar-size helpers
# ---------------------------------------------------------------------------

_BAR_SIZE_SECONDS = {
    "1 min": 60,
    "5 mins": 300,
    "15 mins": 900,
    "1 hour": 3600,
    "1 day": 86400,
}


def _resample_rule(bar_size: str) -> str:
    """Return a pandas resample-compatible frequency string."""
    mapping = {
        "1 min": "1min",
        "5 mins": "5min",
        "15 mins": "15min",
        "1 hour": "1h",
        "1 day": "1D",
    }
    return mapping[bar_size]


# ---------------------------------------------------------------------------
# MarketDataFeed
# ---------------------------------------------------------------------------

class MarketDataFeed:
    """Streams and stores IBKR market data for all configured instruments.

    Usage::

        feed = MarketDataFeed()
        await feed.connect()
        await feed.fetch_historical_bars()   # warmup
        await feed.stream_realtime_bars()     # blocks while streaming
    """

    def __init__(self) -> None:
        self._ib = IB()
        self._contracts: Dict[str, Contract] = _build_contracts()
        self._cross_asset_contracts: Dict[str, Contract] = _build_cross_asset_contracts()
        self._crack_contracts: Dict[str, Contract] = _build_crack_spread_contracts()
        self._connected = False
        self._reconnect_attempts = 0

        # In-memory cache: {instrument: {bar_size: pd.DataFrame}}
        self._frames: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(
            lambda: {bs: pd.DataFrame() for bs in settings.BAR_SIZES}
        )

        # Track active realtime-bar subscriptions for teardown
        self._realtime_handles: list = []
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Contract roll tracking: {symbol: last_qualified_expiry}
        self._active_expiries: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to IBKR TWS / Gateway with automatic reconnection.

        Retries up to ``settings.IBKR_MAX_RECONNECT_ATTEMPTS`` with an
        exponential back-off capped at 60 s.  Once connected a heartbeat
        coroutine is started that monitors the connection and triggers
        reconnection on drop.
        """
        while self._reconnect_attempts < settings.IBKR_MAX_RECONNECT_ATTEMPTS:
            try:
                logger.info(
                    "Connecting to IBKR at {}:{} (client {}) attempt {}/{}",
                    settings.IBKR_HOST,
                    settings.IBKR_PORT,
                    settings.IBKR_CLIENT_ID,
                    self._reconnect_attempts + 1,
                    settings.IBKR_MAX_RECONNECT_ATTEMPTS,
                )
                await asyncio.wait_for(
                    self._ib.connectAsync(
                        host=settings.IBKR_HOST,
                        port=settings.IBKR_PORT,
                        clientId=settings.IBKR_CLIENT_ID,
                    ),
                    timeout=settings.IBKR_TIMEOUT,
                )
                self._connected = True
                self._reconnect_attempts = 0
                logger.info("IBKR connection established.")

                # Qualify all contracts so IBKR resolves conIds
                qualified = await asyncio.gather(
                    *[
                        self._ib.qualifyContractsAsync(c)
                        for c in self._contracts.values()
                    ],
                    return_exceptions=True,
                )
                for sym, result in zip(self._contracts, qualified):
                    if isinstance(result, Exception):
                        logger.warning("Failed to qualify {}: {}", sym, result)

                # Start heartbeat monitor
                self._heartbeat_task = asyncio.ensure_future(self._heartbeat())
                return

            except (ConnectionRefusedError, asyncio.TimeoutError, OSError) as exc:
                self._reconnect_attempts += 1
                delay = min(
                    settings.IBKR_RECONNECT_DELAY * (2 ** (self._reconnect_attempts - 1)),
                    60,
                )
                logger.warning(
                    "Connection failed ({}). Retrying in {:.0f}s ...",
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise ConnectionError(
            f"Could not connect after {settings.IBKR_MAX_RECONNECT_ATTEMPTS} attempts."
        )

    async def disconnect(self) -> None:
        """Gracefully disconnect and cancel all subscriptions."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        for handle in self._realtime_handles:
            self._ib.cancelRealTimeBars(handle)
        self._realtime_handles.clear()
        if self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False
        logger.info("IBKR disconnected.")

    async def _heartbeat(self) -> None:
        """Periodically verify the connection and reconnect if lost."""
        while True:
            await asyncio.sleep(settings.IBKR_HEARTBEAT_INTERVAL)
            if not self._ib.isConnected():
                logger.warning("Heartbeat detected IBKR disconnect. Reconnecting ...")
                self._connected = False
                try:
                    await self.connect()
                    # Re-subscribe real-time bars after reconnect
                    await self.stream_realtime_bars()
                except ConnectionError:
                    logger.critical("Heartbeat reconnection failed. Stopping heartbeat.")
                    return

    # ------------------------------------------------------------------
    # Historical bars (warmup)
    # ------------------------------------------------------------------

    async def fetch_historical_bars(
        self,
        lookback: str | None = None,
        bar_sizes: List[str] | None = None,
    ) -> None:
        """Download historical bars for every instrument/bar-size pair.

        Parameters
        ----------
        lookback:
            IB duration string, e.g. ``"5 D"``.  Defaults to
            ``settings.HISTORICAL_LOOKBACK``.
        bar_sizes:
            Which bar sizes to request.  Defaults to ``settings.BAR_SIZES``.
        """
        lookback = lookback or settings.HISTORICAL_LOOKBACK
        bar_sizes = bar_sizes or settings.BAR_SIZES

        for symbol, contract in self._contracts.items():
            for bar_size in bar_sizes:
                try:
                    bars: List[IBBarData] = await self._ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr=lookback,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=2,
                    )
                    if not bars:
                        logger.warning(
                            "No historical bars returned for {} @ {}",
                            symbol,
                            bar_size,
                        )
                        continue

                    df = util.df(bars)
                    df.rename(columns={"date": "timestamp"}, inplace=True)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)

                    # Ensure we have the minimum warmup requirement
                    if len(df) < settings.WARMUP_BARS:
                        logger.warning(
                            "{} @ {}: only {} bars (need {})",
                            symbol,
                            bar_size,
                            len(df),
                            settings.WARMUP_BARS,
                        )

                    # Persist to cache
                    self._frames[symbol][bar_size] = df

                    # Persist to DB
                    self._store_bars(symbol, bar_size, df)

                    logger.info(
                        "Loaded {} historical bars for {} @ {}",
                        len(df),
                        symbol,
                        bar_size,
                    )

                except Exception:
                    logger.exception(
                        "Error fetching historical bars for {} @ {}",
                        symbol,
                        bar_size,
                    )

    # ------------------------------------------------------------------
    # Real-time bars
    # ------------------------------------------------------------------

    async def stream_realtime_bars(self) -> None:
        """Subscribe to 5-second real-time bars for every instrument.

        Incoming ticks are aggregated into every configured bar size and
        persisted both in-memory and to the database.  This coroutine runs
        indefinitely.
        """
        for symbol, contract in self._contracts.items():
            handle = self._ib.reqRealTimeBars(
                contract,
                barSize=5,
                whatToShow="TRADES",
                useRTH=False,
            )
            handle.updateEvent += lambda bars, _has_new, sym=symbol: self._on_realtime_bar(
                sym, bars
            )
            self._realtime_handles.append(handle)
            logger.info("Subscribed to real-time bars for {}", symbol)

        # Run the ib_insync event loop until externally cancelled
        try:
            while self._connected:
                await asyncio.sleep(0.25)
                self._ib.sleep(0)  # pump ib_insync events
        except asyncio.CancelledError:
            logger.info("Real-time streaming cancelled.")

    def _on_realtime_bar(self, symbol: str, bars) -> None:
        """Callback for each incoming 5-second real-time bar.

        Appends the tick to the 1-min accumulator and, when a period
        boundary is crossed, rolls up into every configured bar size.
        """
        bar = bars[-1]
        tick_ts = pd.Timestamp(
            datetime.fromtimestamp(bar.time, tz=timezone.utc)
        )
        row = {
            "open": bar.open_,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume if bar.volume != -1 else 0,
            "vwap": bar.wap if bar.wap != -1 else None,
        }

        # Append to 1-min frame as-is (will be resampled)
        one_min_df = self._frames[symbol]["1 min"]
        new_row = pd.DataFrame([row], index=[tick_ts])
        new_row.index.name = "timestamp"
        self._frames[symbol]["1 min"] = pd.concat([one_min_df, new_row])

        # Aggregate to higher timeframes whenever a new tick arrives
        self._aggregate_bars(symbol)

    def _aggregate_bars(self, symbol: str) -> None:
        """Resample the 1-min frame into every higher bar size and persist
        any newly completed bars to the database."""
        base_df = self._frames[symbol]["1 min"]
        if base_df.empty:
            return

        for bar_size in settings.BAR_SIZES:
            if bar_size == "1 min":
                continue
            rule = _resample_rule(bar_size)
            resampled = base_df.resample(rule).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "vwap": "last",
                }
            ).dropna(subset=["open"])

            prev = self._frames[symbol][bar_size]
            if not resampled.empty:
                new_bars = resampled
                if not prev.empty:
                    new_bars = resampled.loc[resampled.index > prev.index[-1]]
                if not new_bars.empty:
                    self._frames[symbol][bar_size] = pd.concat([prev, new_bars])
                    self._store_bars(symbol, bar_size, new_bars)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _store_bars(self, symbol: str, bar_size: str, df: pd.DataFrame) -> None:
        """Bulk-insert bars into the database, skipping duplicates."""
        session = get_session()
        try:
            for ts, row in df.iterrows():
                ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                existing = (
                    session.query(BarData)
                    .filter_by(instrument=symbol, bar_size=bar_size, timestamp=ts_dt)
                    .first()
                )
                if existing:
                    continue
                record = BarData(
                    instrument=symbol,
                    bar_size=bar_size,
                    timestamp=ts_dt,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                    vwap=float(row["vwap"]) if pd.notna(row.get("vwap")) else None,
                )
                session.add(record)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to store bars for {} @ {}", symbol, bar_size)
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_dataframe(
        self, symbol: str, bar_size: str = "1 min"
    ) -> pd.DataFrame:
        """Return the cached DataFrame for a given instrument and bar size.

        Parameters
        ----------
        symbol:
            Instrument ticker (``CL``, ``BZ``, ``USO``, ``XLE``).
        bar_size:
            One of ``settings.BAR_SIZES``.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame indexed by UTC timestamp.
        """
        return self._frames[symbol][bar_size].copy()

    @property
    def is_connected(self) -> bool:
        """Whether the IBKR connection is alive."""
        return self._connected and self._ib.isConnected()

    @property
    def instruments(self) -> List[str]:
        """List of subscribed instrument symbols."""
        return list(self._contracts.keys())

    # ------------------------------------------------------------------
    # Cross-asset data for correlation features
    # ------------------------------------------------------------------

    async def fetch_cross_asset_bars(
        self,
        lookback: str | None = None,
        bar_size: str = "5 mins",
    ) -> None:
        """Fetch historical bars for cross-asset instruments (DXY, SPX).

        These are used for oil-DXY and oil-SPX correlation features.
        """
        lookback = lookback or settings.HISTORICAL_LOOKBACK
        all_contracts = {**self._cross_asset_contracts, **self._crack_contracts}

        for symbol, contract in all_contracts.items():
            try:
                bars = await self._ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr=lookback,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,
                )
                if not bars:
                    logger.warning("No cross-asset bars for {} @ {}", symbol, bar_size)
                    continue

                df = util.df(bars)
                df.rename(columns={"date": "timestamp"}, inplace=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                self._frames[symbol][bar_size] = df
                logger.info("Loaded {} cross-asset bars for {} @ {}", len(df), symbol, bar_size)

            except Exception:
                logger.exception("Error fetching cross-asset bars for {}", symbol)

    # ------------------------------------------------------------------
    # Contract roll detection
    # ------------------------------------------------------------------

    async def check_contract_roll(self, symbol: str) -> bool:
        """Check if a futures contract needs to be rolled to the next month.

        Detects roll by comparing the currently qualified contract's expiry
        against the last known expiry. When a roll is detected the contract
        is re-qualified and the cache is updated.

        Returns True if a roll occurred.
        """
        if symbol not in settings.FUTURES_INSTRUMENTS:
            return False

        contract = self._contracts[symbol]
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified or not qualified[0]:
                return False

            new_contract = qualified[0]
            new_expiry = getattr(new_contract, "lastTradeDateOrContractMonth", "")
            old_expiry = self._active_expiries.get(symbol, "")

            if old_expiry and new_expiry != old_expiry:
                logger.warning(
                    "Contract roll detected for {}: {} -> {}",
                    symbol,
                    old_expiry,
                    new_expiry,
                )
                self._contracts[symbol] = new_contract
                self._active_expiries[symbol] = new_expiry
                return True

            self._active_expiries[symbol] = new_expiry
            return False

        except Exception:
            logger.exception("Error checking contract roll for {}", symbol)
            return False

    def get_enriched_dataframe(
        self, symbol: str, bar_size: str = "5 mins"
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame enriched with cross-asset columns.

        Merges DXY, SPX, crack spread components, and second-month futures
        data as additional columns for the technical engine.
        """
        df = self.get_dataframe(symbol, bar_size)
        if df.empty:
            return df

        # Add DXY close for oil-DXY correlation
        dxy_df = self._frames.get("DX", {}).get(bar_size, pd.DataFrame())
        if not dxy_df.empty and "close" in dxy_df.columns:
            df["close_dxy"] = dxy_df["close"].reindex(df.index, method="ffill")

        # Add SPX close for oil-SPX correlation
        es_df = self._frames.get("ES", {}).get(bar_size, pd.DataFrame())
        if not es_df.empty and "close" in es_df.columns:
            df["close_spx"] = es_df["close"].reindex(df.index, method="ffill")

        # Add crack spread components
        rb_df = self._frames.get("RB", {}).get(bar_size, pd.DataFrame())
        if not rb_df.empty and "close" in rb_df.columns:
            df["close_rb"] = rb_df["close"].reindex(df.index, method="ffill")

        ho_df = self._frames.get("HO", {}).get(bar_size, pd.DataFrame())
        if not ho_df.empty and "close" in ho_df.columns:
            df["close_ho"] = ho_df["close"].reindex(df.index, method="ffill")

        # WTI/Brent spread
        if symbol == "CL":
            bz_df = self._frames.get("BZ", {}).get(bar_size, pd.DataFrame())
            if not bz_df.empty and "close" in bz_df.columns:
                df["close_wti"] = df["close"]
                df["close_brent"] = bz_df["close"].reindex(df.index, method="ffill")
        elif symbol == "BZ":
            cl_df = self._frames.get("CL", {}).get(bar_size, pd.DataFrame())
            if not cl_df.empty and "close" in cl_df.columns:
                df["close_wti"] = cl_df["close"].reindex(df.index, method="ffill")
                df["close_brent"] = df["close"]

        return df
