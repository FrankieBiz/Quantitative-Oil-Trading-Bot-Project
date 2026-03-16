"""
Backtesting engine for the Oil Quantitative Trading Bot.

Provides vectorized backtesting via vectorbt, walk-forward analysis,
and Monte Carlo simulation for strategy robustness evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from config import settings
from db.models import Trade, TradeDirection, TradeStatus


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestParams:
    """Parameters controlling a single backtest run.

    Attributes:
        init_cash: Starting portfolio equity in USD.
        slippage_pct: One-way slippage as a fraction (e.g. 0.0005 = 0.05%).
        commission_per_contract: Per-contract commission in USD (for futures).
        commission_pct: Fallback percentage-based commission (for ETFs).
        risk_free_rate: Annualised risk-free rate for Sharpe calculations.
        freq: Pandas frequency string that matches the bar cadence.
        use_per_contract_commission: Use per-contract instead of pct commission.
    """

    init_cash: float = 100_000.0
    slippage_pct: float = settings.SLIPPAGE_PCT
    commission_per_contract: float = settings.COMMISSION_PER_CONTRACT
    commission_pct: float = 0.001
    risk_free_rate: float = settings.RISK_FREE_RATE
    freq: str = "1D"
    use_per_contract_commission: bool = False


@dataclass
class BacktestResult:
    """Container for everything produced by a single backtest run.

    Attributes:
        portfolio: The vectorbt Portfolio object with full analytics.
        equity_curve: Time-indexed Series of portfolio equity.
        returns: Time-indexed Series of simple period returns.
        drawdown: Time-indexed Series of drawdown from running peak.
        trades_df: DataFrame of individual round-trip trades.
        params: The parameters that produced this result.
        period_label: Human-readable label (e.g. 'in_sample', 'fold_3').
    """

    portfolio: Any  # vbt.Portfolio
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    drawdown: pd.Series = field(default_factory=pd.Series)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: BacktestParams = field(default_factory=BacktestParams)
    period_label: str = ""


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward optimisation.

    Attributes:
        fold_results: Per-fold BacktestResult for each test window.
        combined_equity: Equity curve stitched from all test windows.
        combined_returns: Returns stitched from all test windows.
        n_folds: Number of folds executed.
        summary: Dict of aggregate statistics across folds.
    """

    fold_results: List[BacktestResult] = field(default_factory=list)
    combined_equity: pd.Series = field(default_factory=pd.Series)
    combined_returns: pd.Series = field(default_factory=pd.Series)
    n_folds: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """Results of a Monte Carlo simulation on an equity curve.

    Attributes:
        ruin_probability: Fraction of paths that hit the ruin threshold.
        ci_95: (lower, upper) tuple for final equity at 95 % confidence.
        ci_99: (lower, upper) tuple for final equity at 99 % confidence.
        median_final_equity: Median terminal equity across simulations.
        mean_final_equity: Mean terminal equity across simulations.
        simulated_paths: 2-D array of shape (n_steps, n_simulations).
        max_drawdowns: Array of per-path maximum drawdowns.
    """

    ruin_probability: float = 0.0
    ci_95: Tuple[float, float] = (0.0, 0.0)
    ci_99: Tuple[float, float] = (0.0, 0.0)
    median_final_equity: float = 0.0
    mean_final_equity: float = 0.0
    simulated_paths: np.ndarray = field(default_factory=lambda: np.array([]))
    max_drawdowns: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Vectorised backtesting engine built on top of *vectorbt*.

    The engine supports:
    * single-period backtests with configurable slippage / commission,
    * walk-forward analysis with rolling train/test windows,
    * Monte Carlo simulation for ruin-probability estimation.

    Example::

        engine = BacktestEngine()
        result = engine.run_backtest(signals_df, prices_df)
        wf = engine.walk_forward_backtest(data)
        mc = engine.monte_carlo_simulation(result.equity_curve)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, params: Optional[BacktestParams] = None) -> None:
        """Initialise the engine with optional default parameters.

        Args:
            params: Default :class:`BacktestParams`.  If *None*, sensible
                    defaults (including settings from ``config.settings``)
                    are used.
        """
        self.params = params or BacktestParams()
        logger.info(
            "BacktestEngine initialised  |  cash={cash}  slip={slip}  comm={comm}",
            cash=self.params.init_cash,
            slip=self.params.slippage_pct,
            comm=self.params.commission_pct,
        )

    # ------------------------------------------------------------------
    # Core back-test
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        params: Optional[BacktestParams] = None,
        period_label: str = "",
    ) -> BacktestResult:
        """Execute a vectorised backtest over a single contiguous period.

        ``signals_df`` must contain boolean columns ``entries`` and ``exits``
        aligned to the same datetime index as ``prices_df['close']``.  An
        optional ``short_entries`` / ``short_exits`` pair is respected when
        present, enabling long+short strategies.

        Args:
            signals_df: DataFrame with at least ``entries`` and ``exits``
                        columns (boolean).  Optional ``short_entries`` /
                        ``short_exits`` for short-side signals.
            prices_df:  DataFrame with at least a ``close`` column. ``open``
                        is used for slippage modelling when available.
            params:     Override parameters for this run only.
            period_label: Descriptive tag stored in the result.

        Returns:
            A fully populated :class:`BacktestResult`.

        Raises:
            ValueError: If required columns are missing from the inputs.
        """
        p = params or self.params

        # ---- validate inputs ------------------------------------------------
        if "entries" not in signals_df.columns or "exits" not in signals_df.columns:
            raise ValueError("signals_df must contain 'entries' and 'exits' columns")
        if "close" not in prices_df.columns:
            raise ValueError("prices_df must contain a 'close' column")

        # Align indices
        common_idx = signals_df.index.intersection(prices_df.index)
        if common_idx.empty:
            raise ValueError("signals_df and prices_df have no overlapping index values")

        signals_df = signals_df.loc[common_idx]
        prices_df = prices_df.loc[common_idx]

        close = prices_df["close"]

        entries = signals_df["entries"].astype(bool)
        exits = signals_df["exits"].astype(bool)

        has_short = "short_entries" in signals_df.columns and "short_exits" in signals_df.columns

        logger.info(
            "Running backtest  |  period={lbl}  bars={n}  long_entries={le}  "
            "short_entries={se}",
            lbl=period_label or "default",
            n=len(common_idx),
            le=int(entries.sum()),
            se=int(signals_df["short_entries"].sum()) if has_short else 0,
        )

        # ---- build slippage-adjusted price ----------------------------------
        # Model slippage by shifting the execution price away from close.
        # Buys fill at close * (1 + slip), sells at close * (1 - slip).
        slippage_frac = p.slippage_pct  # 0.0002 = 0.02 %

        # ---- run vectorbt portfolio -----------------------------------------
        if has_short:
            short_entries = signals_df["short_entries"].astype(bool)
            short_exits = signals_df["short_exits"].astype(bool)

            portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=p.init_cash,
                fees=p.commission_pct,
                slippage=slippage_frac,
                freq=p.freq,
                direction="both",
            )
        else:
            portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                init_cash=p.init_cash,
                fees=p.commission_pct,
                slippage=slippage_frac,
                freq=p.freq,
            )

        # ---- extract analytics ----------------------------------------------
        equity_curve = portfolio.value()
        returns = portfolio.returns()
        drawdown = portfolio.drawdown()

        trades_records = portfolio.trades.records_readable
        trades_df = pd.DataFrame(trades_records) if len(trades_records) > 0 else pd.DataFrame()

        result = BacktestResult(
            portfolio=portfolio,
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            trades_df=trades_df,
            params=p,
            period_label=period_label,
        )

        total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100 if len(equity_curve) > 1 else 0.0
        logger.info(
            "Backtest complete  |  period={lbl}  total_return={ret:.2f}%  "
            "trades={nt}  max_dd={dd:.2f}%",
            lbl=period_label or "default",
            ret=total_ret,
            nt=len(trades_df),
            dd=float(drawdown.min()) * 100 if len(drawdown) > 0 else 0.0,
        )

        return result

    # ------------------------------------------------------------------
    # In-sample / Out-of-sample convenience
    # ------------------------------------------------------------------

    def run_in_sample(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        params: Optional[BacktestParams] = None,
    ) -> BacktestResult:
        """Run a backtest over the configured in-sample period (2019-2022).

        The date range is taken from ``config.settings``.  Data outside the
        window is discarded before execution.

        Args:
            signals_df: Signal DataFrame (see :meth:`run_backtest`).
            prices_df:  Price DataFrame (see :meth:`run_backtest`).
            params:     Optional parameter overrides.

        Returns:
            :class:`BacktestResult` for the in-sample window.
        """
        start = pd.Timestamp(settings.BACKTEST_IN_SAMPLE_START)
        end = pd.Timestamp(settings.BACKTEST_IN_SAMPLE_END)
        logger.info("In-sample window: {} to {}", start.date(), end.date())

        sig_slice = signals_df.loc[start:end]
        price_slice = prices_df.loc[start:end]

        return self.run_backtest(sig_slice, price_slice, params=params, period_label="in_sample")

    def run_out_of_sample(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        params: Optional[BacktestParams] = None,
    ) -> BacktestResult:
        """Run a backtest over the configured out-of-sample period (2023-2024).

        Args:
            signals_df: Signal DataFrame (see :meth:`run_backtest`).
            prices_df:  Price DataFrame (see :meth:`run_backtest`).
            params:     Optional parameter overrides.

        Returns:
            :class:`BacktestResult` for the out-of-sample window.
        """
        start = pd.Timestamp(settings.BACKTEST_OUT_SAMPLE_START)
        end = pd.Timestamp(settings.BACKTEST_OUT_SAMPLE_END)
        logger.info("Out-of-sample window: {} to {}", start.date(), end.date())

        sig_slice = signals_df.loc[start:end]
        price_slice = prices_df.loc[start:end]

        return self.run_backtest(sig_slice, price_slice, params=params, period_label="out_of_sample")

    # ------------------------------------------------------------------
    # Walk-forward analysis
    # ------------------------------------------------------------------

    def walk_forward_backtest(
        self,
        data: pd.DataFrame,
        train_days: int = settings.WALK_FORWARD_TRAIN_DAYS,
        test_days: int = settings.WALK_FORWARD_TEST_DAYS,
        min_folds: int = settings.WALK_FORWARD_FOLDS_MIN,
        signal_generator: Optional[Any] = None,
        params: Optional[BacktestParams] = None,
    ) -> WalkForwardResult:
        """Perform anchored walk-forward analysis.

        The dataset is divided into sequential (train, test) windows.  For
        each fold the ``signal_generator`` callable is invoked on the train
        slice to produce signals, which are then evaluated on the
        immediately-following test slice.

        If no ``signal_generator`` is supplied the method assumes that
        ``data`` already contains pre-computed ``entries``, ``exits`` (and
        optionally ``short_entries``, ``short_exits``) columns alongside a
        ``close`` column, and simply slices them per fold.

        Args:
            data:     DataFrame with ``close`` and signal columns, or raw
                      OHLCV data when ``signal_generator`` is provided.
            train_days: Number of calendar days per training window.
            test_days:  Number of calendar days per test window.
            min_folds:  Minimum number of folds required.  If the data is
                        too short to produce this many, a warning is logged.
            signal_generator: Optional callable
                ``(train_df, test_df) -> signals_df`` where the returned
                DataFrame covers the test window and contains signal columns.
            params:   Optional parameter overrides applied to every fold.

        Returns:
            :class:`WalkForwardResult` with per-fold and combined results.
        """
        p = params or self.params
        train_td = pd.Timedelta(days=train_days)
        test_td = pd.Timedelta(days=test_days)
        step_td = test_td  # non-overlapping test windows

        start_date = data.index.min()
        end_date = data.index.max()

        # Build fold boundaries
        folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        cursor = start_date + train_td
        while cursor + test_td <= end_date:
            train_start = cursor - train_td
            train_end = cursor
            test_start = cursor
            test_end = cursor + test_td
            folds.append((train_start, train_end, test_start, test_end))
            cursor += step_td

        if len(folds) < min_folds:
            logger.warning(
                "Walk-forward produced only {n} folds (requested min {m}).  "
                "Consider providing more data or reducing window sizes.",
                n=len(folds),
                m=min_folds,
            )

        logger.info(
            "Walk-forward  |  folds={nf}  train={tr}d  test={te}d  "
            "span={s} to {e}",
            nf=len(folds),
            tr=train_days,
            te=test_days,
            s=start_date.date(),
            e=end_date.date(),
        )

        fold_results: List[BacktestResult] = []
        equity_parts: List[pd.Series] = []
        return_parts: List[pd.Series] = []

        for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
            train_slice = data.loc[tr_s:tr_e]
            test_slice = data.loc[te_s:te_e]

            if test_slice.empty:
                logger.debug("Fold {} test window is empty, skipping.", i)
                continue

            # Generate or extract signals for test window
            if signal_generator is not None:
                signals_df = signal_generator(train_slice, test_slice)
            else:
                signal_cols = ["entries", "exits"]
                if "short_entries" in data.columns:
                    signal_cols += ["short_entries", "short_exits"]
                signals_df = test_slice[signal_cols]

            prices_df = test_slice[["close"]].copy()
            if "open" in test_slice.columns:
                prices_df["open"] = test_slice["open"]

            fold_label = f"fold_{i}"
            result = self.run_backtest(signals_df, prices_df, params=p, period_label=fold_label)
            fold_results.append(result)

            equity_parts.append(result.equity_curve)
            return_parts.append(result.returns)

        # Stitch equity curves: chain them so each fold starts where the
        # previous one ended.
        combined_equity = self._stitch_equity_curves(equity_parts, p.init_cash)
        combined_returns = pd.concat(return_parts) if return_parts else pd.Series(dtype=float)

        # Aggregate summary stats
        fold_total_returns = [
            (r.equity_curve.iloc[-1] / r.equity_curve.iloc[0] - 1)
            for r in fold_results
            if len(r.equity_curve) > 1
        ]

        summary: Dict[str, Any] = {
            "n_folds": len(fold_results),
            "mean_fold_return": float(np.mean(fold_total_returns)) if fold_total_returns else 0.0,
            "median_fold_return": float(np.median(fold_total_returns)) if fold_total_returns else 0.0,
            "std_fold_return": float(np.std(fold_total_returns)) if fold_total_returns else 0.0,
            "pct_profitable_folds": float(np.mean([r > 0 for r in fold_total_returns])) if fold_total_returns else 0.0,
            "total_return": float(combined_equity.iloc[-1] / p.init_cash - 1) if len(combined_equity) > 0 else 0.0,
        }

        logger.info(
            "Walk-forward complete  |  folds={nf}  total_return={tr:.2f}%  "
            "profitable_folds={pf:.0f}%",
            nf=summary["n_folds"],
            tr=summary["total_return"] * 100,
            pf=summary["pct_profitable_folds"] * 100,
        )

        return WalkForwardResult(
            fold_results=fold_results,
            combined_equity=combined_equity,
            combined_returns=combined_returns,
            n_folds=len(fold_results),
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self,
        equity_curve: pd.Series,
        n_simulations: int = settings.MONTE_CARLO_SIMULATIONS,
        ruin_threshold: float = 0.5,
        seed: Optional[int] = 42,
    ) -> MonteCarloResult:
        """Run a Monte Carlo simulation by bootstrapping trade returns.

        The method extracts period-over-period returns from *equity_curve*,
        then for each simulation randomly resamples those returns (with
        replacement) to construct a synthetic equity path.  This produces a
        distribution of terminal outcomes and maximum drawdowns.

        Args:
            equity_curve: Time-indexed Series of portfolio equity.
            n_simulations: Number of random paths to generate.
            ruin_threshold: Fraction of initial equity below which a path
                            is considered "ruined".  E.g. 0.5 means losing
                            50% of starting capital.
            seed: Random seed for reproducibility.

        Returns:
            :class:`MonteCarloResult` with ruin probability and confidence
            intervals.
        """
        if len(equity_curve) < 2:
            logger.warning("Equity curve too short for Monte Carlo simulation.")
            return MonteCarloResult()

        rng = np.random.default_rng(seed)

        returns = equity_curve.pct_change().dropna().values
        n_steps = len(returns)
        init_equity = float(equity_curve.iloc[0])
        ruin_level = init_equity * ruin_threshold

        logger.info(
            "Monte Carlo  |  simulations={ns}  steps={nst}  ruin_threshold={rt:.0f}%",
            ns=n_simulations,
            nst=n_steps,
            rt=ruin_threshold * 100,
        )

        # Vectorised bootstrap: sample return indices for all sims at once
        sampled_indices = rng.integers(0, n_steps, size=(n_steps, n_simulations))
        sampled_returns = returns[sampled_indices]  # (n_steps, n_sims)

        # Build equity paths: init_equity * cumprod(1 + r)
        growth_factors = 1.0 + sampled_returns
        cum_growth = np.cumprod(growth_factors, axis=0)
        paths = init_equity * cum_growth  # (n_steps, n_sims)

        # Prepend the starting equity row
        start_row = np.full((1, n_simulations), init_equity)
        full_paths = np.vstack([start_row, paths])  # (n_steps+1, n_sims)

        # Terminal equities
        terminal = full_paths[-1, :]

        # Ruin: any path that dipped below ruin_level at any point
        min_equity_per_path = full_paths.min(axis=0)
        ruin_count = int((min_equity_per_path <= ruin_level).sum())
        ruin_probability = ruin_count / n_simulations

        # Per-path max drawdown
        running_max = np.maximum.accumulate(full_paths, axis=0)
        drawdowns = (full_paths - running_max) / running_max
        max_drawdowns = drawdowns.min(axis=0)  # most negative per path

        # Confidence intervals on terminal equity
        ci_95 = (float(np.percentile(terminal, 2.5)), float(np.percentile(terminal, 97.5)))
        ci_99 = (float(np.percentile(terminal, 0.5)), float(np.percentile(terminal, 99.5)))

        result = MonteCarloResult(
            ruin_probability=ruin_probability,
            ci_95=ci_95,
            ci_99=ci_99,
            median_final_equity=float(np.median(terminal)),
            mean_final_equity=float(np.mean(terminal)),
            simulated_paths=full_paths,
            max_drawdowns=max_drawdowns,
        )

        logger.info(
            "Monte Carlo complete  |  ruin_prob={rp:.2f}%  "
            "median_equity={me:,.0f}  95% CI=[{lo:,.0f}, {hi:,.0f}]",
            rp=ruin_probability * 100,
            me=result.median_final_equity,
            lo=ci_95[0],
            hi=ci_95[1],
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stitch_equity_curves(
        parts: List[pd.Series],
        init_cash: float,
    ) -> pd.Series:
        """Chain multiple equity curves so each starts where the last ended.

        Args:
            parts: List of per-fold equity Series.
            init_cash: Starting cash to scale the first fold.

        Returns:
            A single stitched equity Series with a continuous index.
        """
        if not parts:
            return pd.Series(dtype=float)

        stitched_parts: List[pd.Series] = []
        running_equity = init_cash

        for part in parts:
            if part.empty:
                continue
            scale = running_equity / float(part.iloc[0]) if float(part.iloc[0]) != 0 else 1.0
            scaled = part * scale
            stitched_parts.append(scaled)
            running_equity = float(scaled.iloc[-1])

        if not stitched_parts:
            return pd.Series(dtype=float)

        return pd.concat(stitched_parts)

    @staticmethod
    def split_in_out_sample(
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into in-sample and out-of-sample portions.

        Uses date boundaries from ``config.settings``.

        Args:
            data: DatetimeIndex-ed DataFrame spanning the full date range.

        Returns:
            Tuple of ``(in_sample_df, out_of_sample_df)``.
        """
        is_start = pd.Timestamp(settings.BACKTEST_IN_SAMPLE_START)
        is_end = pd.Timestamp(settings.BACKTEST_IN_SAMPLE_END)
        oos_start = pd.Timestamp(settings.BACKTEST_OUT_SAMPLE_START)
        oos_end = pd.Timestamp(settings.BACKTEST_OUT_SAMPLE_END)

        in_sample = data.loc[is_start:is_end]
        out_of_sample = data.loc[oos_start:oos_end]

        logger.debug(
            "Split  |  IS rows={is_n}  OOS rows={oos_n}",
            is_n=len(in_sample),
            oos_n=len(out_of_sample),
        )

        return in_sample, out_of_sample

    def trades_to_orm(self, trades_df: pd.DataFrame, instrument: str = "CL") -> List[Trade]:
        """Convert a vectorbt trades DataFrame to a list of ORM Trade objects.

        This is useful for persisting backtest trades into the database for
        analysis through the dashboard or post-mortem review.

        Args:
            trades_df: DataFrame produced by ``portfolio.trades.records_readable``.
            instrument: Instrument symbol to tag each trade with.

        Returns:
            List of unsaved :class:`~db.models.Trade` instances.
        """
        orm_trades: List[Trade] = []

        if trades_df.empty:
            return orm_trades

        for idx, row in trades_df.iterrows():
            pnl = float(row.get("PnL", 0.0))
            direction = TradeDirection.LONG if row.get("Direction", "Long") == "Long" else TradeDirection.SHORT

            trade = Trade(
                trade_id=f"bt_{instrument}_{idx}",
                instrument=instrument,
                direction=direction,
                status=TradeStatus.CLOSED,
                entry_time=row.get("Entry Timestamp", None),
                entry_price=float(row.get("Avg Entry Price", 0.0)),
                entry_quantity=float(row.get("Size", 0.0)),
                exit_time=row.get("Exit Timestamp", None),
                exit_price=float(row.get("Avg Exit Price", 0.0)),
                realized_pnl=pnl,
                commission=float(row.get("Entry Fees", 0.0)) + float(row.get("Exit Fees", 0.0)),
            )
            orm_trades.append(trade)

        logger.debug("Converted {} backtest trades to ORM objects.", len(orm_trades))
        return orm_trades
