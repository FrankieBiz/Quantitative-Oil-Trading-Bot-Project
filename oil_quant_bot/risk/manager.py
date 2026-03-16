"""
Risk Management Engine for the Oil Quantitative Trading Bot.

Provides position sizing (Kelly Criterion), stop-loss / take-profit management,
trailing stops, portfolio-level guards, VaR estimation, Sharpe monitoring, and
a master gate that must approve every trade before execution.

All tuneable parameters are imported from ``config.settings`` so they can be
changed in a single place.  Alert records are persisted via ``db.models``.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config import settings
from db.models import (
    MarketRegime,
    SystemAlert,
    Trade,
    TradeDirection,
    TradeStatus,
    get_session,
)


class RiskManager:
    """Centralised risk-management engine.

    Instantiate once at bot start-up and call :meth:`evaluate_trade` before
    every order.  The class keeps lightweight in-memory state (recent trades,
    Sharpe history) and delegates persistence to the ORM layer.

    Parameters
    ----------
    portfolio_value : float
        Current total portfolio equity (cash + unrealised PnL).
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    def __init__(self, portfolio_value: float) -> None:
        self.portfolio_value: float = portfolio_value

        # Rolling window of the last N closed trades (PnL values) for Kelly.
        self._trade_results: deque[float] = deque(
            maxlen=settings.KELLY_ROLLING_TRADES,
        )

        # Sharpe breach tracking: list of booleans per day.
        self._sharpe_below_reduction: deque[bool] = deque(
            maxlen=settings.SHARPE_REDUCTION_CONSECUTIVE_DAYS,
        )
        self._sharpe_below_halt: deque[bool] = deque(
            maxlen=settings.SHARPE_HALT_CONSECUTIVE_DAYS,
        )

        # Operational flags.
        self.paper_mode_forced: bool = False
        self.size_reduction_active: bool = False

        logger.info(
            "RiskManager initialised | portfolio_value={v} | paper_mode={p}",
            v=portfolio_value,
            p=self.paper_mode_forced,
        )

    # --------------------------------------------------------------------- #
    # 1. Kelly Criterion Position Sizing
    # --------------------------------------------------------------------- #

    def record_trade_result(self, pnl: float) -> None:
        """Append a closed-trade PnL to the rolling window.

        Parameters
        ----------
        pnl : float
            Net profit/loss of the closed trade (positive = win).
        """
        self._trade_results.append(pnl)

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Return the half-Kelly position fraction, capped at the config max.

        The Kelly Criterion optimal fraction is::

            f* = (p * b - q) / b

        where *p* = win probability, *q* = 1 - p, and
        *b* = ratio of average win to average loss.

        We take **half-Kelly** (``settings.KELLY_FRACTION * f*``) and clamp
        the result to ``[0, settings.MAX_POSITION_FRACTION]``.

        Parameters
        ----------
        win_rate : float
            Probability of a winning trade (0-1).
        avg_win : float
            Mean profit on winning trades (positive).
        avg_loss : float
            Mean loss on losing trades (positive magnitude).

        Returns
        -------
        float
            Recommended fraction of portfolio to risk on the next trade.
        """
        if avg_loss <= 0.0 or avg_win <= 0.0:
            logger.warning(
                "Kelly inputs invalid (avg_win={w}, avg_loss={l}) -> 0.0",
                w=avg_win,
                l=avg_loss,
            )
            return 0.0

        p = win_rate
        q = 1.0 - p
        b = avg_win / avg_loss

        f_star = (p * b - q) / b
        half_kelly = settings.KELLY_FRACTION * f_star

        # Clamp to [0, MAX_POSITION_FRACTION].
        clamped = float(np.clip(half_kelly, 0.0, settings.MAX_POSITION_FRACTION))
        logger.debug(
            "Kelly | p={p:.3f} b={b:.3f} f*={fs:.4f} half={hk:.4f} clamped={c:.4f}",
            p=p,
            b=b,
            fs=f_star,
            hk=half_kelly,
            c=clamped,
        )
        return clamped

    def kelly_from_history(self) -> float:
        """Derive Kelly fraction from the rolling 50-trade window.

        Returns 0.0 if the window contains fewer than 2 trades.

        Returns
        -------
        float
            Recommended position fraction.
        """
        results = list(self._trade_results)
        if len(results) < 2:
            logger.debug("Not enough trades for Kelly ({n})", n=len(results))
            return 0.0

        wins = [r for r in results if r > 0.0]
        losses = [abs(r) for r in results if r <= 0.0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(results)
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean(losses))

        return self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

    # --------------------------------------------------------------------- #
    # 2. Stop-Loss
    # --------------------------------------------------------------------- #

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
    ) -> float:
        """Compute a fixed ATR-based stop-loss price.

        Parameters
        ----------
        entry_price : float
            Execution price of the entry order.
        direction : str
            ``"LONG"`` or ``"SHORT"``.
        atr : float
            Current ATR(14) value for the instrument.

        Returns
        -------
        float
            Stop-loss price level.
        """
        stop_distance = atr * settings.STOP_LOSS_ATR_MULT

        if direction.upper() == "LONG":
            stop = entry_price - stop_distance
        else:
            stop = entry_price + stop_distance

        logger.debug(
            "StopLoss | entry={e} dir={d} atr={a} stop={s}",
            e=entry_price,
            d=direction,
            a=atr,
            s=stop,
        )
        return round(stop, 6)

    # --------------------------------------------------------------------- #
    # 3. Take-Profit
    # --------------------------------------------------------------------- #

    def calculate_take_profit(
        self,
        entry_price: float,
        direction: str,
        atr: float,
    ) -> Optional[float]:
        """Compute an ATR-based take-profit price.

        The take-profit distance is ``ATR * TAKE_PROFIT_ATR_MULT``.  The
        implied risk/reward ratio must meet ``MIN_RISK_REWARD`` or the trade
        is rejected (returns ``None``).

        Parameters
        ----------
        entry_price : float
            Execution price.
        direction : str
            ``"LONG"`` or ``"SHORT"``.
        atr : float
            Current ATR(14).

        Returns
        -------
        float or None
            Take-profit price, or ``None`` if the risk/reward ratio is below
            the configured minimum.
        """
        tp_distance = atr * settings.TAKE_PROFIT_ATR_MULT
        stop_distance = atr * settings.STOP_LOSS_ATR_MULT

        if stop_distance <= 0.0:
            logger.warning("Stop distance is zero; cannot compute R/R")
            return None

        risk_reward = tp_distance / stop_distance
        if risk_reward < settings.MIN_RISK_REWARD:
            logger.warning(
                "R/R {rr:.2f} < minimum {m} -> rejecting take-profit",
                rr=risk_reward,
                m=settings.MIN_RISK_REWARD,
            )
            return None

        if direction.upper() == "LONG":
            tp = entry_price + tp_distance
        else:
            tp = entry_price - tp_distance

        logger.debug(
            "TakeProfit | entry={e} dir={d} atr={a} tp={t} R/R={rr:.2f}",
            e=entry_price,
            d=direction,
            a=atr,
            t=tp,
            rr=risk_reward,
        )
        return round(tp, 6)

    # --------------------------------------------------------------------- #
    # 4. Trailing Stop
    # --------------------------------------------------------------------- #

    def update_trailing_stop(
        self,
        current_price: float,
        high_water_mark: float,
        direction: str,
        atr: float,
        entry_price: Optional[float] = None,
    ) -> Optional[float]:
        """Compute an updated trailing-stop price.

        The trailing stop **activates** only when the unrealised gain
        (``current_price - entry_price`` for longs) exceeds
        ``TRAILING_ACTIVATION_ATR_MULT * ATR``.  Once active it trails the
        high-water mark by ``TRAILING_DISTANCE_ATR_MULT * ATR``.

        For longs the *high_water_mark* is the highest price observed since
        entry; for shorts it is the lowest price observed since entry.

        Parameters
        ----------
        current_price : float
            Latest market price.
        high_water_mark : float
            Best price observed since entry (highest for longs, lowest for
            shorts).
        direction : str
            ``"LONG"`` or ``"SHORT"``.
        atr : float
            Current ATR(14).
        entry_price : float, optional
            Original entry price.  When provided the activation check uses
            the actual unrealised gain.  When ``None`` the high-water mark
            itself is used as the entry proxy (i.e. the caller guarantees
            activation has already been confirmed).

        Returns
        -------
        float or None
            New trailing-stop price, or ``None`` if the trailing stop is not
            yet activated.
        """
        activation_threshold = atr * settings.TRAILING_ACTIVATION_ATR_MULT
        trail_distance = atr * settings.TRAILING_DISTANCE_ATR_MULT

        if direction.upper() == "LONG":
            # Unrealised gain for a long = current_price - entry_price.
            ref = entry_price if entry_price is not None else high_water_mark
            unrealised_gain = current_price - ref
            if unrealised_gain < activation_threshold:
                return None  # not enough gain to activate trailing stop
            trailing_stop = high_water_mark - trail_distance
        else:
            # SHORT: gain = entry_price - current_price.
            ref = entry_price if entry_price is not None else high_water_mark
            unrealised_gain = ref - current_price
            if unrealised_gain < activation_threshold:
                return None
            trailing_stop = high_water_mark + trail_distance

        logger.debug(
            "TrailingStop | dir={d} price={p} hwm={h} atr={a} trail={t}",
            d=direction,
            p=current_price,
            h=high_water_mark,
            a=atr,
            t=trailing_stop,
        )
        return round(trailing_stop, 6)

    # --------------------------------------------------------------------- #
    # 5. Portfolio Guards
    # --------------------------------------------------------------------- #

    def check_daily_loss(self, portfolio: Dict) -> bool:
        """Return ``True`` if trading should be **halted** due to daily loss.

        Parameters
        ----------
        portfolio : dict
            Must contain ``"equity"`` (current) and ``"equity_start_of_day"``.

        Returns
        -------
        bool
            ``True`` means *halt trading* (daily loss exceeds limit).
        """
        equity = portfolio.get("equity", 0.0)
        start = portfolio.get("equity_start_of_day", equity)
        if start <= 0.0:
            return False

        daily_loss_pct = (start - equity) / start
        if daily_loss_pct >= settings.MAX_DAILY_LOSS_PCT:
            msg = (
                f"Daily loss {daily_loss_pct:.2%} exceeds limit "
                f"{settings.MAX_DAILY_LOSS_PCT:.2%} -> HALT"
            )
            logger.warning(msg)
            self._store_alert("WARNING", "daily_loss", msg)
            return True
        return False

    def check_max_drawdown(self, equity: float, peak_equity: float) -> bool:
        """Return ``True`` if trading should be **halted** due to drawdown.

        Parameters
        ----------
        equity : float
            Current portfolio equity.
        peak_equity : float
            All-time (or session) peak equity.

        Returns
        -------
        bool
            ``True`` means *halt trading*.
        """
        if peak_equity <= 0.0:
            return False

        drawdown = (peak_equity - equity) / peak_equity
        if drawdown >= settings.MAX_DRAWDOWN_PCT:
            msg = (
                f"Drawdown {drawdown:.2%} exceeds limit "
                f"{settings.MAX_DRAWDOWN_PCT:.2%} -> HALT"
            )
            logger.warning(msg)
            self._store_alert("CRITICAL", "drawdown", msg)
            return True
        return False

    def check_position_limits(self, open_positions: List[Dict]) -> bool:
        """Return ``True`` if the maximum number of open positions is reached.

        Parameters
        ----------
        open_positions : list[dict]
            Each dict must contain at least ``"instrument"``.

        Returns
        -------
        bool
            ``True`` means *reject new trades* (limit reached).
        """
        if len(open_positions) >= settings.MAX_OPEN_POSITIONS:
            logger.info(
                "Position limit reached ({n}/{m})",
                n=len(open_positions),
                m=settings.MAX_OPEN_POSITIONS,
            )
            return True
        return False

    def check_correlated_positions(self, open_positions: List[Dict]) -> bool:
        """Return ``True`` if adding another position would breach correlation limits.

        CL and BZ are treated as correlated (defined in
        ``settings.CORRELATED_GROUPS``).  At most
        ``settings.MAX_CORRELATED_POSITIONS`` instruments from the same
        group may be open simultaneously.

        Parameters
        ----------
        open_positions : list[dict]
            Each dict must contain ``"instrument"`` (e.g. ``"CL"``).

        Returns
        -------
        bool
            ``True`` means *reject* because too many correlated positions.
        """
        instruments = [p["instrument"] for p in open_positions]

        for group in settings.CORRELATED_GROUPS:
            count = sum(1 for inst in instruments if inst in group)
            if count >= settings.MAX_CORRELATED_POSITIONS:
                logger.info(
                    "Correlated-position limit reached for group {g} ({c}/{m})",
                    g=group,
                    c=count,
                    m=settings.MAX_CORRELATED_POSITIONS,
                )
                return True
        return False

    def calculate_var_95(
        self,
        positions: List[Dict],
        returns_history: np.ndarray,
    ) -> float:
        """Estimate 1-day 95 % Value-at-Risk via Monte Carlo simulation.

        Parameters
        ----------
        positions : list[dict]
            Each dict has ``"instrument"``, ``"notional"`` (dollar exposure).
        returns_history : numpy.ndarray
            2-D array of shape ``(n_days, n_instruments)`` with daily returns
            corresponding to ``positions`` in the same column order.

        Returns
        -------
        float
            VaR as a **positive** fraction of total notional (e.g. 0.018 =
            1.8 %).  Returns 0.0 if inputs are insufficient.
        """
        if len(positions) == 0 or returns_history.size == 0:
            return 0.0

        notionals = np.array([p["notional"] for p in positions], dtype=np.float64)
        total_notional = np.sum(np.abs(notionals))
        if total_notional == 0.0:
            return 0.0

        weights = notionals / total_notional

        # Estimate mean and covariance from historical returns.
        mean_returns = np.mean(returns_history, axis=0)
        cov_matrix = np.cov(returns_history, rowvar=False)

        # Handle single-instrument edge case (cov is scalar).
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]])
        if mean_returns.ndim == 0:
            mean_returns = np.array([float(mean_returns)])

        rng = np.random.default_rng(seed=42)
        simulated = rng.multivariate_normal(
            mean_returns,
            cov_matrix,
            size=settings.VAR_SIMULATIONS,
        )  # shape: (n_sims, n_instruments)

        portfolio_returns = simulated @ weights  # shape: (n_sims,)

        # VaR at 95 %: the 5th percentile loss (negative return).
        var_95 = -float(np.percentile(portfolio_returns, 5))
        var_95 = max(var_95, 0.0)  # VaR is expressed as positive loss

        logger.debug(
            "VaR(95%) = {v:.4f} | total_notional={tn:.2f}",
            v=var_95,
            tn=total_notional,
        )
        return var_95

    def calculate_marginal_var(
        self,
        existing_positions: List[Dict],
        proposed_trade: Dict,
        returns_history: np.ndarray,
    ) -> float:
        """Estimate 1-day 95 % VaR with the proposed trade included.

        This gives a better pre-trade risk picture than checking VaR on
        the existing portfolio alone, because it captures the marginal
        impact of the new position on overall portfolio risk.

        Parameters
        ----------
        existing_positions : list[dict]
            Currently open positions, each with ``"instrument"`` and
            ``"notional"``.
        proposed_trade : dict
            Must contain ``"instrument"`` and ``"notional"`` for the new
            trade.
        returns_history : numpy.ndarray
            2-D array of shape ``(n_days, n_instruments)`` where the
            **last column** corresponds to the proposed trade's
            instrument.  The first *N* columns correspond to
            ``existing_positions`` in the same order.

        Returns
        -------
        float
            VaR(95 %) as a positive fraction of total notional for the
            combined portfolio (existing + proposed).  Returns 0.0 if
            inputs are insufficient.
        """
        combined_positions = list(existing_positions) + [proposed_trade]
        return self.calculate_var_95(combined_positions, returns_history)

    def dynamic_stop_adjustment(
        self,
        stop_distance: float,
        returns_history: np.ndarray,
    ) -> float:
        """Adjust a stop distance based on the current volatility regime.

        The method compares the current historical volatility (standard
        deviation of the most recent returns window) against the median
        historical volatility to classify the regime:

        * **High-vol** (hist_vol > 1.5x median): widen stops by 1.3x.
        * **Low-vol**  (hist_vol < 0.5x median): tighten stops by 0.7x.
        * **Normal**: no adjustment.

        Parameters
        ----------
        stop_distance : float
            The base stop distance (e.g. ``ATR * multiplier``).
        returns_history : numpy.ndarray
            1-D array of recent daily returns used to estimate volatility.

        Returns
        -------
        float
            Adjusted stop distance.
        """
        if returns_history.size < 2:
            return stop_distance

        # Rolling volatilities: use a window equal to 1/4 of the history
        # length (minimum 2) to build a distribution of vol estimates.
        window = max(2, len(returns_history) // 4)
        vols: List[float] = []
        for i in range(window, len(returns_history) + 1):
            segment = returns_history[i - window : i]
            vols.append(float(np.std(segment, ddof=1)))

        if len(vols) < 2:
            return stop_distance

        current_vol = vols[-1]
        median_vol = float(np.median(vols))

        if median_vol <= 0.0:
            return stop_distance

        if current_vol > 1.5 * median_vol:
            multiplier = 1.3
            label = "high-vol"
        elif current_vol < 0.5 * median_vol:
            multiplier = 0.7
            label = "low-vol"
        else:
            multiplier = 1.0
            label = "normal"

        adjusted = stop_distance * multiplier
        logger.debug(
            "DynamicStop | regime={lbl} cur_vol={cv:.6f} med_vol={mv:.6f} "
            "mult={m} base={b:.6f} adjusted={a:.6f}",
            lbl=label,
            cv=current_vol,
            mv=median_vol,
            m=multiplier,
            b=stop_distance,
            a=adjusted,
        )
        return adjusted

    # --------------------------------------------------------------------- #
    # 6. Sharpe Monitoring (rolling 30-day)
    # --------------------------------------------------------------------- #

    def calculate_rolling_sharpe(
        self,
        daily_returns: np.ndarray,
        risk_free_rate: float = settings.RISK_FREE_RATE,
    ) -> float:
        """Compute the annualised Sharpe ratio over a rolling window.

        Parameters
        ----------
        daily_returns : numpy.ndarray
            Array of daily portfolio returns (e.g. last 30 days).
        risk_free_rate : float
            Annualised risk-free rate (default from settings).

        Returns
        -------
        float
            Annualised Sharpe ratio.  Returns 0.0 when standard deviation
            is zero or data is insufficient.
        """
        if len(daily_returns) < 2:
            return 0.0

        window = daily_returns[-settings.SHARPE_ROLLING_DAYS:]
        daily_rf = risk_free_rate / 252.0
        excess = window - daily_rf
        std = float(np.std(excess, ddof=1))

        if std == 0.0:
            return 0.0

        sharpe = (float(np.mean(excess)) / std) * np.sqrt(252)
        return float(sharpe)

    def update_sharpe_state(self, current_sharpe: float) -> None:
        """Update internal Sharpe-breach trackers and set operational flags.

        Call this once per trading day after computing the rolling Sharpe.

        Parameters
        ----------
        current_sharpe : float
            Today's rolling Sharpe ratio.
        """
        # Track consecutive days below reduction threshold.
        self._sharpe_below_reduction.append(
            current_sharpe < settings.SHARPE_REDUCTION_THRESHOLD
        )
        self._sharpe_below_halt.append(
            current_sharpe < settings.SHARPE_HALT_THRESHOLD
        )

        # Check reduction condition: Sharpe < 0.5 for N consecutive days.
        if (
            len(self._sharpe_below_reduction)
            == settings.SHARPE_REDUCTION_CONSECUTIVE_DAYS
            and all(self._sharpe_below_reduction)
        ):
            if not self.size_reduction_active:
                msg = (
                    f"Sharpe < {settings.SHARPE_REDUCTION_THRESHOLD} for "
                    f"{settings.SHARPE_REDUCTION_CONSECUTIVE_DAYS} consecutive "
                    f"days -> reducing position sizes by "
                    f"{settings.SHARPE_POSITION_REDUCTION:.0%}"
                )
                logger.warning(msg)
                self._store_alert("WARNING", "sharpe", msg)
            self.size_reduction_active = True
        else:
            self.size_reduction_active = False

        # Check halt condition: Sharpe < 0 for M consecutive days.
        if (
            len(self._sharpe_below_halt)
            == settings.SHARPE_HALT_CONSECUTIVE_DAYS
            and all(self._sharpe_below_halt)
        ):
            if not self.paper_mode_forced:
                msg = (
                    f"Sharpe < {settings.SHARPE_HALT_THRESHOLD} for "
                    f"{settings.SHARPE_HALT_CONSECUTIVE_DAYS} consecutive "
                    f"days -> switching to PAPER MODE"
                )
                logger.warning(msg)
                self._store_alert("CRITICAL", "sharpe", msg)
            self.paper_mode_forced = True
        else:
            self.paper_mode_forced = False

        logger.debug(
            "Sharpe state | sharpe={s:.3f} reduce={r} paper={p}",
            s=current_sharpe,
            r=self.size_reduction_active,
            p=self.paper_mode_forced,
        )

    # --------------------------------------------------------------------- #
    # 7. Master Gate
    # --------------------------------------------------------------------- #

    def evaluate_trade(
        self,
        trade_params: Dict,
    ) -> Tuple[bool, str, float]:
        """Run every risk check and return a final verdict.

        This is the **single entry-point** that the execution layer must call
        before submitting any order.

        Parameters
        ----------
        trade_params : dict
            Required keys:

            * ``instrument`` (str) -- e.g. ``"CL"``.
            * ``direction`` (str) -- ``"LONG"`` or ``"SHORT"``.
            * ``entry_price`` (float).
            * ``atr`` (float) -- current ATR(14).
            * ``portfolio`` (dict) -- with ``"equity"`` and
              ``"equity_start_of_day"``.
            * ``peak_equity`` (float).
            * ``open_positions`` (list[dict]) -- each with ``"instrument"``
              and ``"notional"``.
            * ``returns_history`` (numpy.ndarray) -- for VaR.
            * ``daily_returns`` (numpy.ndarray) -- for Sharpe.
            * ``requested_size`` (float) -- desired dollar notional.

        Returns
        -------
        tuple[bool, str, float]
            ``(approved, reason, adjusted_size)`` where *approved* is
            ``False`` when the trade must **not** be placed and *reason*
            explains why.  *adjusted_size* may be smaller than the requested
            size due to Kelly / Sharpe adjustments.
        """
        instrument = trade_params["instrument"]
        direction = trade_params["direction"]
        entry_price = trade_params["entry_price"]
        atr = trade_params["atr"]
        portfolio = trade_params["portfolio"]
        peak_equity = trade_params["peak_equity"]
        open_positions = trade_params["open_positions"]
        returns_history = trade_params.get("returns_history", np.array([]))
        daily_returns = trade_params.get("daily_returns", np.array([]))
        requested_size = trade_params["requested_size"]

        # -- Paper-mode gate ------------------------------------------------
        if self.paper_mode_forced:
            return (False, "Paper-mode enforced (negative Sharpe streak)", 0.0)

        # -- Daily loss guard -----------------------------------------------
        if self.check_daily_loss(portfolio):
            return (False, "Daily loss limit breached", 0.0)

        # -- Drawdown guard -------------------------------------------------
        equity = portfolio.get("equity", 0.0)
        if self.check_max_drawdown(equity, peak_equity):
            return (False, "Max drawdown limit breached", 0.0)

        # -- Position count guard -------------------------------------------
        if self.check_position_limits(open_positions):
            return (
                False,
                f"Max open positions ({settings.MAX_OPEN_POSITIONS}) reached",
                0.0,
            )

        # -- Correlated position guard --------------------------------------
        # Simulate adding the proposed instrument.
        simulated_positions = open_positions + [{"instrument": instrument}]
        if self.check_correlated_positions(simulated_positions):
            return (
                False,
                f"Correlated position limit for {instrument} reached",
                0.0,
            )

        # -- Take-profit feasibility (R/R check) ---------------------------
        tp = self.calculate_take_profit(entry_price, direction, atr)
        if tp is None:
            return (False, "Risk/reward ratio below minimum", 0.0)

        # -- VaR guard (marginal VaR when possible) --------------------------
        if returns_history.size > 0 and len(open_positions) > 0:
            proposed_trade = {
                "instrument": instrument,
                "notional": requested_size,
            }
            var = self.calculate_marginal_var(
                open_positions, proposed_trade, returns_history,
            )
            if var > settings.VAR_95_DAILY_LIMIT:
                msg = (
                    f"Marginal VaR(95%) {var:.4f} exceeds limit "
                    f"{settings.VAR_95_DAILY_LIMIT}"
                )
                logger.warning(msg)
                self._store_alert("WARNING", "var", msg)
                return (False, msg, 0.0)

        # -- Kelly-based sizing ---------------------------------------------
        kelly_frac = self.kelly_from_history()
        if kelly_frac > 0.0 and self.portfolio_value > 0.0:
            kelly_size = kelly_frac * self.portfolio_value
            adjusted_size = min(requested_size, kelly_size)
        else:
            adjusted_size = requested_size

        # Cap at MAX_POSITION_FRACTION of portfolio regardless.
        max_notional = settings.MAX_POSITION_FRACTION * self.portfolio_value
        adjusted_size = min(adjusted_size, max_notional)

        # -- Sharpe reduction -----------------------------------------------
        if self.size_reduction_active:
            adjusted_size *= settings.SHARPE_POSITION_REDUCTION
            logger.info(
                "Size reduced by {r:.0%} due to low Sharpe",
                r=settings.SHARPE_POSITION_REDUCTION,
            )

        # -- Regime-aware sizing --------------------------------------------
        regime: Optional[MarketRegime] = trade_params.get("regime")
        if regime is not None:
            if regime == MarketRegime.TRENDING:
                regime_mult = settings.TRENDING_POSITION_MULTIPLIER
            elif regime == MarketRegime.CHOPPY:
                regime_mult = 1.0 - settings.CHOPPY_POSITION_REDUCTION
            else:
                # MEAN_REVERTING or any future default
                regime_mult = 1.0
            adjusted_size *= regime_mult
            logger.info(
                "Regime adjustment | regime={r} multiplier={m:.2f} "
                "adjusted_size={s:.2f}",
                r=regime.value,
                m=regime_mult,
                s=adjusted_size,
            )

        # -- Final sanity ---------------------------------------------------
        if adjusted_size <= 0.0:
            return (False, "Adjusted position size is zero", 0.0)

        stop = self.calculate_stop_loss(entry_price, direction, atr)
        logger.info(
            "Trade APPROVED | {inst} {dir} size={sz:.2f} stop={sl} tp={tp}",
            inst=instrument,
            dir=direction,
            sz=adjusted_size,
            sl=stop,
            tp=tp,
        )
        return (True, "Approved", adjusted_size)

    # --------------------------------------------------------------------- #
    # 8. Alert Persistence
    # --------------------------------------------------------------------- #

    def _store_alert(self, level: str, category: str, message: str) -> None:
        """Persist a :class:`SystemAlert` record to the database.

        Parameters
        ----------
        level : str
            One of ``"INFO"``, ``"WARNING"``, ``"CRITICAL"``.
        category : str
            Short tag such as ``"drawdown"``, ``"sharpe"``, ``"var"``, etc.
        message : str
            Human-readable description.
        """
        try:
            session = get_session()
            alert = SystemAlert(
                timestamp=datetime.utcnow(),
                level=level,
                category=category,
                message=message,
                acknowledged=False,
            )
            session.add(alert)
            session.commit()
            logger.debug("Stored SystemAlert: [{level}] {cat}", level=level, cat=category)
        except Exception as exc:
            logger.error("Failed to store SystemAlert: {e}", e=exc)
        finally:
            try:
                session.close()
            except Exception:
                pass

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def update_portfolio_value(self, new_value: float) -> None:
        """Update the cached portfolio value used for sizing calculations.

        Parameters
        ----------
        new_value : float
            Current total portfolio equity.
        """
        self.portfolio_value = new_value
        logger.debug("Portfolio value updated to {v}", v=new_value)
