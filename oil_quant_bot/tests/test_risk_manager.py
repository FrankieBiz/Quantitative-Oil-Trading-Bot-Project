"""Comprehensive unit tests for oil_quant_bot.risk.manager.RiskManager.

Tests are self-contained: external dependencies (DB sessions, config settings)
are mocked so the suite runs without a live database or env-specific config.
"""

from __future__ import annotations

import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup -- allow importing from oil_quant_bot without installation.
# ---------------------------------------------------------------------------
_OIL_QUANT_BOT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir,
)
sys.path.insert(0, os.path.abspath(_OIL_QUANT_BOT_DIR))

# ---------------------------------------------------------------------------
# Provide a fake ``config.settings`` module BEFORE importing the RiskManager,
# so we never need the real settings file (which may depend on .env / secrets).
# ---------------------------------------------------------------------------
_FAKE_SETTINGS = SimpleNamespace(
    KELLY_FRACTION=0.5,
    KELLY_ROLLING_TRADES=50,
    MAX_POSITION_FRACTION=0.15,
    STOP_LOSS_ATR_MULT=2.0,
    TAKE_PROFIT_ATR_MULT=3.5,
    MIN_RISK_REWARD=1.75,
    TRAILING_ACTIVATION_ATR_MULT=1.5,
    TRAILING_DISTANCE_ATR_MULT=1.0,
    MAX_DAILY_LOSS_PCT=0.03,
    MAX_DRAWDOWN_PCT=0.12,
    MAX_OPEN_POSITIONS=4,
    MAX_CORRELATED_POSITIONS=2,
    CORRELATED_GROUPS=[{"CL", "BZ"}],
    VAR_95_DAILY_LIMIT=0.02,
    VAR_SIMULATIONS=10_000,
    SHARPE_ROLLING_DAYS=30,
    SHARPE_REDUCTION_THRESHOLD=0.5,
    SHARPE_REDUCTION_CONSECUTIVE_DAYS=5,
    SHARPE_HALT_THRESHOLD=0.0,
    SHARPE_HALT_CONSECUTIVE_DAYS=10,
    SHARPE_POSITION_REDUCTION=0.5,
    RISK_FREE_RATE=0.05,
)

sys.modules.setdefault("config", MagicMock())
sys.modules["config"].settings = _FAKE_SETTINGS
sys.modules.setdefault("config.settings", _FAKE_SETTINGS)

# Fake db.models so we never touch a real database.
_fake_db = MagicMock()
sys.modules.setdefault("db", _fake_db)
sys.modules.setdefault("db.models", _fake_db)
# Provide the names that ``risk.manager`` imports directly.
_fake_db.SystemAlert = MagicMock
_fake_db.Trade = MagicMock
_fake_db.TradeDirection = MagicMock
_fake_db.TradeStatus = MagicMock
_fake_db.get_session = MagicMock(return_value=MagicMock())

from risk.manager import RiskManager  # noqa: E402

# Alias for readability.
settings = _FAKE_SETTINGS


# ========================================================================= #
# Fixtures
# ========================================================================= #

@pytest.fixture
def rm() -> RiskManager:
    """Fresh RiskManager with a $100 000 portfolio."""
    return RiskManager(portfolio_value=100_000.0)


def _make_trade_params(**overrides) -> dict:
    """Return a valid ``trade_params`` dict for :meth:`evaluate_trade`.

    Override any key by passing it as a keyword argument.
    """
    defaults = dict(
        instrument="CL",
        direction="LONG",
        entry_price=75.0,
        atr=1.5,
        portfolio={"equity": 100_000.0, "equity_start_of_day": 100_000.0},
        peak_equity=100_000.0,
        open_positions=[],
        returns_history=np.array([]),
        daily_returns=np.array([]),
        requested_size=10_000.0,
    )
    defaults.update(overrides)
    return defaults


# ========================================================================= #
# 1. Kelly Criterion
# ========================================================================= #

class TestKellyCriterion:
    """Tests for ``calculate_kelly_fraction`` and ``kelly_from_history``."""

    def test_basic_kelly(self, rm: RiskManager):
        """Standard inputs: 60% win rate, 2:1 avg_win/avg_loss."""
        # f* = (p*b - q) / b = (0.6*2 - 0.4)/2 = (1.2-0.4)/2 = 0.4
        # half-Kelly = 0.5 * 0.4 = 0.2 -> capped at 0.15
        frac = rm.calculate_kelly_fraction(win_rate=0.6, avg_win=200.0, avg_loss=100.0)
        assert frac == pytest.approx(0.15)  # capped

    def test_half_kelly_below_cap(self, rm: RiskManager):
        """Half-Kelly lands below the 15% cap."""
        # b = 1.5, f* = (0.55*1.5 - 0.45)/1.5 = (0.825-0.45)/1.5 = 0.25
        # half = 0.5*0.25 = 0.125
        frac = rm.calculate_kelly_fraction(win_rate=0.55, avg_win=150.0, avg_loss=100.0)
        assert frac == pytest.approx(0.125)

    def test_kelly_capped_at_max(self, rm: RiskManager):
        """Very strong edge must be capped at MAX_POSITION_FRACTION (15%)."""
        frac = rm.calculate_kelly_fraction(win_rate=0.9, avg_win=500.0, avg_loss=50.0)
        assert frac == pytest.approx(settings.MAX_POSITION_FRACTION)

    def test_negative_kelly_returns_zero(self, rm: RiskManager):
        """When edge is negative, Kelly fraction is clamped to 0."""
        # b = 0.5, f* = (0.3*0.5 - 0.7)/0.5 = (0.15-0.7)/0.5 = -1.1
        frac = rm.calculate_kelly_fraction(win_rate=0.3, avg_win=50.0, avg_loss=100.0)
        assert frac == 0.0

    def test_zero_avg_loss_returns_zero(self, rm: RiskManager):
        frac = rm.calculate_kelly_fraction(win_rate=0.6, avg_win=100.0, avg_loss=0.0)
        assert frac == 0.0

    def test_zero_avg_win_returns_zero(self, rm: RiskManager):
        frac = rm.calculate_kelly_fraction(win_rate=0.6, avg_win=0.0, avg_loss=100.0)
        assert frac == 0.0

    def test_negative_avg_loss_returns_zero(self, rm: RiskManager):
        frac = rm.calculate_kelly_fraction(win_rate=0.6, avg_win=100.0, avg_loss=-50.0)
        assert frac == 0.0

    def test_fifty_percent_win_rate_1_to_1(self, rm: RiskManager):
        """50% win rate with 1:1 R/R -> f*=0 -> fraction=0."""
        frac = rm.calculate_kelly_fraction(win_rate=0.5, avg_win=100.0, avg_loss=100.0)
        assert frac == 0.0

    def test_kelly_from_history_insufficient_data(self, rm: RiskManager):
        """Fewer than 2 trades -> 0."""
        rm.record_trade_result(100.0)
        assert rm.kelly_from_history() == 0.0

    def test_kelly_from_history_all_wins(self, rm: RiskManager):
        """All wins, no losses -> 0 (division guard)."""
        for _ in range(10):
            rm.record_trade_result(50.0)
        assert rm.kelly_from_history() == 0.0

    def test_kelly_from_history_all_losses(self, rm: RiskManager):
        """All losses, no wins -> 0."""
        for _ in range(10):
            rm.record_trade_result(-50.0)
        assert rm.kelly_from_history() == 0.0

    def test_kelly_from_history_mixed(self, rm: RiskManager):
        """Mixed results should produce a sensible fraction."""
        wins = [100.0] * 6
        losses = [-50.0] * 4
        for pnl in wins + losses:
            rm.record_trade_result(pnl)
        frac = rm.kelly_from_history()
        # win_rate=0.6, avg_win=100, avg_loss=50 -> b=2
        # f*=(0.6*2-0.4)/2=0.4, half=0.2 -> capped at 0.15
        assert frac == pytest.approx(0.15)


# ========================================================================= #
# 2. Stop-Loss
# ========================================================================= #

class TestStopLoss:

    def test_long_stop_loss(self, rm: RiskManager):
        stop = rm.calculate_stop_loss(entry_price=75.0, direction="LONG", atr=1.5)
        # 75 - 1.5*2.0 = 75 - 3 = 72
        assert stop == pytest.approx(72.0)

    def test_short_stop_loss(self, rm: RiskManager):
        stop = rm.calculate_stop_loss(entry_price=75.0, direction="SHORT", atr=1.5)
        # 75 + 1.5*2.0 = 78
        assert stop == pytest.approx(78.0)

    def test_stop_loss_case_insensitive(self, rm: RiskManager):
        stop = rm.calculate_stop_loss(entry_price=80.0, direction="long", atr=2.0)
        assert stop == pytest.approx(80.0 - 2.0 * settings.STOP_LOSS_ATR_MULT)

    def test_stop_loss_different_atr(self, rm: RiskManager):
        stop = rm.calculate_stop_loss(entry_price=100.0, direction="LONG", atr=5.0)
        assert stop == pytest.approx(100.0 - 5.0 * 2.0)

    def test_stop_loss_rounding(self, rm: RiskManager):
        stop = rm.calculate_stop_loss(entry_price=75.123456789, direction="LONG", atr=1.0)
        # Should be rounded to 6 decimals.
        expected = round(75.123456789 - 1.0 * 2.0, 6)
        assert stop == pytest.approx(expected)


# ========================================================================= #
# 3. Take-Profit
# ========================================================================= #

class TestTakeProfit:

    def test_long_take_profit(self, rm: RiskManager):
        tp = rm.calculate_take_profit(entry_price=75.0, direction="LONG", atr=1.5)
        # tp_distance = 1.5 * 3.5 = 5.25; R/R = 3.5/2.0 = 1.75 >= 1.75 OK
        assert tp == pytest.approx(75.0 + 5.25)

    def test_short_take_profit(self, rm: RiskManager):
        tp = rm.calculate_take_profit(entry_price=75.0, direction="SHORT", atr=1.5)
        assert tp == pytest.approx(75.0 - 5.25)

    def test_rr_rejection(self, rm: RiskManager):
        """When R/R < MIN_RISK_REWARD, return None.

        We temporarily lower TAKE_PROFIT_ATR_MULT so that the ratio
        falls below the threshold.
        """
        original = settings.TAKE_PROFIT_ATR_MULT
        try:
            # R/R = 1.0 / 2.0 = 0.5 < 1.75
            settings.TAKE_PROFIT_ATR_MULT = 1.0
            tp = rm.calculate_take_profit(entry_price=75.0, direction="LONG", atr=1.5)
            assert tp is None
        finally:
            settings.TAKE_PROFIT_ATR_MULT = original

    def test_take_profit_zero_atr(self, rm: RiskManager):
        """ATR of 0 -> stop_distance = 0 -> returns None."""
        tp = rm.calculate_take_profit(entry_price=75.0, direction="LONG", atr=0.0)
        assert tp is None


# ========================================================================= #
# 4. Trailing Stop
# ========================================================================= #

class TestTrailingStop:

    def test_long_not_activated(self, rm: RiskManager):
        """Gain below activation threshold -> None."""
        # activation = 1.5 * 1.5 = 2.25; gain = 76 - 75 = 1.0 < 2.25
        result = rm.update_trailing_stop(
            current_price=76.0, high_water_mark=76.0,
            direction="LONG", atr=1.5, entry_price=75.0,
        )
        assert result is None

    def test_long_activated(self, rm: RiskManager):
        """Gain above activation -> trailing stop computed."""
        # activation = 2.25; gain = 78 - 75 = 3 >= 2.25
        # trail = hwm - trail_distance = 78.5 - 1.5*1.0 = 77.0
        result = rm.update_trailing_stop(
            current_price=78.0, high_water_mark=78.5,
            direction="LONG", atr=1.5, entry_price=75.0,
        )
        assert result == pytest.approx(77.0)

    def test_short_not_activated(self, rm: RiskManager):
        # activation = 2.25; gain = 75 - 74 = 1 < 2.25
        result = rm.update_trailing_stop(
            current_price=74.0, high_water_mark=74.0,
            direction="SHORT", atr=1.5, entry_price=75.0,
        )
        assert result is None

    def test_short_activated(self, rm: RiskManager):
        # activation = 2.25; gain = 75 - 72 = 3 >= 2.25
        # trail = hwm + trail_distance = 71.5 + 1.5 = 73.0
        result = rm.update_trailing_stop(
            current_price=72.0, high_water_mark=71.5,
            direction="SHORT", atr=1.5, entry_price=75.0,
        )
        assert result == pytest.approx(73.0)

    def test_long_no_entry_price(self, rm: RiskManager):
        """When entry_price is None, hwm is used as ref -> gain is always 0 for
        current_price <= hwm, so activation depends on current_price vs hwm."""
        # ref = hwm = 78.5; gain = 78.0 - 78.5 = -0.5 < 2.25 -> None
        result = rm.update_trailing_stop(
            current_price=78.0, high_water_mark=78.5,
            direction="LONG", atr=1.5, entry_price=None,
        )
        assert result is None

    def test_trail_distance(self, rm: RiskManager):
        """Verify trail distance = ATR * TRAILING_DISTANCE_ATR_MULT."""
        atr = 2.0
        entry = 70.0
        hwm = 76.0
        current = 75.5
        # activation = 2.0 * 1.5 = 3.0; gain = 75.5 - 70 = 5.5 >= 3.0
        # trail = 76.0 - 2.0*1.0 = 74.0
        result = rm.update_trailing_stop(
            current_price=current, high_water_mark=hwm,
            direction="LONG", atr=atr, entry_price=entry,
        )
        assert result == pytest.approx(74.0)


# ========================================================================= #
# 5. Portfolio Guards
# ========================================================================= #

class TestPortfolioGuards:

    # -- Daily loss ---

    def test_daily_loss_below_limit(self, rm: RiskManager):
        portfolio = {"equity": 98_000.0, "equity_start_of_day": 100_000.0}
        # loss = 2% < 3%
        assert rm.check_daily_loss(portfolio) is False

    def test_daily_loss_at_limit(self, rm: RiskManager):
        portfolio = {"equity": 97_000.0, "equity_start_of_day": 100_000.0}
        # loss = 3% == 3% -> halt
        assert rm.check_daily_loss(portfolio) is True

    def test_daily_loss_above_limit(self, rm: RiskManager):
        portfolio = {"equity": 96_000.0, "equity_start_of_day": 100_000.0}
        assert rm.check_daily_loss(portfolio) is True

    def test_daily_loss_positive_day(self, rm: RiskManager):
        portfolio = {"equity": 102_000.0, "equity_start_of_day": 100_000.0}
        assert rm.check_daily_loss(portfolio) is False

    def test_daily_loss_zero_start(self, rm: RiskManager):
        portfolio = {"equity": 100.0, "equity_start_of_day": 0.0}
        assert rm.check_daily_loss(portfolio) is False

    # -- Max drawdown ---

    def test_drawdown_below_limit(self, rm: RiskManager):
        assert rm.check_max_drawdown(equity=90_000.0, peak_equity=100_000.0) is False

    def test_drawdown_at_limit(self, rm: RiskManager):
        # 12% drawdown
        assert rm.check_max_drawdown(equity=88_000.0, peak_equity=100_000.0) is True

    def test_drawdown_above_limit(self, rm: RiskManager):
        assert rm.check_max_drawdown(equity=85_000.0, peak_equity=100_000.0) is True

    def test_drawdown_zero_peak(self, rm: RiskManager):
        assert rm.check_max_drawdown(equity=100.0, peak_equity=0.0) is False

    # -- Position limits ---

    def test_position_limit_below(self, rm: RiskManager):
        positions = [{"instrument": "CL"}, {"instrument": "BZ"}, {"instrument": "NG"}]
        assert rm.check_position_limits(positions) is False  # 3 < 4

    def test_position_limit_at_max(self, rm: RiskManager):
        positions = [{"instrument": f"X{i}"} for i in range(4)]
        assert rm.check_position_limits(positions) is True  # 4 >= 4

    def test_position_limit_above_max(self, rm: RiskManager):
        positions = [{"instrument": f"X{i}"} for i in range(5)]
        assert rm.check_position_limits(positions) is True

    def test_position_limit_empty(self, rm: RiskManager):
        assert rm.check_position_limits([]) is False

    # -- Correlated positions ---

    def test_correlated_below_limit(self, rm: RiskManager):
        # Only 1 from {"CL", "BZ"} -> OK
        positions = [{"instrument": "CL"}, {"instrument": "NG"}]
        assert rm.check_correlated_positions(positions) is False

    def test_correlated_at_limit(self, rm: RiskManager):
        # 2 from {"CL", "BZ"} -> breach
        positions = [{"instrument": "CL"}, {"instrument": "BZ"}]
        assert rm.check_correlated_positions(positions) is True

    def test_correlated_no_group_match(self, rm: RiskManager):
        positions = [{"instrument": "NG"}, {"instrument": "GC"}]
        assert rm.check_correlated_positions(positions) is False

    def test_correlated_empty(self, rm: RiskManager):
        assert rm.check_correlated_positions([]) is False


# ========================================================================= #
# 6. VaR Calculation
# ========================================================================= #

class TestVaR:

    def test_var_empty_positions(self, rm: RiskManager):
        assert rm.calculate_var_95([], np.array([])) == 0.0

    def test_var_empty_returns(self, rm: RiskManager):
        positions = [{"instrument": "CL", "notional": 10_000.0}]
        assert rm.calculate_var_95(positions, np.array([])) == 0.0

    def test_var_zero_notional(self, rm: RiskManager):
        positions = [{"instrument": "CL", "notional": 0.0}]
        returns = np.random.default_rng(0).normal(0, 0.01, (60, 1))
        assert rm.calculate_var_95(positions, returns) == 0.0

    def test_var_single_instrument_reasonable(self, rm: RiskManager):
        """VaR should be a small positive number for moderate volatility."""
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0005, 0.015, (252, 1))
        positions = [{"instrument": "CL", "notional": 50_000.0}]
        var = rm.calculate_var_95(positions, returns)
        assert var > 0.0
        assert var < 0.10  # less than 10% for daily VaR

    def test_var_multi_instrument(self, rm: RiskManager):
        """VaR with 2 instruments should produce a positive value."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, (120, 2))
        positions = [
            {"instrument": "CL", "notional": 30_000.0},
            {"instrument": "BZ", "notional": 20_000.0},
        ]
        var = rm.calculate_var_95(positions, returns)
        assert var > 0.0
        assert var < 0.15

    def test_var_deterministic_with_seed(self, rm: RiskManager):
        """Same inputs -> same VaR (seed is fixed at 42 inside the method)."""
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0, 0.01, (100, 1))
        positions = [{"instrument": "CL", "notional": 10_000.0}]
        var1 = rm.calculate_var_95(positions, returns)
        var2 = rm.calculate_var_95(positions, returns)
        assert var1 == pytest.approx(var2)


# ========================================================================= #
# 7. Rolling Sharpe Monitoring
# ========================================================================= #

class TestSharpeMonitoring:

    def test_rolling_sharpe_basic(self, rm: RiskManager):
        """Positive returns should produce a positive Sharpe."""
        daily = np.full(30, 0.002)  # 0.2% per day
        sharpe = rm.calculate_rolling_sharpe(daily)
        assert sharpe > 0.0

    def test_rolling_sharpe_insufficient_data(self, rm: RiskManager):
        assert rm.calculate_rolling_sharpe(np.array([0.01])) == 0.0

    def test_rolling_sharpe_zero_std(self, rm: RiskManager):
        """Constant returns -> std=0 -> 0."""
        daily_rf = settings.RISK_FREE_RATE / 252.0
        daily = np.full(30, daily_rf)  # excess = 0 for every day
        sharpe = rm.calculate_rolling_sharpe(daily)
        assert sharpe == 0.0

    def test_sharpe_reduction_flag(self, rm: RiskManager):
        """5 consecutive days below 0.5 triggers size reduction."""
        for _ in range(settings.SHARPE_REDUCTION_CONSECUTIVE_DAYS):
            rm.update_sharpe_state(0.3)  # < 0.5
        assert rm.size_reduction_active is True

    def test_sharpe_reduction_cleared(self, rm: RiskManager):
        """A good day clears the reduction flag."""
        for _ in range(settings.SHARPE_REDUCTION_CONSECUTIVE_DAYS):
            rm.update_sharpe_state(0.3)
        assert rm.size_reduction_active is True
        rm.update_sharpe_state(0.8)  # above threshold
        assert rm.size_reduction_active is False

    def test_sharpe_halt_paper_mode(self, rm: RiskManager):
        """10 consecutive days below 0 triggers paper mode."""
        for _ in range(settings.SHARPE_HALT_CONSECUTIVE_DAYS):
            rm.update_sharpe_state(-0.5)  # < 0
        assert rm.paper_mode_forced is True

    def test_sharpe_halt_cleared(self, rm: RiskManager):
        """Paper mode cleared when streak breaks."""
        for _ in range(settings.SHARPE_HALT_CONSECUTIVE_DAYS):
            rm.update_sharpe_state(-0.5)
        assert rm.paper_mode_forced is True
        rm.update_sharpe_state(0.1)  # positive
        assert rm.paper_mode_forced is False

    def test_sharpe_not_enough_consecutive_days(self, rm: RiskManager):
        """4 out of 5 days bad should NOT trigger reduction."""
        for _ in range(settings.SHARPE_REDUCTION_CONSECUTIVE_DAYS - 1):
            rm.update_sharpe_state(0.3)
        # Deque not yet full -> no trigger.
        assert rm.size_reduction_active is False


# ========================================================================= #
# 8. evaluate_trade -- Master Gate
# ========================================================================= #

class TestEvaluateTrade:

    def test_approved_basic(self, rm: RiskManager):
        """Clean trade with no breaches -> approved."""
        params = _make_trade_params()
        approved, reason, size = rm.evaluate_trade(params)
        assert approved is True
        assert reason == "Approved"
        assert size > 0.0

    def test_rejected_paper_mode(self, rm: RiskManager):
        rm.paper_mode_forced = True
        approved, reason, _ = rm.evaluate_trade(_make_trade_params())
        assert approved is False
        assert "Paper-mode" in reason

    def test_rejected_daily_loss(self, rm: RiskManager):
        portfolio = {"equity": 96_000.0, "equity_start_of_day": 100_000.0}
        params = _make_trade_params(portfolio=portfolio)
        approved, reason, _ = rm.evaluate_trade(params)
        assert approved is False
        assert "Daily loss" in reason

    def test_rejected_max_drawdown(self, rm: RiskManager):
        portfolio = {"equity": 85_000.0, "equity_start_of_day": 100_000.0}
        params = _make_trade_params(portfolio=portfolio, peak_equity=100_000.0)
        approved, reason, _ = rm.evaluate_trade(params)
        assert approved is False
        assert "drawdown" in reason.lower() or "Daily loss" in reason

    def test_rejected_position_limit(self, rm: RiskManager):
        positions = [{"instrument": f"X{i}"} for i in range(4)]
        params = _make_trade_params(open_positions=positions)
        approved, reason, _ = rm.evaluate_trade(params)
        assert approved is False
        assert "positions" in reason.lower()

    def test_rejected_correlated_limit(self, rm: RiskManager):
        """Adding CL when CL+BZ already open -> reject."""
        positions = [{"instrument": "CL"}, {"instrument": "BZ"}]
        params = _make_trade_params(instrument="CL", open_positions=positions)
        approved, reason, _ = rm.evaluate_trade(params)
        assert approved is False
        assert "Correlated" in reason

    def test_rejected_rr_ratio(self, rm: RiskManager):
        """If R/R is too low, trade is rejected."""
        original = settings.TAKE_PROFIT_ATR_MULT
        try:
            settings.TAKE_PROFIT_ATR_MULT = 1.0  # R/R = 0.5 < 1.75
            params = _make_trade_params()
            approved, reason, _ = rm.evaluate_trade(params)
            assert approved is False
            assert "Risk/reward" in reason
        finally:
            settings.TAKE_PROFIT_ATR_MULT = original

    def test_size_capped_at_max_fraction(self, rm: RiskManager):
        """Requested size > 15% of portfolio -> capped."""
        params = _make_trade_params(requested_size=50_000.0)
        approved, reason, size = rm.evaluate_trade(params)
        assert approved is True
        assert size <= settings.MAX_POSITION_FRACTION * rm.portfolio_value

    def test_size_reduced_by_sharpe(self, rm: RiskManager):
        """When size_reduction_active, the final size is halved."""
        rm.size_reduction_active = True
        params = _make_trade_params(requested_size=10_000.0)
        approved, reason, size = rm.evaluate_trade(params)
        assert approved is True
        assert size == pytest.approx(10_000.0 * settings.SHARPE_POSITION_REDUCTION)

    def test_kelly_constrains_size(self, rm: RiskManager):
        """When Kelly history suggests smaller size, use that."""
        # Build a history with modest edge.
        wins = [60.0] * 6
        losses = [-80.0] * 4
        for pnl in wins + losses:
            rm.record_trade_result(pnl)
        # win_rate=0.6, avg_win=60, avg_loss=80, b=0.75
        # f* = (0.6*0.75-0.4)/0.75 = (0.45-0.4)/0.75 = 0.0667
        # half = 0.5*0.0667 = 0.0333
        # kelly_size = 0.0333 * 100_000 = 3333
        params = _make_trade_params(requested_size=10_000.0)
        approved, reason, size = rm.evaluate_trade(params)
        assert approved is True
        assert size < 10_000.0  # Kelly should limit it
        expected_kelly = 0.5 * ((0.6 * 0.75 - 0.4) / 0.75) * 100_000
        assert size == pytest.approx(expected_kelly, rel=0.01)

    def test_approved_short_direction(self, rm: RiskManager):
        params = _make_trade_params(direction="SHORT")
        approved, reason, size = rm.evaluate_trade(params)
        assert approved is True
        assert size > 0.0

    def test_var_rejection(self, rm: RiskManager):
        """If VaR exceeds the limit, the trade is rejected."""
        # Build returns with high volatility so VaR > 0.02.
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.10, (100, 1))
        positions = [{"instrument": "BZ", "notional": 50_000.0}]
        params = _make_trade_params(
            instrument="NG",
            open_positions=positions,
            returns_history=returns,
        )
        approved, reason, _ = rm.evaluate_trade(params)
        # With std=0.10, VaR should be very high -> rejection.
        assert approved is False
        assert "VaR" in reason

    def test_zero_portfolio_value(self):
        """Zero portfolio value -> adjusted_size = 0 -> rejected."""
        rm = RiskManager(portfolio_value=0.0)
        params = _make_trade_params(requested_size=1000.0)
        approved, reason, size = rm.evaluate_trade(params)
        assert approved is False
        assert size == 0.0
