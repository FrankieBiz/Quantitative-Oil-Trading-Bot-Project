"""
Performance metrics and reporting for the Oil Quantitative Trading Bot.

Calculates all standard risk-adjusted return metrics, generates formatted
reports, and produces interactive Plotly visualisations (equity curve,
drawdown, monthly-returns heatmap).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from config import settings
from db.models import Trade


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR = 252
_MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Core metric calculations
# ---------------------------------------------------------------------------

class BacktestMetrics:
    """Compute and store a comprehensive set of backtest performance metrics.

    Instantiate with an equity curve and (optionally) a trades DataFrame,
    then call :meth:`compute_all` to populate every metric at once, or
    access individual properties / methods.

    Attributes:
        equity: Time-indexed equity Series.
        returns: Period returns derived from ``equity``.
        trades_df: Round-trip trades DataFrame (vectorbt format with a ``PnL``
            column, or a general DataFrame with ``pnl`` / ``realized_pnl``).
        risk_free_rate: Annualised risk-free rate used for Sharpe etc.
        freq_days: Number of calendar days per bar (1 for daily data).
    """

    def __init__(
        self,
        equity: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        risk_free_rate: float = settings.RISK_FREE_RATE,
        freq_days: float = 1.0,
    ) -> None:
        """Initialise metrics from an equity curve.

        Args:
            equity: Time-indexed Series of portfolio equity values.
            trades_df: Optional DataFrame of individual trades.  Recognised
                       PnL column names: ``PnL``, ``pnl``, ``realized_pnl``.
            risk_free_rate: Annualised risk-free rate for excess-return
                            calculations.
            freq_days: Average number of calendar days between bars.
                       Defaults to 1 (daily).
        """
        self.equity = equity
        self.returns = equity.pct_change().dropna()
        self.trades_df = trades_df if trades_df is not None else pd.DataFrame()
        self.risk_free_rate = risk_free_rate
        self.freq_days = freq_days

        # Derived quantities cached on first access
        self._drawdown: Optional[pd.Series] = None
        self._metrics: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Drawdown helpers
    # ------------------------------------------------------------------

    @property
    def drawdown(self) -> pd.Series:
        """Time-indexed drawdown series (negative values indicate losses).

        Returns:
            Series of drawdown fractions from running peak.
        """
        if self._drawdown is None:
            running_max = self.equity.cummax()
            self._drawdown = (self.equity - running_max) / running_max
        return self._drawdown

    # ------------------------------------------------------------------
    # Individual metric methods
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        """Calculate total return as a fraction (e.g. 0.25 = 25 %).

        Returns:
            Total return over the entire equity curve.
        """
        if len(self.equity) < 2:
            return 0.0
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)

    def cagr(self) -> float:
        """Compound Annual Growth Rate.

        Returns:
            CAGR as a decimal fraction.
        """
        if len(self.equity) < 2:
            return 0.0
        n_years = len(self.returns) * self.freq_days / 365.25
        if n_years <= 0:
            return 0.0
        total = self.equity.iloc[-1] / self.equity.iloc[0]
        return float(total ** (1.0 / n_years) - 1)

    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio.

        Uses the daily risk-free rate derived from the annualised setting
        in ``config.settings``.

        Returns:
            Sharpe ratio (annualised).
        """
        if len(self.returns) < 2:
            return 0.0
        periods_per_year = _TRADING_DAYS_PER_YEAR / self.freq_days
        rf_per_period = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
        excess = self.returns - rf_per_period
        mean_excess = float(excess.mean())
        std = float(excess.std(ddof=1))
        if std == 0:
            return 0.0
        return mean_excess / std * np.sqrt(periods_per_year)

    def sortino_ratio(self) -> float:
        """Annualised Sortino ratio (downside deviation only).

        Returns:
            Sortino ratio.
        """
        if len(self.returns) < 2:
            return 0.0
        periods_per_year = _TRADING_DAYS_PER_YEAR / self.freq_days
        rf_per_period = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
        excess = self.returns - rf_per_period
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf") if float(excess.mean()) > 0 else 0.0
        downside_std = float(np.sqrt((downside ** 2).mean()))
        if downside_std == 0:
            return 0.0
        return float(excess.mean()) / downside_std * np.sqrt(periods_per_year)

    def calmar_ratio(self) -> float:
        """Calmar ratio (CAGR / |max drawdown|).

        Returns:
            Calmar ratio.
        """
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return self.cagr() / mdd

    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction (e.g. -0.12 = -12 %).

        Returns:
            Maximum drawdown value.
        """
        if self.drawdown.empty:
            return 0.0
        return float(self.drawdown.min())

    def max_drawdown_duration(self) -> int:
        """Duration (in bars) of the longest drawdown period.

        A drawdown period starts when equity drops below a new high and
        ends when it recovers to a new high.

        Returns:
            Number of bars in the longest underwater spell.
        """
        if self.drawdown.empty:
            return 0
        is_underwater = self.drawdown < 0
        if not is_underwater.any():
            return 0

        # Group consecutive underwater bars
        groups = (~is_underwater).cumsum()
        underwater_groups = groups[is_underwater]
        if underwater_groups.empty:
            return 0
        return int(underwater_groups.value_counts().max())

    def win_rate(self) -> float:
        """Fraction of trades that were profitable.

        Returns:
            Win rate as a decimal (0.0 to 1.0).
        """
        pnls = self._trade_pnls()
        if len(pnls) == 0:
            return 0.0
        return float((pnls > 0).sum() / len(pnls))

    def profit_factor(self) -> float:
        """Gross profit divided by gross loss.

        Returns:
            Profit factor.  Returns ``inf`` if there are no losing trades.
        """
        pnls = self._trade_pnls()
        if len(pnls) == 0:
            return 0.0
        gross_profit = float(pnls[pnls > 0].sum())
        gross_loss = float(abs(pnls[pnls < 0].sum()))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def avg_win(self) -> float:
        """Average profit of winning trades.

        Returns:
            Mean PnL of trades with positive PnL.
        """
        pnls = self._trade_pnls()
        winners = pnls[pnls > 0]
        return float(winners.mean()) if len(winners) > 0 else 0.0

    def avg_loss(self) -> float:
        """Average loss of losing trades (returned as a negative number).

        Returns:
            Mean PnL of trades with negative PnL.
        """
        pnls = self._trade_pnls()
        losers = pnls[pnls < 0]
        return float(losers.mean()) if len(losers) > 0 else 0.0

    def avg_risk_reward(self) -> float:
        """Average reward-to-risk ratio (|avg win| / |avg loss|).

        Returns:
            Risk/reward ratio.
        """
        aw = self.avg_win()
        al = self.avg_loss()
        if al == 0:
            return 0.0
        return abs(aw / al)

    def total_trades(self) -> int:
        """Total number of round-trip trades.

        Returns:
            Trade count.
        """
        return len(self._trade_pnls())

    def expectancy(self) -> float:
        """Calculate trade expectancy: (win_rate * avg_win) - (loss_rate * avg_loss).

        Returns:
            Expected dollar value per trade.
        """
        wr = self.win_rate()
        aw = self.avg_win()
        al = self.avg_loss()
        return wr * aw + (1 - wr) * al  # al is already negative

    def max_consecutive_wins(self) -> int:
        """Maximum consecutive winning trades.

        Returns:
            Length of longest winning streak.
        """
        pnls = self._trade_pnls()
        if len(pnls) == 0:
            return 0
        wins = (pnls > 0).astype(int)
        return int(self._max_consecutive(wins))

    def max_consecutive_losses(self) -> int:
        """Maximum consecutive losing trades.

        Returns:
            Length of longest losing streak.
        """
        pnls = self._trade_pnls()
        if len(pnls) == 0:
            return 0
        losses = (pnls <= 0).astype(int)
        return int(self._max_consecutive(losses))

    @staticmethod
    def _max_consecutive(binary_series: pd.Series) -> int:
        """Count the maximum run of 1s in a binary series."""
        if binary_series.empty:
            return 0
        groups = binary_series.ne(binary_series.shift()).cumsum()
        ones = binary_series[binary_series == 1]
        if ones.empty:
            return 0
        return int(ones.groupby(groups).count().max())

    def avg_trade_duration(self) -> float:
        """Average trade duration in bars.

        Returns:
            Mean number of bars per trade, or 0.0 if unavailable.
        """
        if self.trades_df.empty:
            return 0.0
        for dur_col in ("Duration", "duration"):
            if dur_col in self.trades_df.columns:
                durations = self.trades_df[dur_col].dropna()
                if len(durations) > 0:
                    return float(durations.mean())
        # Try from entry/exit timestamps
        for entry_col, exit_col in [
            ("Entry Timestamp", "Exit Timestamp"),
            ("entry_time", "exit_time"),
        ]:
            if entry_col in self.trades_df.columns and exit_col in self.trades_df.columns:
                entries = pd.to_datetime(self.trades_df[entry_col])
                exits = pd.to_datetime(self.trades_df[exit_col])
                durations = (exits - entries).dt.total_seconds() / 3600  # hours
                valid = durations.dropna()
                if len(valid) > 0:
                    return float(valid.mean())
        return 0.0

    def pnl_percentiles(self) -> dict:
        """Compute PnL distribution percentiles.

        Returns:
            Dict with p10, p25, p50, p75, p90, skewness of trade PnLs.
        """
        pnls = self._trade_pnls()
        if len(pnls) < 2:
            return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "skewness": 0.0}
        return {
            "p10": float(np.percentile(pnls, 10)),
            "p25": float(np.percentile(pnls, 25)),
            "p50": float(np.percentile(pnls, 50)),
            "p75": float(np.percentile(pnls, 75)),
            "p90": float(np.percentile(pnls, 90)),
            "skewness": float(pnls.skew()) if hasattr(pnls, 'skew') else 0.0,
        }

    def trades_per_day(self) -> float:
        """Average number of trades per calendar day.

        Returns:
            Trades per day.
        """
        n_days = len(self.returns) * self.freq_days
        if n_days == 0:
            return 0.0
        return self.total_trades() / n_days

    def best_trade(self) -> float:
        """PnL of the single best trade.

        Returns:
            Maximum trade PnL.
        """
        pnls = self._trade_pnls()
        return float(pnls.max()) if len(pnls) > 0 else 0.0

    def worst_trade(self) -> float:
        """PnL of the single worst trade.

        Returns:
            Minimum trade PnL.
        """
        pnls = self._trade_pnls()
        return float(pnls.min()) if len(pnls) > 0 else 0.0

    def monthly_returns(self) -> pd.DataFrame:
        """Pivot table of monthly returns suitable for a heatmap.

        Returns:
            DataFrame with years as rows, months (1-12) as columns, and
            fractional returns as values.
        """
        if self.returns.empty:
            return pd.DataFrame()

        monthly = self.returns.copy()
        monthly.index = pd.DatetimeIndex(monthly.index)
        monthly = monthly.groupby([monthly.index.year, monthly.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly.index.names = ["Year", "Month"]
        monthly = monthly.reset_index()
        monthly.columns = ["Year", "Month", "Return"]
        pivot = monthly.pivot(index="Year", columns="Month", values="Return")
        month_labels = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pivot.columns = [month_labels[c - 1] for c in pivot.columns]
        return pivot

    # ------------------------------------------------------------------
    # Rolling analytics
    # ------------------------------------------------------------------

    def rolling_sharpe(self, window: int = settings.SHARPE_ROLLING_DAYS) -> pd.Series:
        """Calculate a rolling Sharpe ratio over a sliding window.

        Args:
            window: Number of bars in the rolling window.

        Returns:
            Time-indexed Series of rolling Sharpe values.
        """
        if len(self.returns) < window:
            return pd.Series(dtype=float)
        periods_per_year = _TRADING_DAYS_PER_YEAR / self.freq_days
        rf_per_period = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
        excess = self.returns - rf_per_period
        rolling_mean = excess.rolling(window).mean()
        rolling_std = self.returns.rolling(window).std(ddof=1)
        sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
        return sharpe

    def value_at_risk(
        self,
        confidence: float = 0.95,
        n_simulations: int = settings.VAR_SIMULATIONS,
        horizon_days: int = 1,
        portfolio_value: float = 100_000.0,
    ) -> Dict[str, float]:
        """Calculate Value-at-Risk using parametric Monte Carlo.

        Args:
            confidence: Confidence level (e.g. 0.95 for 95 %).
            n_simulations: Number of Monte Carlo draws.
            horizon_days: VaR time horizon in days.
            portfolio_value: Current portfolio value for dollar VaR.

        Returns:
            Dictionary with ``var_pct``, ``var_dollar``, ``cvar_pct``,
            ``cvar_dollar``.
        """
        if len(self.returns) < 10:
            return {"var_pct": 0.0, "var_dollar": 0.0, "cvar_pct": 0.0, "cvar_dollar": 0.0}

        mean_ret = float(self.returns.mean())
        std_ret = float(self.returns.std())

        simulated = np.random.default_rng(42).normal(
            mean_ret * horizon_days,
            std_ret * np.sqrt(horizon_days),
            n_simulations,
        )

        var_pct = float(np.percentile(simulated, (1 - confidence) * 100))
        cvar_pct = float(simulated[simulated <= var_pct].mean()) if (simulated <= var_pct).any() else var_pct

        return {
            "var_pct": var_pct,
            "var_dollar": var_pct * portfolio_value,
            "cvar_pct": cvar_pct,
            "cvar_dollar": cvar_pct * portfolio_value,
        }

    # ------------------------------------------------------------------
    # Aggregate computation
    # ------------------------------------------------------------------

    def compute_all(self) -> Dict[str, Any]:
        """Compute every metric and return them in a single dictionary.

        The dictionary keys mirror the method names.  This is the
        recommended entry point when you need a full performance report.

        Returns:
            Dictionary mapping metric names to their computed values.
        """
        self._metrics = {
            "total_return": self.total_return(),
            "total_return_pct": self.total_return() * 100,
            "cagr": self.cagr(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_pct": self.max_drawdown() * 100,
            "max_drawdown_duration": self.max_drawdown_duration(),
            "max_drawdown_duration_days": self.max_drawdown_duration(),
            "win_rate": self.win_rate(),
            "win_rate_pct": self.win_rate() * 100,
            "profit_factor": self.profit_factor(),
            "avg_win": self.avg_win(),
            "avg_loss": self.avg_loss(),
            "avg_rr": self.avg_risk_reward(),
            "avg_risk_reward": self.avg_risk_reward(),
            "total_trades": self.total_trades(),
            "trades_per_day": self.trades_per_day(),
            "best_trade": self.best_trade(),
            "worst_trade": self.worst_trade(),
            "monthly_returns": self.monthly_returns(),
            "monthly_heatmap": self.monthly_returns(),
            "equity_curve": self.equity,
            "drawdown_series": self.drawdown,
        }

        # Gross profit / loss for convenience
        pnls = self._trade_pnls()
        self._metrics["gross_profit"] = float(pnls[pnls > 0].sum()) if len(pnls) > 0 else 0.0
        self._metrics["gross_loss"] = float(abs(pnls[pnls < 0].sum())) if len(pnls) > 0 else 0.0
        self._metrics["net_profit"] = self._metrics["gross_profit"] - self._metrics["gross_loss"]

        # New metrics: expectancy, streaks, duration, distribution
        self._metrics["expectancy"] = self.expectancy()
        self._metrics["max_consecutive_wins"] = self.max_consecutive_wins()
        self._metrics["max_consecutive_losses"] = self.max_consecutive_losses()
        self._metrics["avg_trade_duration"] = self.avg_trade_duration()
        self._metrics["pnl_percentiles"] = self.pnl_percentiles()

        logger.info(
            "Metrics computed  |  total_return={tr:.2f}%  sharpe={sh:.2f}  "
            "max_dd={dd:.2f}%  trades={nt}",
            tr=self._metrics["total_return_pct"],
            sh=self._metrics["sharpe_ratio"],
            dd=self._metrics["max_drawdown_pct"],
            nt=self._metrics["total_trades"],
        )

        return self._metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trade_pnls(self) -> pd.Series:
        """Extract PnL series from trades DataFrame.

        Looks for columns named ``PnL``, ``pnl``, or ``realized_pnl``
        (in that order of preference).

        Returns:
            Series of per-trade PnL values.
        """
        if self.trades_df.empty:
            return pd.Series(dtype=float)

        for col in ("PnL", "pnl", "realized_pnl"):
            if col in self.trades_df.columns:
                return self.trades_df[col].dropna()

        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Report generation (module-level convenience functions)
# ---------------------------------------------------------------------------

def generate_report(
    equity: pd.Series,
    trades_df: Optional[pd.DataFrame] = None,
    risk_free_rate: float = settings.RISK_FREE_RATE,
    freq_days: float = 1.0,
) -> Dict[str, Any]:
    """One-shot convenience function to compute all metrics.

    This is equivalent to creating a :class:`BacktestMetrics` instance and
    calling :meth:`~BacktestMetrics.compute_all`, but in a single call.

    Args:
        equity: Time-indexed equity Series.
        trades_df: Optional trades DataFrame.
        risk_free_rate: Annualised risk-free rate.
        freq_days: Bar cadence in calendar days.

    Returns:
        Dictionary of all computed metrics (see
        :meth:`BacktestMetrics.compute_all`).
    """
    bm = BacktestMetrics(
        equity=equity,
        trades_df=trades_df,
        risk_free_rate=risk_free_rate,
        freq_days=freq_days,
    )
    return bm.compute_all()


def format_report(metrics: Dict[str, Any]) -> str:
    """Render a metrics dictionary into a human-readable, multi-line string.

    Suitable for logging or console output.  Numeric values are formatted
    with sensible precision; the monthly returns table is excluded from
    the text representation.

    Args:
        metrics: Dictionary produced by :func:`generate_report` or
                 :meth:`BacktestMetrics.compute_all`.

    Returns:
        Formatted multi-line report string.
    """
    lines = [
        "=" * 60,
        "  BACKTEST PERFORMANCE REPORT",
        "=" * 60,
        "",
        "--- Returns ---",
        f"  Total Return:        {metrics.get('total_return_pct', metrics.get('total_return', 0) * 100):>10.2f} %",
        f"  CAGR:                {metrics.get('cagr', 0) * 100:>10.2f} %",
        "",
        "--- Risk-Adjusted ---",
        f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>10.3f}",
        f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):>10.3f}",
        f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):>10.3f}",
        "",
        "--- Drawdown ---",
        f"  Max Drawdown:        {metrics.get('max_drawdown_pct', metrics.get('max_drawdown', 0) * 100):>10.2f} %",
        f"  Max DD Duration:     {metrics.get('max_drawdown_duration_days', metrics.get('max_drawdown_duration', 0)):>10d} bars",
        "",
        "--- Trade Statistics ---",
        f"  Total Trades:        {metrics.get('total_trades', 0):>10d}",
        f"  Trades / Day:        {metrics.get('trades_per_day', 0):>10.2f}",
        f"  Win Rate:            {metrics.get('win_rate_pct', metrics.get('win_rate', 0) * 100):>10.2f} %",
        f"  Profit Factor:       {metrics.get('profit_factor', 0):>10.2f}",
        f"  Avg Win:             ${metrics.get('avg_win', 0):>10,.2f}",
        f"  Avg Loss:            ${metrics.get('avg_loss', 0):>10,.2f}",
        f"  Avg R/R:             {metrics.get('avg_rr', metrics.get('avg_risk_reward', 0)):>10.2f}",
        f"  Best Trade:          ${metrics.get('best_trade', 0):>10,.2f}",
        f"  Worst Trade:         ${metrics.get('worst_trade', 0):>10,.2f}",
        f"  Net Profit:          ${metrics.get('net_profit', 0):>10,.2f}",
        "",
        "=" * 60,
    ]

    report = "\n".join(lines)
    logger.info("Formatted performance report:\n{}", report)
    return report


# ---------------------------------------------------------------------------
# Plotly visualisations
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity: pd.Series,
    drawdown: pd.Series,
    title: str = "Equity Curve & Drawdown",
) -> go.Figure:
    """Create a two-panel Plotly figure: equity on top, drawdown below.

    Args:
        equity: Time-indexed equity Series.
        drawdown: Time-indexed drawdown Series (negative fractions).
        title: Figure title.

    Returns:
        Plotly ``Figure`` ready for ``.show()`` or serialisation.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown"),
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Equity",
            line=dict(color="#00d4aa", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
        ),
        row=1,
        col=1,
    )

    # Running peak
    running_max = equity.cummax()
    fig.add_trace(
        go.Scatter(
            x=running_max.index,
            y=running_max.values,
            mode="lines",
            name="Peak",
            line=dict(color="grey", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="#ff4444", width=1),
            fillcolor="rgba(255, 68, 68, 0.2)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        height=700,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    logger.debug("Equity curve plot created with {} data points.", len(equity))
    return fig


def plot_monthly_returns(
    monthly_returns: pd.DataFrame,
    title: str = "Monthly Returns Heatmap",
) -> go.Figure:
    """Create a monthly-returns heatmap with annotated cells.

    Args:
        monthly_returns: Pivot table with years as rows and month names as
                         columns.  Values are fractional returns.
                         Typically produced by
                         :meth:`BacktestMetrics.monthly_returns`.
        title: Figure title.

    Returns:
        Plotly ``Figure`` with the heatmap.
    """
    if monthly_returns.empty:
        logger.warning("Empty monthly returns table; returning blank figure.")
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5)
        return fig

    # Build annotation text (percentage strings)
    z_values = monthly_returns.values * 100  # convert to percent
    text_values = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z_values]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=monthly_returns.columns.tolist(),
            y=monthly_returns.index.astype(str).tolist(),
            colorscale=[
                [0.0, "#ff4444"],    # red for losses
                [0.5, "#333333"],    # dark grey for zero
                [1.0, "#00d4aa"],    # green for gains
            ],
            zmid=0,
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>",
            colorbar=dict(title="Return (%)", ticksuffix="%"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(400, 60 * len(monthly_returns) + 100),
        template="plotly_dark",
        yaxis=dict(autorange="reversed"),
    )

    logger.debug(
        "Monthly returns heatmap created  |  years={ny}",
        ny=len(monthly_returns),
    )
    return fig


def plot_trade_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """Plot distribution of individual trade PnL as a bar chart.

    Args:
        trades_df: DataFrame with a PnL column (``PnL``, ``pnl``, or
                   ``realized_pnl``).

    Returns:
        Plotly ``Figure`` with per-trade PnL bars.
    """
    if trades_df is None or trades_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trades to display", x=0.5, y=0.5)
        return fig

    # Resolve PnL column
    pnl_col = None
    for col in ("PnL", "pnl", "realized_pnl"):
        if col in trades_df.columns:
            pnl_col = col
            break
    if pnl_col is None:
        fig = go.Figure()
        fig.add_annotation(text="No PnL column found", x=0.5, y=0.5)
        return fig

    pnl = trades_df[pnl_col].dropna()
    colors = ["#00d4aa" if p > 0 else "#ff4444" for p in pnl]

    fig = go.Figure(
        data=go.Bar(
            x=list(range(len(pnl))),
            y=pnl.values,
            marker_color=colors,
            name="Trade PnL",
        )
    )

    fig.update_layout(
        title="Trade PnL Distribution",
        template="plotly_dark",
        height=400,
        xaxis_title="Trade #",
        yaxis_title="PnL ($)",
    )

    return fig
