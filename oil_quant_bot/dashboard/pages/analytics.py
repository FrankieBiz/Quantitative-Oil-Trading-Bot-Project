"""
Performance Analytics page for the Oil Quant Bot Dashboard.

Provides win/loss KPIs, cumulative P&L chart, P&L distribution histogram,
drawdown chart, Sharpe ratio trend, and win/loss streak analysis.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots
from sqlalchemy import desc, func

from config import settings
from dashboard.theme import COLORS, card, TABLE_STYLE_HEADER, TABLE_STYLE_CELL
from db.models import (
    PortfolioState,
    Trade,
    TradeDirection,
    TradeStatus,
    get_session,
)

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _safe_query(query_func):
    session = get_session()
    try:
        return query_func(session)
    except Exception:
        return None
    finally:
        session.close()


def _fetch_all_closed_trades() -> pd.DataFrame:
    """Fetch all closed trades with key columns for analytics."""

    def _query(session):
        rows = (
            session.query(
                Trade.trade_id,
                Trade.instrument,
                Trade.direction,
                Trade.entry_time,
                Trade.exit_time,
                Trade.entry_price,
                Trade.exit_price,
                Trade.realized_pnl,
                Trade.commission,
                Trade.slippage,
                Trade.regime,
                Trade.confidence,
            )
            .filter(Trade.status == TradeStatus.CLOSED)
            .order_by(Trade.exit_time)
            .all()
        )
        return rows

    cols = [
        "trade_id", "instrument", "direction", "entry_time", "exit_time",
        "entry_price", "exit_price", "realized_pnl", "commission",
        "slippage", "regime", "confidence",
    ]
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["direction"] = df["direction"].apply(
        lambda d: d.value if hasattr(d, "value") else str(d)
    )
    df["regime"] = df["regime"].apply(
        lambda r: r.value if hasattr(r, "value") else str(r) if r else "N/A"
    )
    df["realized_pnl"] = df["realized_pnl"].fillna(0.0)
    df["commission"] = df["commission"].fillna(0.0)
    df["slippage"] = df["slippage"].fillna(0.0)
    df["net_pnl"] = df["realized_pnl"] - df["commission"] - df["slippage"]
    return df


def _fetch_portfolio_history() -> pd.DataFrame:
    def _query(session):
        rows = (
            session.query(
                PortfolioState.timestamp,
                PortfolioState.equity,
                PortfolioState.drawdown,
                PortfolioState.sharpe_30d,
                PortfolioState.daily_return,
            )
            .order_by(PortfolioState.timestamp)
            .all()
        )
        return rows

    cols = ["timestamp", "equity", "drawdown", "sharpe_30d", "daily_return"]
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------


def _compute_kpis(trades_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute performance KPIs from trade and portfolio data."""
    kpis: Dict[str, Any] = {
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "avg_trade_duration": "N/A",
        "current_streak": 0,
        "longest_win_streak": 0,
        "longest_loss_streak": 0,
    }

    if trades_df.empty:
        return kpis

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]

    kpis["total_trades"] = len(trades_df)
    kpis["win_rate"] = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0.0
    kpis["total_pnl"] = trades_df["net_pnl"].sum()
    kpis["avg_win"] = wins["net_pnl"].mean() if not wins.empty else 0.0
    kpis["avg_loss"] = losses["net_pnl"].mean() if not losses.empty else 0.0
    kpis["best_trade"] = trades_df["net_pnl"].max()
    kpis["worst_trade"] = trades_df["net_pnl"].min()

    gross_profit = wins["net_pnl"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["net_pnl"].sum()) if not losses.empty else 0.0
    kpis["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Streaks
    is_win = (trades_df["net_pnl"] > 0).values
    streaks = _compute_streaks(is_win)
    kpis["current_streak"] = streaks["current"]
    kpis["longest_win_streak"] = streaks["longest_win"]
    kpis["longest_loss_streak"] = streaks["longest_loss"]

    # Duration
    if trades_df["entry_time"].notna().any() and trades_df["exit_time"].notna().any():
        durations = (
            pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])
        ).dropna()
        if not durations.empty:
            avg_dur = durations.mean()
            hours = int(avg_dur.total_seconds() // 3600)
            minutes = int((avg_dur.total_seconds() % 3600) // 60)
            kpis["avg_trade_duration"] = f"{hours}h {minutes}m"

    # Portfolio-level metrics
    if not portfolio_df.empty:
        kpis["max_drawdown"] = portfolio_df["drawdown"].max() * 100 if portfolio_df["drawdown"].notna().any() else 0.0
        if portfolio_df["sharpe_30d"].notna().any():
            kpis["sharpe_ratio"] = portfolio_df["sharpe_30d"].iloc[-1] or 0.0

    return kpis


def _compute_streaks(is_win: np.ndarray) -> Dict[str, int]:
    """Compute win/loss streak stats."""
    if len(is_win) == 0:
        return {"current": 0, "longest_win": 0, "longest_loss": 0}

    longest_win = 0
    longest_loss = 0
    current = 0
    streak = 0

    for i, w in enumerate(is_win):
        if i == 0:
            streak = 1
        elif w == is_win[i - 1]:
            streak += 1
        else:
            streak = 1

        if w:
            longest_win = max(longest_win, streak)
        else:
            longest_loss = max(longest_loss, streak)

    # Current streak (positive = wins, negative = losses)
    if len(is_win) > 0:
        last = is_win[-1]
        current = streak if last else -streak

    return {"current": current, "longest_win": longest_win, "longest_loss": longest_loss}


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_cumulative_pnl_chart(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.add_annotation(text="No trade data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=COLORS["muted"], size=16))
    else:
        cumulative = trades_df["net_pnl"].cumsum()
        fig.add_trace(go.Scatter(
            x=trades_df["exit_time"],
            y=cumulative,
            mode="lines",
            name="Cumulative P&L",
            line=dict(color=COLORS["accent"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 210, 255, 0.08)",
        ))
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.5)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=50, r=20, t=10, b=30),
        yaxis=dict(title="Cumulative P&L ($)"),
        height=350,
    )
    return fig


def _build_pnl_distribution(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.add_annotation(text="No trade data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=COLORS["muted"], size=16))
    else:
        colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in trades_df["net_pnl"]]
        fig.add_trace(go.Histogram(
            x=trades_df["net_pnl"],
            nbinsx=30,
            marker_color=COLORS["accent"],
            opacity=0.8,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=COLORS["warning"])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=50, r=20, t=10, b=30),
        xaxis=dict(title="Trade P&L ($)"),
        yaxis=dict(title="Count"),
        height=350,
    )
    return fig


def _build_drawdown_chart(portfolio_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if portfolio_df.empty or not portfolio_df["drawdown"].notna().any():
        fig.add_annotation(text="No drawdown data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=COLORS["muted"], size=16))
    else:
        fig.add_trace(go.Scatter(
            x=portfolio_df["timestamp"],
            y=portfolio_df["drawdown"] * -100,
            mode="lines",
            fill="tozeroy",
            line=dict(color=COLORS["negative"], width=1.5),
            fillcolor="rgba(255, 23, 68, 0.15)",
            name="Drawdown",
        ))
        # Max drawdown line
        max_dd = portfolio_df["drawdown"].max() * -100
        fig.add_hline(y=max_dd, line_dash="dot", line_color=COLORS["warning"],
                       annotation_text=f"Max: {max_dd:.1f}%",
                       annotation_font_color=COLORS["warning"])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=50, r=20, t=10, b=30),
        yaxis=dict(title="Drawdown (%)"),
        height=300,
    )
    return fig


def _build_win_loss_by_regime(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=COLORS["muted"], size=16))
    else:
        regime_stats = trades_df.groupby("regime").agg(
            wins=("net_pnl", lambda x: (x > 0).sum()),
            losses=("net_pnl", lambda x: (x <= 0).sum()),
            avg_pnl=("net_pnl", "mean"),
        ).reset_index()

        fig.add_trace(go.Bar(
            x=regime_stats["regime"],
            y=regime_stats["wins"],
            name="Wins",
            marker_color=COLORS["positive"],
        ))
        fig.add_trace(go.Bar(
            x=regime_stats["regime"],
            y=regime_stats["losses"],
            name="Losses",
            marker_color=COLORS["negative"],
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        barmode="group",
        margin=dict(l=40, r=20, t=10, b=30),
        yaxis=dict(title="Trade Count"),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# KPI card builder
# ---------------------------------------------------------------------------


def _kpi_card(label: str, value: str, color: str = COLORS["text"],
              sublabel: str = "") -> dbc.Card:
    children = [
        html.P(label, style={
            "color": COLORS["muted"], "fontSize": "0.75rem",
            "marginBottom": "2px", "textTransform": "uppercase",
        }),
        html.H4(value, style={
            "color": color, "fontWeight": "700", "marginBottom": "0",
        }),
    ]
    if sublabel:
        children.append(html.Small(sublabel, style={"color": COLORS["muted"]}))

    return dbc.Card(
        dbc.CardBody(children, style={
            "padding": "12px 16px",
            "backgroundColor": COLORS["card"],
        }),
        style={
            "backgroundColor": COLORS["card"],
            "border": "1px solid #3a3a5c",
            "textAlign": "center",
        },
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def layout() -> html.Div:
    return html.Div([
        # KPI row
        html.Div(id="analytics-kpis"),
        html.Br(),
        # Charts row 1: Cumulative P&L + P&L Distribution
        dbc.Row([
            dbc.Col(card("Cumulative P&L", "cum-pnl-body",
                         [dcc.Graph(id="cum-pnl-chart")]), lg=7, md=12),
            dbc.Col(card("P&L Distribution", "pnl-dist-body",
                         [dcc.Graph(id="pnl-dist-chart")]), lg=5, md=12),
        ], className="g-3"),
        # Charts row 2: Drawdown + Win/Loss by Regime
        dbc.Row([
            dbc.Col(card("Drawdown", "drawdown-body",
                         [dcc.Graph(id="drawdown-chart")]), lg=7, md=12),
            dbc.Col(card("Win/Loss by Regime", "regime-wl-body",
                         [dcc.Graph(id="regime-wl-chart")]), lg=5, md=12),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Register callbacks
# ---------------------------------------------------------------------------


def register_callbacks(app):
    @app.callback(
        [
            Output("analytics-kpis", "children"),
            Output("cum-pnl-chart", "figure"),
            Output("pnl-dist-chart", "figure"),
            Output("drawdown-chart", "figure"),
            Output("regime-wl-chart", "figure"),
        ],
        Input("interval-refresh", "n_intervals"),
    )
    def update_analytics(_n):
        trades_df = _fetch_all_closed_trades()
        portfolio_df = _fetch_portfolio_history()
        kpis = _compute_kpis(trades_df, portfolio_df)

        # Build KPI cards row
        pnl_color = COLORS["positive"] if kpis["total_pnl"] >= 0 else COLORS["negative"]
        streak_val = kpis["current_streak"]
        streak_color = COLORS["positive"] if streak_val > 0 else COLORS["negative"] if streak_val < 0 else COLORS["muted"]
        streak_label = f"{abs(streak_val)} {'W' if streak_val > 0 else 'L'}" if streak_val != 0 else "0"

        pf_str = f"{kpis['profit_factor']:.2f}" if kpis["profit_factor"] != float("inf") else "INF"

        kpi_row = dbc.Row([
            dbc.Col(_kpi_card("Total Trades", str(kpis["total_trades"])), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Win Rate", f"{kpis['win_rate']:.1f}%",
                              COLORS["positive"] if kpis["win_rate"] >= 50 else COLORS["negative"]),
                    lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Total P&L", f"${kpis['total_pnl']:,.0f}", pnl_color), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Profit Factor", pf_str), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Max Drawdown", f"{kpis['max_drawdown']:.1f}%", COLORS["negative"]), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Sharpe (30d)", f"{kpis['sharpe_ratio']:.2f}"), lg=2, md=4, xs=6),
        ], className="g-2 mb-2")

        kpi_row2 = dbc.Row([
            dbc.Col(_kpi_card("Avg Win", f"${kpis['avg_win']:,.0f}", COLORS["positive"]), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Avg Loss", f"${kpis['avg_loss']:,.0f}", COLORS["negative"]), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Best Trade", f"${kpis['best_trade']:,.0f}", COLORS["positive"]), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Worst Trade", f"${kpis['worst_trade']:,.0f}", COLORS["negative"]), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Avg Duration", kpis["avg_trade_duration"]), lg=2, md=4, xs=6),
            dbc.Col(_kpi_card("Streak", streak_label, streak_color,
                              f"Best W:{kpis['longest_win_streak']} L:{kpis['longest_loss_streak']}"),
                    lg=2, md=4, xs=6),
        ], className="g-2")

        kpi_section = html.Div([kpi_row, kpi_row2])

        return (
            kpi_section,
            _build_cumulative_pnl_chart(trades_df),
            _build_pnl_distribution(trades_df),
            _build_drawdown_chart(portfolio_df),
            _build_win_loss_by_regime(trades_df),
        )
