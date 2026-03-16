"""
Plotly Dash monitoring dashboard for the Oil Quantitative Trading Bot.

Provides real-time visualization of portfolio equity, open positions, recent
trades, composite sentiment scores, feature importances, returns heatmaps,
active alerts, and system health.  All data is sourced from the SQLAlchemy
database defined in ``db.models``.

Usage::

    from dashboard.app import DashboardApp

    dashboard = DashboardApp()
    app = dashboard.create_app()
    app.run(debug=True)
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html
from loguru import logger
from sqlalchemy import desc, func

from config import settings
from db.models import (
    CompositeSentiment,
    MarketRegime,
    ModelVersion,
    PortfolioState,
    SystemAlert,
    Trade,
    TradeDirection,
    TradeStatus,
    get_session,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS: Dict[str, str] = {
    "background": "#1e1e2f",
    "card": "#27293d",
    "text": "#e0e0e0",
    "accent": "#00d2ff",
    "positive": "#00e676",
    "negative": "#ff1744",
    "warning": "#ffab00",
    "muted": "#6c757d",
    "header_bg": "#1a1a2e",
    "table_header": "#2c2c44",
    "table_row_even": "#27293d",
    "table_row_odd": "#1e1e2f",
}


# ---------------------------------------------------------------------------
# Helper: safe DB query wrapper
# ---------------------------------------------------------------------------

def _safe_query(query_func):
    """Execute *query_func* inside a session, returning the result or ``None``.

    The session is always closed after the call, regardless of success or
    failure.  Errors are logged but not re-raised so that individual dashboard
    components degrade gracefully.

    Parameters
    ----------
    query_func : callable
        A callable that accepts a single ``session`` argument and returns
        the desired query result.

    Returns
    -------
    Any or None
        The return value of *query_func*, or ``None`` if an exception was
        raised during execution.
    """
    session = get_session()
    try:
        return query_func(session)
    except Exception as exc:
        logger.error(f"Dashboard DB query failed: {exc}")
        return None
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Data-fetching helpers
# ---------------------------------------------------------------------------

def _fetch_equity_curve() -> pd.DataFrame:
    """Return a DataFrame of portfolio equity snapshots ordered by time.

    Columns: ``timestamp``, ``equity``, ``drawdown``.

    Returns
    -------
    pd.DataFrame
        Equity-curve data.  Empty DataFrame if no data exists.
    """

    def _query(session):
        rows = (
            session.query(
                PortfolioState.timestamp,
                PortfolioState.equity,
                PortfolioState.drawdown,
            )
            .order_by(PortfolioState.timestamp)
            .all()
        )
        return rows

    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown"])
    return pd.DataFrame(rows, columns=["timestamp", "equity", "drawdown"])


def _fetch_open_positions() -> pd.DataFrame:
    """Return a DataFrame of currently open trades.

    Columns: ``instrument``, ``direction``, ``quantity``, ``entry_price``,
    ``current_price``, ``unrealized_pnl``.

    Returns
    -------
    pd.DataFrame
        Open-position data.  Empty DataFrame if none exist.
    """

    def _query(session):
        rows = (
            session.query(
                Trade.instrument,
                Trade.direction,
                Trade.entry_quantity,
                Trade.entry_price,
                Trade.unrealized_pnl,
            )
            .filter(Trade.status == TradeStatus.OPEN)
            .all()
        )
        return rows

    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(
            columns=[
                "instrument",
                "direction",
                "quantity",
                "entry_price",
                "current_price",
                "unrealized_pnl",
            ]
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "instrument",
            "direction",
            "quantity",
            "entry_price",
            "unrealized_pnl",
        ],
    )
    df["direction"] = df["direction"].apply(
        lambda d: d.value if hasattr(d, "value") else str(d)
    )
    # Derive current_price from entry + unrealized PnL / qty (approximation)
    df["unrealized_pnl"] = df["unrealized_pnl"].fillna(0.0)
    df["quantity"] = df["quantity"].fillna(0.0)
    df["entry_price"] = df["entry_price"].fillna(0.0)
    df["current_price"] = df.apply(
        lambda r: (
            r["entry_price"] + r["unrealized_pnl"] / r["quantity"]
            if r["quantity"] != 0
            else r["entry_price"]
        ),
        axis=1,
    )
    df = df[
        [
            "instrument",
            "direction",
            "quantity",
            "entry_price",
            "current_price",
            "unrealized_pnl",
        ]
    ]
    return df


def _fetch_recent_trades(limit: int = 20) -> pd.DataFrame:
    """Return the most recent *limit* closed trades.

    Columns: ``trade_id``, ``instrument``, ``direction``, ``entry_time``,
    ``exit_time``, ``entry_price``, ``exit_price``, ``realized_pnl``,
    ``status``.

    Parameters
    ----------
    limit : int, optional
        Maximum number of trades to return (default ``20``).

    Returns
    -------
    pd.DataFrame
        Recent-trades data.  Empty DataFrame if none exist.
    """

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
                Trade.status,
            )
            .order_by(desc(Trade.created_at))
            .limit(limit)
            .all()
        )
        return rows

    cols = [
        "trade_id",
        "instrument",
        "direction",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "realized_pnl",
        "status",
    ]
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["direction"] = df["direction"].apply(
        lambda d: d.value if hasattr(d, "value") else str(d)
    )
    df["status"] = df["status"].apply(
        lambda s: s.value if hasattr(s, "value") else str(s)
    )
    df["realized_pnl"] = df["realized_pnl"].fillna(0.0)
    return df


def _fetch_latest_css() -> Tuple[float, float]:
    """Return the latest CSS score and its momentum.

    Returns
    -------
    tuple[float, float]
        ``(css_score, css_momentum)``.  Defaults to ``(0.0, 0.0)`` if no
        data is available.
    """

    def _query(session):
        row = (
            session.query(
                CompositeSentiment.css_score,
                CompositeSentiment.css_momentum,
            )
            .order_by(desc(CompositeSentiment.timestamp))
            .first()
        )
        return row

    row = _safe_query(_query)
    if row is None:
        return 0.0, 0.0
    return float(row[0]), float(row[1] or 0.0)


def _fetch_feature_importances(top_n: int = 15) -> pd.DataFrame:
    """Return the top *top_n* feature importances from the active model.

    Columns: ``feature``, ``importance``.

    Parameters
    ----------
    top_n : int, optional
        Number of features to return (default ``15``).

    Returns
    -------
    pd.DataFrame
        Feature-importance data.  Empty DataFrame if unavailable.
    """

    def _query(session):
        row = (
            session.query(ModelVersion.feature_importances)
            .filter(ModelVersion.is_active.is_(True))
            .order_by(desc(ModelVersion.trained_at))
            .first()
        )
        return row

    row = _safe_query(_query)
    if row is None or row[0] is None:
        return pd.DataFrame(columns=["feature", "importance"])
    importances: Dict[str, float] = row[0]
    df = pd.DataFrame(
        list(importances.items()), columns=["feature", "importance"]
    )
    df = df.sort_values("importance", ascending=False).head(top_n)
    return df


def _fetch_returns_data() -> pd.DataFrame:
    """Return daily returns with ``timestamp`` and ``daily_return`` columns.

    Returns
    -------
    pd.DataFrame
        Portfolio daily returns.  Empty DataFrame if unavailable.
    """

    def _query(session):
        rows = (
            session.query(
                PortfolioState.timestamp,
                PortfolioState.daily_return,
            )
            .filter(PortfolioState.daily_return.isnot(None))
            .order_by(PortfolioState.timestamp)
            .all()
        )
        return rows

    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "daily_return"])
    return pd.DataFrame(rows, columns=["timestamp", "daily_return"])


def _fetch_active_alerts(limit: int = 50) -> pd.DataFrame:
    """Return unacknowledged system alerts.

    Columns: ``timestamp``, ``level``, ``category``, ``message``.

    Parameters
    ----------
    limit : int, optional
        Maximum alerts to return (default ``50``).

    Returns
    -------
    pd.DataFrame
        Alert data.  Empty DataFrame if none exist.
    """

    def _query(session):
        rows = (
            session.query(
                SystemAlert.timestamp,
                SystemAlert.level,
                SystemAlert.category,
                SystemAlert.message,
            )
            .filter(SystemAlert.acknowledged.is_(False))
            .order_by(desc(SystemAlert.timestamp))
            .limit(limit)
            .all()
        )
        return rows

    cols = ["timestamp", "level", "category", "message"]
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def _fetch_system_health() -> Dict[str, Any]:
    """Return system health indicators.

    Returns
    -------
    dict
        Keys: ``ibkr_connected``, ``data_feed_ok``, ``last_retrain``,
        ``active_model_version``, ``equity``, ``open_positions``,
        ``regime``.
    """

    def _query(session):
        # Latest portfolio state
        ps = (
            session.query(PortfolioState)
            .order_by(desc(PortfolioState.timestamp))
            .first()
        )
        # Active model
        mv = (
            session.query(ModelVersion)
            .filter(ModelVersion.is_active.is_(True))
            .order_by(desc(ModelVersion.trained_at))
            .first()
        )
        # Connection alert check (most recent connection-category alert)
        conn_alert = (
            session.query(SystemAlert)
            .filter(SystemAlert.category == "connection")
            .filter(SystemAlert.acknowledged.is_(False))
            .order_by(desc(SystemAlert.timestamp))
            .first()
        )
        # Data feed alert check
        data_alert = (
            session.query(SystemAlert)
            .filter(SystemAlert.category == "data_feed")
            .filter(SystemAlert.acknowledged.is_(False))
            .order_by(desc(SystemAlert.timestamp))
            .first()
        )
        return {
            "portfolio": ps,
            "model": mv,
            "conn_alert": conn_alert,
            "data_alert": data_alert,
        }

    raw = _safe_query(_query)
    if raw is None:
        return {
            "ibkr_connected": False,
            "data_feed_ok": False,
            "last_retrain": "N/A",
            "active_model_version": "N/A",
            "equity": 0.0,
            "open_positions": 0,
            "regime": "N/A",
        }

    ps = raw["portfolio"]
    mv = raw["model"]
    conn_alert = raw["conn_alert"]
    data_alert = raw["data_alert"]

    return {
        "ibkr_connected": conn_alert is None,
        "data_feed_ok": data_alert is None,
        "last_retrain": (
            mv.trained_at.strftime("%Y-%m-%d %H:%M") if mv and mv.trained_at else "N/A"
        ),
        "active_model_version": mv.version if mv else "N/A",
        "equity": ps.equity if ps else 0.0,
        "open_positions": ps.open_positions if ps else 0,
        "regime": (
            ps.regime.value if ps and ps.regime else "N/A"
        ),
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_equity_chart(df: pd.DataFrame) -> go.Figure:
    """Build a Plotly line chart of portfolio equity over time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp`` and ``equity`` columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Styled equity-curve figure.
    """
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="No equity data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=COLORS["muted"], size=16),
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["equity"],
                mode="lines",
                name="Equity",
                line=dict(color=COLORS["accent"], width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 210, 255, 0.08)",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(title=""),
        yaxis=dict(title="Equity ($)"),
        height=320,
    )
    return fig


def _build_css_gauge(css_score: float) -> go.Figure:
    """Build a Plotly gauge indicator for the Composite Sentiment Score.

    Parameters
    ----------
    css_score : float
        Sentiment score in the range [-1, 1].

    Returns
    -------
    plotly.graph_objects.Figure
        Gauge figure with green/yellow/red colour bands.
    """
    bar_color = COLORS["positive"] if css_score >= 0 else COLORS["negative"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=css_score,
            number=dict(font=dict(color=COLORS["text"])),
            gauge=dict(
                axis=dict(range=[-1, 1], tickcolor=COLORS["muted"]),
                bar=dict(color=bar_color),
                bgcolor=COLORS["card"],
                borderwidth=0,
                steps=[
                    dict(range=[-1, -0.3], color="#b71c1c"),
                    dict(range=[-0.3, 0.3], color="#f9a825"),
                    dict(range=[0.3, 1], color="#1b5e20"),
                ],
            ),
            title=dict(text="CSS", font=dict(color=COLORS["text"], size=14)),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=30, r=30, t=40, b=20),
        height=250,
    )
    return fig


def _build_feature_importance_chart(df: pd.DataFrame) -> go.Figure:
    """Build a horizontal bar chart of the top feature importances.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``feature`` and ``importance`` columns, already sorted
        descending by importance and limited to the desired number of rows.

    Returns
    -------
    plotly.graph_objects.Figure
        Horizontal bar chart.
    """
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="No feature importance data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=COLORS["muted"], size=16),
        )
    else:
        df_sorted = df.sort_values("importance", ascending=True)
        fig.add_trace(
            go.Bar(
                x=df_sorted["importance"],
                y=df_sorted["feature"],
                orientation="h",
                marker=dict(color=COLORS["accent"]),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=120, r=20, t=30, b=30),
        xaxis=dict(title="Importance"),
        yaxis=dict(title=""),
        height=400,
    )
    return fig


def _build_returns_heatmap(df: pd.DataFrame) -> go.Figure:
    """Build a weekly returns heatmap grouped by ISO week and day-of-week.

    Each cell represents the daily return for a given weekday and ISO week.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp`` and ``daily_return`` columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap figure.
    """
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="No returns data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=COLORS["muted"], size=16),
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            margin=dict(l=40, r=20, t=30, b=30),
            height=300,
        )
        return fig

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["weekday"] = df["timestamp"].dt.day_name()
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["year"] = df["timestamp"].dt.year
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    pivot = df.pivot_table(
        index="weekday",
        columns="year_week",
        values="daily_return",
        aggfunc="mean",
    )
    # Reindex to consistent weekday order (only include days present)
    present_days = [d for d in day_order if d in pivot.index]
    pivot = pivot.reindex(present_days)

    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, COLORS["negative"]],
                [0.5, "#2c2c44"],
                [1.0, COLORS["positive"]],
            ],
            zmid=0,
            colorbar=dict(
                title="Return",
                tickformat=".2%",
                titlefont=dict(color=COLORS["text"]),
                tickfont=dict(color=COLORS["text"]),
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=80, r=20, t=30, b=50),
        xaxis=dict(title="Year-Week"),
        yaxis=dict(title=""),
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Layout component builders
# ---------------------------------------------------------------------------

def _status_badge(connected: bool, label: str) -> dbc.Badge:
    """Return a colour-coded Bootstrap badge for a status indicator.

    Parameters
    ----------
    connected : bool
        ``True`` renders a green (success) badge, ``False`` renders red
        (danger).
    label : str
        Text displayed inside the badge.

    Returns
    -------
    dbc.Badge
        The styled badge component.
    """
    return dbc.Badge(
        label,
        color="success" if connected else "danger",
        className="me-2 p-2",
    )


def _card(title: str, body_id: str, children=None) -> dbc.Card:
    """Return a styled dashboard card wrapping arbitrary content.

    Parameters
    ----------
    title : str
        Card header text.
    body_id : str
        HTML ``id`` attribute for the card body ``div`` (used as callback
        output target).
    children : dash component or list, optional
        Initial card body contents.

    Returns
    -------
    dbc.Card
        The styled card component.
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                title,
                style={
                    "backgroundColor": COLORS["header_bg"],
                    "color": COLORS["accent"],
                    "fontWeight": "600",
                    "fontSize": "0.95rem",
                },
            ),
            dbc.CardBody(
                children or [],
                id=body_id,
                style={
                    "backgroundColor": COLORS["card"],
                    "padding": "12px",
                },
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": "1px solid #3a3a5c",
            "marginBottom": "12px",
        },
    )


# ---------------------------------------------------------------------------
# Alert-level colour mapping
# ---------------------------------------------------------------------------
_ALERT_LEVEL_COLOUR = {
    "CRITICAL": COLORS["negative"],
    "WARNING": COLORS["warning"],
    "INFO": COLORS["accent"],
}


# ---------------------------------------------------------------------------
# DashboardApp
# ---------------------------------------------------------------------------

class DashboardApp:
    """Factory for the Oil Quant Bot monitoring dashboard.

    Instantiate and call :meth:`create_app` to obtain a fully configured
    ``Dash`` application instance ready to be served.

    Parameters
    ----------
    update_interval_ms : int, optional
        Auto-refresh interval in milliseconds.  Defaults to the value of
        ``settings.DASHBOARD_UPDATE_INTERVAL`` (5 000 ms).

    Attributes
    ----------
    update_interval_ms : int
        The configured refresh interval.
    app : dash.Dash or None
        The Dash application instance (populated after :meth:`create_app`).

    Examples
    --------
    >>> dashboard = DashboardApp()
    >>> app = dashboard.create_app()
    >>> app.run(host="0.0.0.0", port=8050, debug=False)
    """

    def __init__(
        self, update_interval_ms: int | None = None
    ) -> None:
        self.update_interval_ms: int = (
            update_interval_ms
            if update_interval_ms is not None
            else settings.DASHBOARD_UPDATE_INTERVAL
        )
        self.app: Optional[dash.Dash] = None
        logger.info(
            "DashboardApp initialised (interval={}ms)", self.update_interval_ms
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_app(self) -> dash.Dash:
        """Build and return the Dash application.

        The returned app is configured with:

        * A dark-themed responsive layout using ``dash-bootstrap-components``
          (DARKLY theme).
        * Nine dashboard sections (header, equity curve, positions, trades,
          CSS gauge, feature importance, returns heatmap, alerts, system
          health).
        * Independent callbacks for each component, all driven by a single
          ``dcc.Interval`` component with a configurable update period.

        Returns
        -------
        dash.Dash
            The fully configured Dash application.
        """
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="Oil Quant Bot Dashboard",
            update_title=None,
            suppress_callback_exceptions=True,
        )

        self.app.layout = self._build_layout()
        self._register_callbacks()

        logger.info("Dash app created successfully")
        return self.app

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> dbc.Container:
        """Assemble the full dashboard layout.

        Returns
        -------
        dbc.Container
            Root layout container with all sections arranged in a responsive
            Bootstrap grid.
        """
        return dbc.Container(
            fluid=True,
            style={
                "backgroundColor": COLORS["background"],
                "minHeight": "100vh",
                "padding": "16px",
            },
            children=[
                # -- Interval timer --
                dcc.Interval(
                    id="interval-refresh",
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),
                # -- Header --
                self._header_section(),
                # Row 1: Equity curve (wide) + CSS gauge (narrow)
                dbc.Row(
                    [
                        dbc.Col(
                            _card(
                                "Live Equity Curve",
                                "equity-curve-body",
                                children=[dcc.Graph(id="equity-chart")],
                            ),
                            lg=8,
                            md=12,
                        ),
                        dbc.Col(
                            _card(
                                "Composite Sentiment Score",
                                "css-gauge-body",
                                children=[dcc.Graph(id="css-gauge")],
                            ),
                            lg=4,
                            md=12,
                        ),
                    ],
                    className="g-3",
                ),
                # Row 2: Open positions + Recent trades
                dbc.Row(
                    [
                        dbc.Col(
                            _card(
                                "Open Positions",
                                "open-positions-body",
                                children=[html.Div(id="positions-table")],
                            ),
                            lg=5,
                            md=12,
                        ),
                        dbc.Col(
                            _card(
                                "Recent Trades (Last 20)",
                                "recent-trades-body",
                                children=[html.Div(id="trades-table")],
                            ),
                            lg=7,
                            md=12,
                        ),
                    ],
                    className="g-3",
                ),
                # Row 3: Feature importance + Returns heatmap
                dbc.Row(
                    [
                        dbc.Col(
                            _card(
                                "Feature Importance (Top 15)",
                                "feature-importance-body",
                                children=[dcc.Graph(id="feature-chart")],
                            ),
                            lg=6,
                            md=12,
                        ),
                        dbc.Col(
                            _card(
                                "Returns Heatmap",
                                "returns-heatmap-body",
                                children=[dcc.Graph(id="returns-heatmap")],
                            ),
                            lg=6,
                            md=12,
                        ),
                    ],
                    className="g-3",
                ),
                # Row 4: Active alerts + System health
                dbc.Row(
                    [
                        dbc.Col(
                            _card(
                                "Active Alerts",
                                "alerts-panel-body",
                                children=[html.Div(id="alerts-content")],
                            ),
                            lg=7,
                            md=12,
                        ),
                        dbc.Col(
                            _card(
                                "System Health",
                                "health-panel-body",
                                children=[html.Div(id="health-content")],
                            ),
                            lg=5,
                            md=12,
                        ),
                    ],
                    className="g-3",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    @staticmethod
    def _header_section() -> dbc.Row:
        """Build the top header bar with title and status indicator placeholders.

        Returns
        -------
        dbc.Row
            A single Bootstrap row containing the title and live status badges.
        """
        return dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H3(
                            "Oil Quant Bot Dashboard",
                            style={
                                "color": COLORS["accent"],
                                "fontWeight": "700",
                                "marginBottom": "0",
                                "display": "inline-block",
                            },
                        ),
                        html.Span(
                            id="header-status-badges",
                            style={
                                "display": "inline-block",
                                "marginLeft": "24px",
                                "verticalAlign": "middle",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": COLORS["header_bg"],
                        "padding": "14px 20px",
                        "borderRadius": "6px",
                        "marginBottom": "12px",
                        "border": "1px solid #3a3a5c",
                    },
                ),
                width=12,
            ),
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        """Register all Dash callbacks on ``self.app``.

        Each dashboard section has its own callback keyed to the shared
        ``dcc.Interval`` component so that sections update independently.
        """

        # -- Header status badges --
        @self.app.callback(
            Output("header-status-badges", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_header_badges(_n: int):
            """Refresh IBKR and data-feed status badges in the header.

            Parameters
            ----------
            _n : int
                Interval tick counter (unused beyond triggering).

            Returns
            -------
            list
                List of ``dbc.Badge`` components.
            """
            health = _fetch_system_health()
            mode_label = "LIVE" if settings.LIVE_MODE else "PAPER"
            mode_colour = "danger" if settings.LIVE_MODE else "info"
            return [
                dbc.Badge(
                    mode_label,
                    color=mode_colour,
                    className="me-2 p-2",
                ),
                _status_badge(health["ibkr_connected"], "IBKR"),
                _status_badge(health["data_feed_ok"], "Data Feed"),
            ]

        # -- Equity curve --
        @self.app.callback(
            Output("equity-chart", "figure"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_equity_chart(_n: int) -> go.Figure:
            """Refresh the live equity curve chart.

            Parameters
            ----------
            _n : int
                Interval tick counter.

            Returns
            -------
            plotly.graph_objects.Figure
                Updated equity chart.
            """
            df = _fetch_equity_curve()
            return _build_equity_chart(df)

        # -- CSS gauge --
        @self.app.callback(
            Output("css-gauge", "figure"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_css_gauge(_n: int) -> go.Figure:
            """Refresh the Composite Sentiment Score gauge.

            Parameters
            ----------
            _n : int
                Interval tick counter.

            Returns
            -------
            plotly.graph_objects.Figure
                Updated gauge figure.
            """
            css_score, _momentum = _fetch_latest_css()
            return _build_css_gauge(css_score)

        # -- Open positions --
        @self.app.callback(
            Output("positions-table", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_positions_table(_n: int):
            """Refresh the open-positions table.

            Parameters
            ----------
            _n : int
                Interval tick counter.

            Returns
            -------
            dash_table.DataTable or html.P
                The positions table, or a placeholder message if empty.
            """
            df = _fetch_open_positions()
            if df.empty:
                return html.P(
                    "No open positions",
                    style={"color": COLORS["muted"], "textAlign": "center"},
                )
            # Format numeric columns
            for col in ["entry_price", "current_price", "unrealized_pnl"]:
                df[col] = df[col].map(lambda v: f"{v:,.2f}")
            df["quantity"] = df["quantity"].map(
                lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else v
            )
            return dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
                style_header={
                    "backgroundColor": COLORS["table_header"],
                    "color": COLORS["text"],
                    "fontWeight": "600",
                    "border": "none",
                },
                style_cell={
                    "backgroundColor": COLORS["card"],
                    "color": COLORS["text"],
                    "border": "1px solid #3a3a5c",
                    "textAlign": "center",
                    "padding": "8px",
                    "fontSize": "0.85rem",
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": COLORS["table_row_odd"],
                    },
                    {
                        "if": {
                            "filter_query": '{unrealized_pnl} contains "-"',
                            "column_id": "unrealized_pnl",
                        },
                        "color": COLORS["negative"],
                        "fontWeight": "600",
                    },
                    {
                        "if": {
                            "filter_query": '{unrealized_pnl} not contains "-"',
                            "column_id": "unrealized_pnl",
                        },
                        "color": COLORS["positive"],
                        "fontWeight": "600",
                    },
                ],
                page_size=10,
            )

        # -- Recent trades --
        @self.app.callback(
            Output("trades-table", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_trades_table(_n: int):
            """Refresh the recent-trades table.

            Parameters
            ----------
            _n : int
                Interval tick counter.

            Returns
            -------
            dash_table.DataTable or html.P
                The trades table, or a placeholder message if empty.
            """
            df = _fetch_recent_trades(limit=20)
            if df.empty:
                return html.P(
                    "No trades recorded yet",
                    style={"color": COLORS["muted"], "textAlign": "center"},
                )
            # Format
            for col in ["entry_price", "exit_price", "realized_pnl"]:
                df[col] = df[col].map(
                    lambda v: f"{v:,.2f}" if v is not None and pd.notna(v) else "-"
                )
            for col in ["entry_time", "exit_time"]:
                df[col] = df[col].map(
                    lambda v: v.strftime("%m/%d %H:%M") if v is not None else "-"
                )
            return dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
                style_header={
                    "backgroundColor": COLORS["table_header"],
                    "color": COLORS["text"],
                    "fontWeight": "600",
                    "border": "none",
                },
                style_cell={
                    "backgroundColor": COLORS["card"],
                    "color": COLORS["text"],
                    "border": "1px solid #3a3a5c",
                    "textAlign": "center",
                    "padding": "8px",
                    "fontSize": "0.85rem",
                    "minWidth": "70px",
                    "maxWidth": "140px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": COLORS["table_row_odd"],
                    },
                    {
                        "if": {
                            "filter_query": '{realized_pnl} contains "-"',
                            "column_id": "realized_pnl",
                        },
                        "color": COLORS["negative"],
                        "fontWeight": "600",
                    },
                    {
                        "if": {
                            "filter_query": '{realized_pnl} not contains "-"',
                            "column_id": "realized_pnl",
                        },
                        "color": COLORS["positive"],
                        "fontWeight": "600",
                    },
                ],
                page_size=20,
            )

        # -- Feature importance --
        @self.app.callback(
            Output("feature-chart", "figure"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_feature_chart(_n: int) -> go.Figure:
            """Refresh the feature-importance horizontal bar chart.

            Parameters
            ----------
            _n : int
                Interval tick counter.

            Returns
            -------
            plotly.graph_objects.Figure
                Updated bar chart.
            """
            df = _fetch_feature_importances(top_n=15)
            return _build_feature_importance_chart(df)

        # -- Returns heatmap --
        @self.app.callback(
            Output("returns-heatmap", "figure"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_returns_heatmap(_n: int) -> go.Figure:
            """Refresh the daily/weekly returns heatmap.

            Parameters
            ----------
            _n : int
                Interval tick counter.

            Returns
            -------
            plotly.graph_objects.Figure
                Updated heatmap figure.
            """
            df = _fetch_returns_data()
            return _build_returns_heatmap(df)

        # -- Active alerts --
        @self.app.callback(
            Output("alerts-content", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_alerts(_n: int):
            """Refresh the active-alerts panel.

            Displays unacknowledged alerts with colour coding by severity
            level (CRITICAL = red, WARNING = amber, INFO = blue).

            Parameters
            ----------
            _n : int
                Interval tick counter.

            Returns
            -------
            list
                List of ``dbc.Alert`` components, or a single muted
                paragraph when no alerts exist.
            """
            df = _fetch_active_alerts()
            if df.empty:
                return html.P(
                    "No active alerts",
                    style={"color": COLORS["muted"], "textAlign": "center"},
                )
            alerts: List = []
            for _, row in df.iterrows():
                level = str(row["level"]).upper()
                color_map = {
                    "CRITICAL": "danger",
                    "WARNING": "warning",
                    "INFO": "info",
                }
                ts_str = (
                    row["timestamp"].strftime("%m/%d %H:%M:%S")
                    if hasattr(row["timestamp"], "strftime")
                    else str(row["timestamp"])
                )
                alerts.append(
                    dbc.Alert(
                        [
                            html.Strong(f"[{level}] {row['category']}"),
                            html.Span(f"  {ts_str}", style={"marginLeft": "8px", "fontSize": "0.8rem"}),
                            html.P(
                                row["message"],
                                style={"marginBottom": "0", "marginTop": "4px"},
                            ),
                        ],
                        color=color_map.get(level, "secondary"),
                        className="mb-2 py-2 px-3",
                        style={"fontSize": "0.85rem"},
                    )
                )
            return alerts

        # -- System health --
        @self.app.callback(
            Output("health-content", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_system_health(_n: int):
            """Refresh the system-health panel.

            Shows connection status for IBKR and data feed, trading mode,
            latest model retrain timestamp, current equity, open positions,
            and detected market regime.

            Parameters
            ----------
            _n : int
                Interval tick counter.

            Returns
            -------
            dbc.ListGroup
                A list-group with health indicator rows.
            """
            health = _fetch_system_health()

            def _row(label: str, value: str, ok: Optional[bool] = None) -> dbc.ListGroupItem:
                """Build a single health-panel row.

                Parameters
                ----------
                label : str
                    Metric label.
                value : str
                    Display value.
                ok : bool or None
                    If bool, a coloured dot is prepended. ``None`` skips.

                Returns
                -------
                dbc.ListGroupItem
                """
                indicator = ""
                value_colour = COLORS["text"]
                if ok is True:
                    indicator = "  "
                    value_colour = COLORS["positive"]
                elif ok is False:
                    indicator = "  "
                    value_colour = COLORS["negative"]
                return dbc.ListGroupItem(
                    html.Div(
                        [
                            html.Span(
                                label,
                                style={
                                    "fontWeight": "600",
                                    "color": COLORS["text"],
                                    "minWidth": "180px",
                                    "display": "inline-block",
                                },
                            ),
                            html.Span(
                                f"{indicator}{value}",
                                style={"color": value_colour},
                            ),
                        ],
                        style={"display": "flex", "justifyContent": "space-between"},
                    ),
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": "1px solid #3a3a5c",
                        "padding": "10px 14px",
                    },
                )

            equity_str = f"${health['equity']:,.2f}" if isinstance(health["equity"], (int, float)) else str(health["equity"])

            return dbc.ListGroup(
                [
                    _row("IBKR Connection", "Connected" if health["ibkr_connected"] else "Disconnected", ok=health["ibkr_connected"]),
                    _row("Data Feed", "Active" if health["data_feed_ok"] else "Inactive", ok=health["data_feed_ok"]),
                    _row("Trading Mode", "LIVE" if settings.LIVE_MODE else "PAPER", ok=None),
                    _row("Active Model", str(health["active_model_version"])),
                    _row("Last Retrain", str(health["last_retrain"])),
                    _row("Equity", equity_str),
                    _row("Open Positions", str(health["open_positions"])),
                    _row("Market Regime", str(health["regime"])),
                ],
                flush=True,
            )

        logger.info("All dashboard callbacks registered")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dashboard = DashboardApp()
    app = dashboard.create_app()
    app.run(
        host=settings.DASHBOARD_HOST,
        port=settings.DASHBOARD_PORT,
        debug=True,
    )
