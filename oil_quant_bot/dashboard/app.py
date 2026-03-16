"""
Plotly Dash monitoring dashboard for the Oil Quantitative Trading Bot.

Multi-tab layout with:
- Overview: equity curve, positions, trades, sentiment, feature importance, alerts
- Performance Analytics: KPIs, cumulative P&L, distribution, drawdown, streaks
- Trade Log: searchable/filterable trade history with CSV export
- System Logs: real-time log viewer with severity filtering

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
from dashboard.theme import COLORS, card, TABLE_STYLE_HEADER, TABLE_STYLE_CELL
from dashboard.pages import analytics, trade_log, system_logs
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
# Helper: safe DB query wrapper
# ---------------------------------------------------------------------------

def _safe_query(query_func):
    """Execute *query_func* inside a session, returning the result or ``None``."""
    session = get_session()
    try:
        return query_func(session)
    except Exception as exc:
        logger.error(f"Dashboard DB query failed: {exc}")
        return None
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Data-fetching helpers (Overview tab)
# ---------------------------------------------------------------------------

def _fetch_equity_curve() -> pd.DataFrame:
    def _query(session):
        return (
            session.query(
                PortfolioState.timestamp,
                PortfolioState.equity,
                PortfolioState.drawdown,
            )
            .order_by(PortfolioState.timestamp)
            .all()
        )
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown"])
    return pd.DataFrame(rows, columns=["timestamp", "equity", "drawdown"])


def _fetch_open_positions() -> pd.DataFrame:
    def _query(session):
        return (
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
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(
            columns=["instrument", "direction", "quantity", "entry_price",
                      "current_price", "unrealized_pnl"]
        )
    df = pd.DataFrame(
        rows, columns=["instrument", "direction", "quantity", "entry_price", "unrealized_pnl"]
    )
    df["direction"] = df["direction"].apply(lambda d: d.value if hasattr(d, "value") else str(d))
    df["unrealized_pnl"] = df["unrealized_pnl"].fillna(0.0)
    df["quantity"] = df["quantity"].fillna(0.0)
    df["entry_price"] = df["entry_price"].fillna(0.0)
    df["current_price"] = df.apply(
        lambda r: r["entry_price"] + r["unrealized_pnl"] / r["quantity"]
        if r["quantity"] != 0 else r["entry_price"],
        axis=1,
    )
    return df[["instrument", "direction", "quantity", "entry_price", "current_price", "unrealized_pnl"]]


def _fetch_recent_trades(limit: int = 20) -> pd.DataFrame:
    def _query(session):
        return (
            session.query(
                Trade.trade_id, Trade.instrument, Trade.direction,
                Trade.entry_time, Trade.exit_time, Trade.entry_price,
                Trade.exit_price, Trade.realized_pnl, Trade.status,
            )
            .order_by(desc(Trade.created_at))
            .limit(limit)
            .all()
        )
    cols = ["trade_id", "instrument", "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "realized_pnl", "status"]
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["direction"] = df["direction"].apply(lambda d: d.value if hasattr(d, "value") else str(d))
    df["status"] = df["status"].apply(lambda s: s.value if hasattr(s, "value") else str(s))
    df["realized_pnl"] = df["realized_pnl"].fillna(0.0)
    return df


def _fetch_latest_css() -> Tuple[float, float]:
    def _query(session):
        return (
            session.query(CompositeSentiment.css_score, CompositeSentiment.css_momentum)
            .order_by(desc(CompositeSentiment.timestamp))
            .first()
        )
    row = _safe_query(_query)
    if row is None:
        return 0.0, 0.0
    return float(row[0]), float(row[1] or 0.0)


def _fetch_feature_importances(top_n: int = 15) -> pd.DataFrame:
    def _query(session):
        return (
            session.query(ModelVersion.feature_importances)
            .filter(ModelVersion.is_active.is_(True))
            .order_by(desc(ModelVersion.trained_at))
            .first()
        )
    row = _safe_query(_query)
    if row is None or row[0] is None:
        return pd.DataFrame(columns=["feature", "importance"])
    importances: Dict[str, float] = row[0]
    df = pd.DataFrame(list(importances.items()), columns=["feature", "importance"])
    return df.sort_values("importance", ascending=False).head(top_n)


def _fetch_returns_data() -> pd.DataFrame:
    def _query(session):
        return (
            session.query(PortfolioState.timestamp, PortfolioState.daily_return)
            .filter(PortfolioState.daily_return.isnot(None))
            .order_by(PortfolioState.timestamp)
            .all()
        )
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "daily_return"])
    return pd.DataFrame(rows, columns=["timestamp", "daily_return"])


def _fetch_active_alerts(limit: int = 50) -> pd.DataFrame:
    def _query(session):
        return (
            session.query(
                SystemAlert.timestamp, SystemAlert.level,
                SystemAlert.category, SystemAlert.message,
            )
            .filter(SystemAlert.acknowledged.is_(False))
            .order_by(desc(SystemAlert.timestamp))
            .limit(limit)
            .all()
        )
    cols = ["timestamp", "level", "category", "message"]
    rows = _safe_query(_query)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def _fetch_system_health() -> Dict[str, Any]:
    def _query(session):
        ps = session.query(PortfolioState).order_by(desc(PortfolioState.timestamp)).first()
        mv = (
            session.query(ModelVersion)
            .filter(ModelVersion.is_active.is_(True))
            .order_by(desc(ModelVersion.trained_at))
            .first()
        )
        conn_alert = (
            session.query(SystemAlert)
            .filter(SystemAlert.category == "connection", SystemAlert.acknowledged.is_(False))
            .order_by(desc(SystemAlert.timestamp))
            .first()
        )
        data_alert = (
            session.query(SystemAlert)
            .filter(SystemAlert.category == "data_feed", SystemAlert.acknowledged.is_(False))
            .order_by(desc(SystemAlert.timestamp))
            .first()
        )
        return {"portfolio": ps, "model": mv, "conn_alert": conn_alert, "data_alert": data_alert}

    raw = _safe_query(_query)
    if raw is None:
        return {
            "ibkr_connected": False, "data_feed_ok": False,
            "last_retrain": "N/A", "active_model_version": "N/A",
            "equity": 0.0, "open_positions": 0, "regime": "N/A",
        }
    ps, mv = raw["portfolio"], raw["model"]
    return {
        "ibkr_connected": raw["conn_alert"] is None,
        "data_feed_ok": raw["data_alert"] is None,
        "last_retrain": mv.trained_at.strftime("%Y-%m-%d %H:%M") if mv and mv.trained_at else "N/A",
        "active_model_version": mv.version if mv else "N/A",
        "equity": ps.equity if ps else 0.0,
        "open_positions": ps.open_positions if ps else 0,
        "regime": ps.regime.value if ps and ps.regime else "N/A",
    }


# ---------------------------------------------------------------------------
# Chart builders (Overview tab)
# ---------------------------------------------------------------------------

def _build_equity_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text="No equity data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=COLORS["muted"], size=16))
    else:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["equity"], mode="lines", name="Equity",
            line=dict(color=COLORS["accent"], width=2),
            fill="tozeroy", fillcolor="rgba(0, 210, 255, 0.08)",
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        margin=dict(l=40, r=20, t=30, b=30), xaxis=dict(title=""),
        yaxis=dict(title="Equity ($)"), height=320,
    )
    return fig


def _build_css_gauge(css_score: float) -> go.Figure:
    bar_color = COLORS["positive"] if css_score >= 0 else COLORS["negative"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=css_score,
        number=dict(font=dict(color=COLORS["text"])),
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor=COLORS["muted"]),
            bar=dict(color=bar_color), bgcolor=COLORS["card"], borderwidth=0,
            steps=[
                dict(range=[-1, -0.3], color="#b71c1c"),
                dict(range=[-0.3, 0.3], color="#f9a825"),
                dict(range=[0.3, 1], color="#1b5e20"),
            ],
        ),
        title=dict(text="CSS", font=dict(color=COLORS["text"], size=14)),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        margin=dict(l=30, r=30, t=40, b=20), height=250,
    )
    return fig


def _build_feature_importance_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text="No feature importance data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=COLORS["muted"], size=16))
    else:
        df_sorted = df.sort_values("importance", ascending=True)
        fig.add_trace(go.Bar(
            x=df_sorted["importance"], y=df_sorted["feature"],
            orientation="h", marker=dict(color=COLORS["accent"]),
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        margin=dict(l=120, r=20, t=30, b=30), xaxis=dict(title="Importance"),
        yaxis=dict(title=""), height=400,
    )
    return fig


def _build_returns_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text="No returns data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=COLORS["muted"], size=16))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
            margin=dict(l=40, r=20, t=30, b=30), height=300,
        )
        return fig

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["weekday"] = df["timestamp"].dt.day_name()
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["year"] = df["timestamp"].dt.year
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    pivot = df.pivot_table(index="weekday", columns="year_week", values="daily_return", aggfunc="mean")
    present_days = [d for d in day_order if d in pivot.index]
    pivot = pivot.reindex(present_days)

    fig.add_trace(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0.0, COLORS["negative"]], [0.5, "#2c2c44"], [1.0, COLORS["positive"]]],
        zmid=0,
        colorbar=dict(title="Return", tickformat=".2%",
                       titlefont=dict(color=COLORS["text"]), tickfont=dict(color=COLORS["text"])),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        margin=dict(l=80, r=20, t=30, b=50), xaxis=dict(title="Year-Week"),
        yaxis=dict(title=""), height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _status_badge(connected: bool, label: str) -> dbc.Badge:
    return dbc.Badge(label, color="success" if connected else "danger", className="me-2 p-2")


# ---------------------------------------------------------------------------
# DashboardApp
# ---------------------------------------------------------------------------

class DashboardApp:
    """Factory for the Oil Quant Bot monitoring dashboard.

    Creates a multi-tab Dash application with Overview, Performance Analytics,
    Trade Log, and System Logs pages.
    """

    def __init__(self, update_interval_ms: int | None = None) -> None:
        self.update_interval_ms: int = (
            update_interval_ms if update_interval_ms is not None
            else settings.DASHBOARD_UPDATE_INTERVAL
        )
        self.app: Optional[dash.Dash] = None
        logger.info("DashboardApp initialised (interval={}ms)", self.update_interval_ms)

    def create_app(self) -> dash.Dash:
        """Build and return the fully configured multi-tab Dash application."""
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="Oil Quant Bot Dashboard",
            update_title=None,
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1.0, maximum-scale=1.0"},
                {"name": "apple-mobile-web-app-capable", "content": "yes"},
                {"name": "apple-mobile-web-app-status-bar-style", "content": "black-translucent"},
                {"name": "theme-color", "content": "#1e1e2f"},
            ],
        )

        # PWA manifest for "Add to Home Screen" on mobile
        self.app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="manifest" href="/assets/manifest.json">
    <style>
        /* Mobile responsive overrides */
        @media (max-width: 768px) {
            .dash-table-container { overflow-x: auto !important; }
            .card-body { padding: 8px !important; }
            h3 { font-size: 1.1rem !important; }
            h4 { font-size: 1rem !important; }
            .tab-content { padding: 4px !important; }
        }
        /* Smooth scrolling */
        html { scroll-behavior: smooth; }
        /* Better touch targets */
        .nav-link { min-height: 44px; display: flex; align-items: center; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''

        self.app.layout = self._build_layout()
        self._register_overview_callbacks()
        analytics.register_callbacks(self.app)
        trade_log.register_callbacks(self.app)
        system_logs.register_callbacks(self.app)

        logger.info("Dash app created with 4 tabs")
        return self.app

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> dbc.Container:
        return dbc.Container(
            fluid=True,
            style={
                "backgroundColor": COLORS["background"],
                "minHeight": "100vh",
                "padding": "16px",
            },
            children=[
                # Interval timer (shared by all tabs)
                dcc.Interval(id="interval-refresh", interval=self.update_interval_ms, n_intervals=0),

                # Header
                self._header_section(),

                # Tabs
                dbc.Tabs(
                    id="main-tabs",
                    active_tab="overview",
                    style={"marginBottom": "12px"},
                    children=[
                        dbc.Tab(
                            label="Overview",
                            tab_id="overview",
                            tab_style={"backgroundColor": COLORS["header_bg"]},
                            active_label_style={"backgroundColor": COLORS["accent"], "color": "#000"},
                            label_style={"color": COLORS["muted"]},
                            children=self._overview_tab(),
                        ),
                        dbc.Tab(
                            label="Performance Analytics",
                            tab_id="analytics",
                            tab_style={"backgroundColor": COLORS["header_bg"]},
                            active_label_style={"backgroundColor": COLORS["accent"], "color": "#000"},
                            label_style={"color": COLORS["muted"]},
                            children=html.Div(analytics.layout(), style={"paddingTop": "12px"}),
                        ),
                        dbc.Tab(
                            label="Trade Log",
                            tab_id="trade-log",
                            tab_style={"backgroundColor": COLORS["header_bg"]},
                            active_label_style={"backgroundColor": COLORS["accent"], "color": "#000"},
                            label_style={"color": COLORS["muted"]},
                            children=html.Div(trade_log.layout(), style={"paddingTop": "12px"}),
                        ),
                        dbc.Tab(
                            label="System Logs",
                            tab_id="system-logs",
                            tab_style={"backgroundColor": COLORS["header_bg"]},
                            active_label_style={"backgroundColor": COLORS["accent"], "color": "#000"},
                            label_style={"color": COLORS["muted"]},
                            children=html.Div(system_logs.layout(), style={"paddingTop": "12px"}),
                        ),
                    ],
                ),
            ],
        )

    def _overview_tab(self) -> html.Div:
        """The original overview dashboard content."""
        return html.Div([
            # Row 1: Equity curve + CSS gauge
            dbc.Row([
                dbc.Col(card("Live Equity Curve", "equity-curve-body",
                             [dcc.Graph(id="equity-chart")]), lg=8, md=12),
                dbc.Col(card("Composite Sentiment Score", "css-gauge-body",
                             [dcc.Graph(id="css-gauge")]), lg=4, md=12),
            ], className="g-3"),
            # Row 2: Open positions + Recent trades
            dbc.Row([
                dbc.Col(card("Open Positions", "open-positions-body",
                             [html.Div(id="positions-table")]), lg=5, md=12),
                dbc.Col(card("Recent Trades (Last 20)", "recent-trades-body",
                             [html.Div(id="trades-table")]), lg=7, md=12),
            ], className="g-3"),
            # Row 3: Feature importance + Returns heatmap
            dbc.Row([
                dbc.Col(card("Feature Importance (Top 15)", "feature-importance-body",
                             [dcc.Graph(id="feature-chart")]), lg=6, md=12),
                dbc.Col(card("Returns Heatmap", "returns-heatmap-body",
                             [dcc.Graph(id="returns-heatmap")]), lg=6, md=12),
            ], className="g-3"),
            # Row 4: Active alerts + System health
            dbc.Row([
                dbc.Col(card("Active Alerts", "alerts-panel-body",
                             [html.Div(id="alerts-content")]), lg=7, md=12),
                dbc.Col(card("System Health", "health-panel-body",
                             [html.Div(id="health-content")]), lg=5, md=12),
            ], className="g-3"),
        ], style={"paddingTop": "12px"})

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    @staticmethod
    def _header_section() -> dbc.Row:
        return dbc.Row(
            dbc.Col(
                html.Div([
                    html.H3("Oil Quant Bot Dashboard", style={
                        "color": COLORS["accent"], "fontWeight": "700",
                        "marginBottom": "0", "display": "inline-block",
                    }),
                    html.Span(id="header-status-badges", style={
                        "display": "inline-block", "marginLeft": "24px",
                        "verticalAlign": "middle",
                    }),
                ], style={
                    "backgroundColor": COLORS["header_bg"],
                    "padding": "14px 20px", "borderRadius": "6px",
                    "marginBottom": "12px", "border": "1px solid #3a3a5c",
                }),
                width=12,
            ),
        )

    # ------------------------------------------------------------------
    # Overview callbacks
    # ------------------------------------------------------------------

    def _register_overview_callbacks(self) -> None:
        # Header status badges
        @self.app.callback(
            Output("header-status-badges", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_header_badges(_n):
            health = _fetch_system_health()
            mode_label = "LIVE" if settings.LIVE_MODE else "PAPER"
            mode_colour = "danger" if settings.LIVE_MODE else "info"
            return [
                dbc.Badge(mode_label, color=mode_colour, className="me-2 p-2"),
                _status_badge(health["ibkr_connected"], "IBKR"),
                _status_badge(health["data_feed_ok"], "Data Feed"),
            ]

        # Equity curve
        @self.app.callback(
            Output("equity-chart", "figure"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_equity_chart(_n):
            return _build_equity_chart(_fetch_equity_curve())

        # CSS gauge
        @self.app.callback(
            Output("css-gauge", "figure"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_css_gauge(_n):
            css_score, _ = _fetch_latest_css()
            return _build_css_gauge(css_score)

        # Open positions
        @self.app.callback(
            Output("positions-table", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_positions_table(_n):
            df = _fetch_open_positions()
            if df.empty:
                return html.P("No open positions", style={"color": COLORS["muted"], "textAlign": "center"})
            for col in ["entry_price", "current_price", "unrealized_pnl"]:
                df[col] = df[col].map(lambda v: f"{v:,.2f}")
            df["quantity"] = df["quantity"].map(
                lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else v
            )
            return dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
                style_header=TABLE_STYLE_HEADER,
                style_cell=TABLE_STYLE_CELL,
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": COLORS["table_row_odd"]},
                    {"if": {"filter_query": '{unrealized_pnl} contains "-"', "column_id": "unrealized_pnl"},
                     "color": COLORS["negative"], "fontWeight": "600"},
                    {"if": {"filter_query": '{unrealized_pnl} not contains "-"', "column_id": "unrealized_pnl"},
                     "color": COLORS["positive"], "fontWeight": "600"},
                ],
                page_size=10,
            )

        # Recent trades
        @self.app.callback(
            Output("trades-table", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_trades_table(_n):
            df = _fetch_recent_trades(limit=20)
            if df.empty:
                return html.P("No trades recorded yet", style={"color": COLORS["muted"], "textAlign": "center"})
            for col in ["entry_price", "exit_price", "realized_pnl"]:
                df[col] = df[col].map(lambda v: f"{v:,.2f}" if v is not None and pd.notna(v) else "-")
            for col in ["entry_time", "exit_time"]:
                df[col] = df[col].map(lambda v: v.strftime("%m/%d %H:%M") if v is not None else "-")
            return dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
                style_header=TABLE_STYLE_HEADER,
                style_cell={**TABLE_STYLE_CELL, "minWidth": "70px", "maxWidth": "140px",
                            "overflow": "hidden", "textOverflow": "ellipsis"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": COLORS["table_row_odd"]},
                    {"if": {"filter_query": '{realized_pnl} contains "-"', "column_id": "realized_pnl"},
                     "color": COLORS["negative"], "fontWeight": "600"},
                    {"if": {"filter_query": '{realized_pnl} not contains "-"', "column_id": "realized_pnl"},
                     "color": COLORS["positive"], "fontWeight": "600"},
                ],
                page_size=20,
            )

        # Feature importance
        @self.app.callback(
            Output("feature-chart", "figure"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_feature_chart(_n):
            return _build_feature_importance_chart(_fetch_feature_importances(top_n=15))

        # Returns heatmap
        @self.app.callback(
            Output("returns-heatmap", "figure"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_returns_heatmap(_n):
            return _build_returns_heatmap(_fetch_returns_data())

        # Active alerts
        @self.app.callback(
            Output("alerts-content", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_alerts(_n):
            df = _fetch_active_alerts()
            if df.empty:
                return html.P("No active alerts", style={"color": COLORS["muted"], "textAlign": "center"})
            alerts: List = []
            for _, row in df.iterrows():
                level = str(row["level"]).upper()
                color_map = {"CRITICAL": "danger", "WARNING": "warning", "INFO": "info"}
                ts_str = (
                    row["timestamp"].strftime("%m/%d %H:%M:%S")
                    if hasattr(row["timestamp"], "strftime") else str(row["timestamp"])
                )
                alerts.append(dbc.Alert([
                    html.Strong(f"[{level}] {row['category']}"),
                    html.Span(f"  {ts_str}", style={"marginLeft": "8px", "fontSize": "0.8rem"}),
                    html.P(row["message"], style={"marginBottom": "0", "marginTop": "4px"}),
                ], color=color_map.get(level, "secondary"),
                   className="mb-2 py-2 px-3", style={"fontSize": "0.85rem"}))
            return alerts

        # System health
        @self.app.callback(
            Output("health-content", "children"),
            Input("interval-refresh", "n_intervals"),
        )
        def update_system_health(_n):
            health = _fetch_system_health()

            def _row(label, value, ok=None):
                indicator = ""
                value_colour = COLORS["text"]
                if ok is True:
                    indicator = "  "
                    value_colour = COLORS["positive"]
                elif ok is False:
                    indicator = "  "
                    value_colour = COLORS["negative"]
                return dbc.ListGroupItem(
                    html.Div([
                        html.Span(label, style={"fontWeight": "600", "color": COLORS["text"],
                                                 "minWidth": "180px", "display": "inline-block"}),
                        html.Span(f"{indicator}{value}", style={"color": value_colour}),
                    ], style={"display": "flex", "justifyContent": "space-between"}),
                    style={"backgroundColor": COLORS["card"], "border": "1px solid #3a3a5c", "padding": "10px 14px"},
                )

            equity_str = f"${health['equity']:,.2f}" if isinstance(health["equity"], (int, float)) else str(health["equity"])
            return dbc.ListGroup([
                _row("IBKR Connection", "Connected" if health["ibkr_connected"] else "Disconnected", ok=health["ibkr_connected"]),
                _row("Data Feed", "Active" if health["data_feed_ok"] else "Inactive", ok=health["data_feed_ok"]),
                _row("Trading Mode", "LIVE" if settings.LIVE_MODE else "PAPER"),
                _row("Active Model", str(health["active_model_version"])),
                _row("Last Retrain", str(health["last_retrain"])),
                _row("Equity", equity_str),
                _row("Open Positions", str(health["open_positions"])),
                _row("Market Regime", str(health["regime"])),
            ], flush=True)

        logger.info("Overview callbacks registered")


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
