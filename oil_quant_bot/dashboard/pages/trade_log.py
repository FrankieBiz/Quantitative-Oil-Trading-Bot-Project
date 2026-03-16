"""
Trade Log page for the Oil Quant Bot Dashboard.

Full searchable/filterable trade history with date range picker,
direction and status filters, and CSV export.
"""

from __future__ import annotations

import io
import datetime as dt
from typing import Any, Dict, List

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html, dash_table, no_update
from sqlalchemy import desc

from config import settings
from dashboard.theme import COLORS, card, TABLE_STYLE_HEADER, TABLE_STYLE_CELL
from db.models import (
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


def _fetch_trades(
    direction: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch trades with optional filters."""

    def _query(session):
        q = session.query(
            Trade.trade_id,
            Trade.instrument,
            Trade.direction,
            Trade.status,
            Trade.entry_time,
            Trade.exit_time,
            Trade.entry_price,
            Trade.exit_price,
            Trade.entry_quantity,
            Trade.realized_pnl,
            Trade.commission,
            Trade.slippage,
            Trade.confidence,
            Trade.regime,
            Trade.stop_loss_price,
            Trade.take_profit_price,
        )
        if direction and direction != "ALL":
            q = q.filter(Trade.direction == TradeDirection(direction))
        if status and status != "ALL":
            q = q.filter(Trade.status == TradeStatus(status))
        if start_date:
            q = q.filter(Trade.entry_time >= start_date)
        if end_date:
            end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            q = q.filter(Trade.entry_time < end)
        return q.order_by(desc(Trade.created_at)).limit(500).all()

    cols = [
        "trade_id", "instrument", "direction", "status", "entry_time",
        "exit_time", "entry_price", "exit_price", "quantity", "realized_pnl",
        "commission", "slippage", "confidence", "regime", "stop_loss",
        "take_profit",
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
    df["regime"] = df["regime"].apply(
        lambda r: r.value if hasattr(r, "value") else str(r) if r else ""
    )
    df["realized_pnl"] = df["realized_pnl"].fillna(0.0)
    df["commission"] = df["commission"].fillna(0.0)
    df["slippage"] = df["slippage"].fillna(0.0)
    df["net_pnl"] = df["realized_pnl"] - df["commission"] - df["slippage"]
    return df


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def layout() -> html.Div:
    return html.Div([
        # Filter bar
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Date Range", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dcc.DatePickerRange(
                            id="trade-log-date-range",
                            display_format="YYYY-MM-DD",
                            style={"backgroundColor": COLORS["card"]},
                        ),
                    ], lg=4, md=6),
                    dbc.Col([
                        html.Label("Direction", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dcc.Dropdown(
                            id="trade-log-direction",
                            options=[
                                {"label": "All", "value": "ALL"},
                                {"label": "Long", "value": "LONG"},
                                {"label": "Short", "value": "SHORT"},
                            ],
                            value="ALL",
                            clearable=False,
                            style={"backgroundColor": "#2c2c44", "color": "#e0e0e0"},
                        ),
                    ], lg=2, md=3),
                    dbc.Col([
                        html.Label("Status", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dcc.Dropdown(
                            id="trade-log-status",
                            options=[
                                {"label": "All", "value": "ALL"},
                                {"label": "Open", "value": "OPEN"},
                                {"label": "Closed", "value": "CLOSED"},
                                {"label": "Cancelled", "value": "CANCELLED"},
                            ],
                            value="ALL",
                            clearable=False,
                            style={"backgroundColor": "#2c2c44", "color": "#e0e0e0"},
                        ),
                    ], lg=2, md=3),
                    dbc.Col([
                        html.Label("\u00a0", style={"fontSize": "0.8rem"}),
                        html.Div([
                            dbc.Button(
                                "Apply Filters",
                                id="trade-log-apply",
                                color="info",
                                size="sm",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Export CSV",
                                id="trade-log-export",
                                color="secondary",
                                size="sm",
                                outline=True,
                            ),
                        ]),
                    ], lg=4, md=12, className="d-flex align-items-end"),
                ], className="g-2"),
            ], style={"backgroundColor": COLORS["card"], "padding": "12px"}),
            style={"backgroundColor": COLORS["card"], "border": "1px solid #3a3a5c", "marginBottom": "12px"},
        ),

        # Summary row
        html.Div(id="trade-log-summary"),

        # Trade table
        html.Div(id="trade-log-table"),

        # Hidden download component
        dcc.Download(id="trade-log-download"),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_callbacks(app):
    @app.callback(
        [
            Output("trade-log-summary", "children"),
            Output("trade-log-table", "children"),
        ],
        [
            Input("trade-log-apply", "n_clicks"),
            Input("interval-refresh", "n_intervals"),
        ],
        [
            State("trade-log-date-range", "start_date"),
            State("trade-log-date-range", "end_date"),
            State("trade-log-direction", "value"),
            State("trade-log-status", "value"),
        ],
    )
    def update_trade_log(_clicks, _n, start_date, end_date, direction, status):
        df = _fetch_trades(direction, status, start_date, end_date)

        # Summary
        if df.empty:
            summary = dbc.Alert(
                "No trades match the selected filters.",
                color="secondary",
                className="mb-2",
                style={"fontSize": "0.85rem"},
            )
        else:
            total = len(df)
            closed = len(df[df["status"] == "CLOSED"])
            total_pnl = df["net_pnl"].sum()
            wins = (df["net_pnl"] > 0).sum()
            pnl_color = COLORS["positive"] if total_pnl >= 0 else COLORS["negative"]
            summary = html.Div(
                dbc.Row([
                    dbc.Col(html.Span(
                        f"Showing {total} trades ({closed} closed)",
                        style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                    ), lg=4),
                    dbc.Col(html.Span([
                        "Net P&L: ",
                        html.Strong(f"${total_pnl:,.2f}", style={"color": pnl_color}),
                    ], style={"fontSize": "0.85rem", "color": COLORS["text"]}), lg=4),
                    dbc.Col(html.Span([
                        "Win Rate: ",
                        html.Strong(
                            f"{wins / closed * 100:.1f}%" if closed > 0 else "N/A",
                            style={"color": COLORS["positive"] if closed > 0 and wins / closed >= 0.5 else COLORS["negative"]},
                        ),
                    ], style={"fontSize": "0.85rem", "color": COLORS["text"]}), lg=4),
                ], className="mb-2"),
            )

        # Table
        if df.empty:
            table = html.P("No trades found", style={"color": COLORS["muted"], "textAlign": "center"})
        else:
            display_df = df.copy()
            for col in ["entry_price", "exit_price", "realized_pnl", "net_pnl", "stop_loss", "take_profit"]:
                display_df[col] = display_df[col].map(
                    lambda v: f"{v:,.2f}" if pd.notna(v) and v != 0 else "-"
                )
            display_df["confidence"] = display_df["confidence"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "-"
            )
            for col in ["entry_time", "exit_time"]:
                display_df[col] = display_df[col].map(
                    lambda v: v.strftime("%Y-%m-%d %H:%M") if v is not None else "-"
                )
            # Select display columns
            show_cols = [
                "trade_id", "instrument", "direction", "status", "entry_time",
                "exit_time", "entry_price", "exit_price", "quantity",
                "net_pnl", "confidence", "regime",
            ]
            display_df = display_df[show_cols]

            table = dash_table.DataTable(
                data=display_df.to_dict("records"),
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_df.columns],
                style_header=TABLE_STYLE_HEADER,
                style_cell={
                    **TABLE_STYLE_CELL,
                    "minWidth": "80px",
                    "maxWidth": "160px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": COLORS["table_row_odd"]},
                    {"if": {"filter_query": '{net_pnl} contains "-"', "column_id": "net_pnl"},
                     "color": COLORS["negative"], "fontWeight": "600"},
                    {"if": {"filter_query": '{net_pnl} not contains "-"', "column_id": "net_pnl"},
                     "color": COLORS["positive"], "fontWeight": "600"},
                    {"if": {"filter_query": '{direction} = "LONG"', "column_id": "direction"},
                     "color": COLORS["positive"]},
                    {"if": {"filter_query": '{direction} = "SHORT"', "column_id": "direction"},
                     "color": COLORS["negative"]},
                    {"if": {"filter_query": '{status} = "OPEN"', "column_id": "status"},
                     "color": COLORS["accent"]},
                    {"if": {"filter_query": '{status} = "CLOSED"', "column_id": "status"},
                     "color": COLORS["muted"]},
                ],
                page_size=25,
                sort_action="native",
                filter_action="native",
                style_filter={
                    "backgroundColor": "#2c2c44",
                    "color": COLORS["text"],
                },
            )

        return summary, table

    @app.callback(
        Output("trade-log-download", "data"),
        Input("trade-log-export", "n_clicks"),
        [
            State("trade-log-date-range", "start_date"),
            State("trade-log-date-range", "end_date"),
            State("trade-log-direction", "value"),
            State("trade-log-status", "value"),
        ],
        prevent_initial_call=True,
    )
    def export_csv(_clicks, start_date, end_date, direction, status):
        df = _fetch_trades(direction, status, start_date, end_date)
        if df.empty:
            return no_update
        return dcc.send_data_frame(df.to_csv, "trade_log.csv", index=False)
