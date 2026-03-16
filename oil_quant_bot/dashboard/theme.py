"""
Shared theme constants for the Oil Quant Bot Dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html

COLORS = {
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

TABLE_STYLE_HEADER = {
    "backgroundColor": COLORS["table_header"],
    "color": COLORS["text"],
    "fontWeight": "600",
    "border": "none",
}

TABLE_STYLE_CELL = {
    "backgroundColor": COLORS["card"],
    "color": COLORS["text"],
    "border": "1px solid #3a3a5c",
    "textAlign": "center",
    "padding": "8px",
    "fontSize": "0.85rem",
}


def card(title: str, body_id: str, children=None) -> dbc.Card:
    """Standard styled dashboard card."""
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
