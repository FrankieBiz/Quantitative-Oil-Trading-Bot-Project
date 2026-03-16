"""
System Logs page for the Oil Quant Bot Dashboard.

Real-time log viewer with severity filtering, auto-scroll,
and log file reading from the configured log directory.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import List

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from loguru import logger

from config import settings
from dashboard.theme import COLORS, card

# ---------------------------------------------------------------------------
# Log reading
# ---------------------------------------------------------------------------

# In-memory ring buffer for recent log entries (populated by loguru sink)
_LOG_BUFFER: deque = deque(maxlen=500)


def _log_sink(message):
    """Loguru sink that captures log records into the ring buffer."""
    record = message.record
    _LOG_BUFFER.append({
        "time": record["time"].strftime("%Y-%m-%d %H:%M:%S"),
        "level": record["level"].name,
        "module": record["module"],
        "message": record["message"],
    })


# Install the sink (idempotent — loguru deduplicates by function identity)
try:
    logger.add(_log_sink, level="DEBUG", format="{message}")
except Exception:
    pass


def _read_log_file(tail_lines: int = 200) -> List[dict]:
    """Read the most recent lines from the log file on disk."""
    log_dir = Path(settings.LOG_DIR)
    if not log_dir.exists():
        return []

    # Find the most recent .log file
    log_files = sorted(log_dir.glob("*.log"), key=os.path.getmtime, reverse=True)
    if not log_files:
        return []

    lines = []
    try:
        with open(log_files[0], "r") as f:
            all_lines = f.readlines()
            for raw in all_lines[-tail_lines:]:
                raw = raw.strip()
                if not raw:
                    continue
                # Parse loguru format: YYYY-MM-DD HH:MM:SS | LEVEL | module:func:line - message
                parts = raw.split("|", 2)
                if len(parts) >= 3:
                    lines.append({
                        "time": parts[0].strip()[:19],
                        "level": parts[1].strip().upper(),
                        "module": "",
                        "message": parts[2].strip(),
                    })
                else:
                    lines.append({
                        "time": "",
                        "level": "INFO",
                        "module": "",
                        "message": raw,
                    })
    except Exception:
        pass
    return lines


def _get_logs(level_filter: str = "ALL", source: str = "buffer") -> List[dict]:
    """Get log entries, optionally filtered by level."""
    if source == "file":
        entries = _read_log_file()
    else:
        entries = list(_LOG_BUFFER)

    if level_filter and level_filter != "ALL":
        entries = [e for e in entries if e["level"] == level_filter]

    return entries


# ---------------------------------------------------------------------------
# Level coloring
# ---------------------------------------------------------------------------

_LEVEL_COLORS = {
    "DEBUG": COLORS["muted"],
    "INFO": COLORS["accent"],
    "WARNING": COLORS["warning"],
    "ERROR": COLORS["negative"],
    "CRITICAL": "#ff1744",
    "SUCCESS": COLORS["positive"],
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def layout() -> html.Div:
    return html.Div([
        # Controls bar
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Log Level", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dcc.Dropdown(
                            id="log-level-filter",
                            options=[
                                {"label": "All Levels", "value": "ALL"},
                                {"label": "DEBUG", "value": "DEBUG"},
                                {"label": "INFO", "value": "INFO"},
                                {"label": "WARNING", "value": "WARNING"},
                                {"label": "ERROR", "value": "ERROR"},
                                {"label": "CRITICAL", "value": "CRITICAL"},
                            ],
                            value="ALL",
                            clearable=False,
                            style={"backgroundColor": "#2c2c44", "color": "#e0e0e0"},
                        ),
                    ], lg=2, md=4),
                    dbc.Col([
                        html.Label("Source", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dcc.Dropdown(
                            id="log-source",
                            options=[
                                {"label": "Live Buffer", "value": "buffer"},
                                {"label": "Log File", "value": "file"},
                            ],
                            value="buffer",
                            clearable=False,
                            style={"backgroundColor": "#2c2c44", "color": "#e0e0e0"},
                        ),
                    ], lg=2, md=4),
                    dbc.Col([
                        html.Label("Auto-refresh", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dbc.Switch(
                            id="log-auto-refresh",
                            value=True,
                            label="",
                            style={"marginTop": "4px"},
                        ),
                    ], lg=1, md=2),
                    dbc.Col([
                        html.Label("\u00a0", style={"fontSize": "0.8rem"}),
                        html.Div(id="log-count-badge"),
                    ], lg=7, md=12, className="d-flex align-items-end justify-content-end"),
                ], className="g-2"),
            ], style={"backgroundColor": COLORS["card"], "padding": "12px"}),
            style={"backgroundColor": COLORS["card"], "border": "1px solid #3a3a5c", "marginBottom": "12px"},
        ),

        # Log display area
        html.Div(
            id="log-viewer",
            style={
                "backgroundColor": "#0d0d1a",
                "border": "1px solid #3a3a5c",
                "borderRadius": "4px",
                "padding": "12px",
                "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
                "fontSize": "0.8rem",
                "maxHeight": "600px",
                "overflowY": "auto",
                "whiteSpace": "pre-wrap",
                "lineHeight": "1.5",
            },
        ),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_callbacks(app):
    @app.callback(
        [
            Output("log-viewer", "children"),
            Output("log-count-badge", "children"),
        ],
        [
            Input("interval-refresh", "n_intervals"),
            Input("log-level-filter", "value"),
            Input("log-source", "value"),
        ],
        State("log-auto-refresh", "value"),
    )
    def update_logs(_n, level_filter, source, auto_refresh):
        entries = _get_logs(level_filter, source)

        count_badge = dbc.Badge(
            f"{len(entries)} entries",
            color="info",
            className="p-2",
        )

        if not entries:
            return (
                html.P("No log entries", style={"color": COLORS["muted"], "textAlign": "center"}),
                count_badge,
            )

        # Build log lines
        log_lines = []
        for entry in reversed(entries):  # newest first
            level = entry.get("level", "INFO")
            color = _LEVEL_COLORS.get(level, COLORS["text"])
            time_str = entry.get("time", "")
            message = entry.get("message", "")

            log_lines.append(
                html.Div([
                    html.Span(f"{time_str} ", style={"color": COLORS["muted"]}),
                    html.Span(
                        f"[{level:>8}] ",
                        style={"color": color, "fontWeight": "600"},
                    ),
                    html.Span(message, style={"color": COLORS["text"]}),
                ], style={"borderBottom": "1px solid #1a1a2e", "padding": "2px 0"})
            )

        return log_lines, count_badge
