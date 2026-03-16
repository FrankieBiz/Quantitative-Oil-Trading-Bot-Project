"""
EIA Weekly Petroleum Status Report scraper.

Fetches the weekly crude-oil inventory data from the EIA API v2, computes
a *surprise* score (z-score of actual vs consensus relative to historical
standard deviation), and persists results to the SentimentRecord table.
Designed to be scheduled every Wednesday at 10:30 AM ET.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import aiohttp
import numpy as np
from loguru import logger

from config import settings
from db.models import SentimentRecord, get_session


# EIA API v2 series IDs of interest
_SERIES = {
    "crude_stocks": "PET.WCESTUS1.W",       # Weekly U.S. ending stocks of crude oil
    "crude_production": "PET.WCRFPUS2.W",    # Weekly U.S. field production of crude oil
    "crude_imports": "PET.WCEIMUS2.W",        # Weekly U.S. imports of crude oil
    "gasoline_stocks": "PET.WGTSTUS1.W",     # Weekly U.S. ending stocks of total gasoline
    "distillate_stocks": "PET.WDISTUS1.W",   # Weekly U.S. ending stocks of distillate fuel oil
}


class EIAScraper:
    """Fetch and interpret the EIA Weekly Petroleum Status Report.

    The scraper pulls the latest data point from the EIA API v2 for each
    tracked series, computes a *surprise* z-score against the trailing
    historical window, and stores the results both as a return value and
    in the ``SentimentRecord`` database table (source ``eia``).

    Usage::

        scraper = EIAScraper()
        report = await scraper.fetch_weekly_report()
        # report = {
        #     "crude_stocks": {"actual": ..., "surprise": ..., ...},
        #     ...
        #     "report_date": "2025-05-07",
        #     "composite_surprise": 1.23,
        # }

    To run on a schedule::

        await scraper.run_scheduled()   # blocks until cancelled
    """

    # Number of historical data points used to compute the standard deviation
    HISTORY_DEPTH = 52  # ~1 year of weekly data

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.getenv("EIA_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "EIA_API_KEY is empty.  EIA data fetches will fail."
            )
        self._base_url = settings.EIA_API_URL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_weekly_report(self) -> Dict[str, Any]:
        """Fetch the latest EIA weekly report and compute surprise scores.

        Returns
        -------
        dict
            Per-series ``actual``, ``previous``, ``change``, ``std``,
            ``surprise``; plus ``report_date`` and ``composite_surprise``.
        """
        report: Dict[str, Any] = {}
        surprises: List[float] = []

        async with aiohttp.ClientSession() as http:
            for label, series_id in _SERIES.items():
                try:
                    data = await self._fetch_series(http, series_id)
                    if data is None:
                        logger.warning("No data returned for series {}", label)
                        continue

                    values = [pt["value"] for pt in data if pt.get("value") is not None]
                    if len(values) < 2:
                        logger.warning("Insufficient data points for {}", label)
                        continue

                    actual = values[0]
                    previous = values[1]
                    change = actual - previous

                    # Historical window for std calculation
                    hist = np.array(values[: self.HISTORY_DEPTH], dtype=float)
                    changes = np.diff(hist)
                    std = float(np.std(changes)) if len(changes) > 1 else 1.0

                    # Surprise = (actual - consensus) / std
                    # We use the previous value as a naive consensus proxy
                    surprise = float(change / std) if std > 0 else 0.0

                    report_date = data[0].get("period", "")

                    report[label] = {
                        "actual": actual,
                        "previous": previous,
                        "change": change,
                        "std": std,
                        "surprise": surprise,
                        "report_date": report_date,
                        "series_id": series_id,
                    }
                    surprises.append(surprise)

                except Exception:
                    logger.exception("Error fetching EIA series {}", label)

        if surprises:
            report["composite_surprise"] = float(np.mean(surprises))
        else:
            report["composite_surprise"] = 0.0

        report["report_date"] = report.get(
            "crude_stocks", {}
        ).get("report_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        # Persist
        self._store_report(report)

        logger.info(
            "EIA weekly report fetched. Composite surprise: {:.3f}",
            report["composite_surprise"],
        )
        return report

    async def run_scheduled(self) -> None:
        """Run in a loop, triggering a fetch every Wednesday at 10:30 AM ET.

        This coroutine sleeps until the next scheduled time and then
        fetches the report.  It runs indefinitely until cancelled.
        """
        tz = ZoneInfo(settings.EIA_REPORT_TZ)

        while True:
            now = datetime.now(tz)
            target = self._next_report_time(now, tz)
            wait_seconds = (target - now).total_seconds()

            logger.info(
                "EIA scheduler: next fetch at {} ({:.1f}h from now)",
                target.isoformat(),
                wait_seconds / 3600,
            )
            await asyncio.sleep(wait_seconds)

            try:
                await self.fetch_weekly_report()
            except Exception:
                logger.exception("Scheduled EIA fetch failed")

    # ------------------------------------------------------------------
    # EIA API interaction
    # ------------------------------------------------------------------

    async def _fetch_series(
        self, http: aiohttp.ClientSession, series_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Call EIA API v2 and return a list of data points (newest first).

        Parameters
        ----------
        http:
            Active aiohttp session.
        series_id:
            EIA series identifier (e.g. ``PET.WCESTUS1.W``).

        Returns
        -------
        list[dict] | None
            Each dict has at least ``period`` and ``value`` keys.
        """
        # Decompose the series_id into route segments for API v2
        parts = series_id.split(".")
        # Build query params
        params = {
            "api_key": self._api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": str(self.HISTORY_DEPTH + 5),
        }

        # Use the general petroleum summary endpoint
        url = self._base_url

        try:
            async with http.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "EIA API returned HTTP {} for {}: {}",
                        resp.status,
                        series_id,
                        body[:300],
                    )
                    return None

                payload = await resp.json()
                data = (
                    payload.get("response", {}).get("data", [])
                )
                if not data:
                    # Fallback: try legacy v1-style endpoint
                    return await self._fetch_series_v1(http, series_id)

                # Normalize to [{period, value}, ...]
                normalized: List[Dict[str, Any]] = []
                for row in data:
                    try:
                        normalized.append(
                            {
                                "period": row.get("period", ""),
                                "value": float(row["value"])
                                if row.get("value") is not None
                                else None,
                            }
                        )
                    except (ValueError, TypeError):
                        continue
                return normalized

        except Exception:
            logger.exception("HTTP error fetching EIA series {}", series_id)
            return None

    async def _fetch_series_v1(
        self, http: aiohttp.ClientSession, series_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Fallback fetcher using the EIA open-data v1 JSON endpoint.

        ``https://api.eia.gov/series/?api_key=...&series_id=...``
        """
        url = "https://api.eia.gov/series/"
        params = {
            "api_key": self._api_key,
            "series_id": series_id,
        }

        try:
            async with http.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()

            series_list = payload.get("series", [])
            if not series_list:
                return None

            raw_data = series_list[0].get("data", [])
            normalized: List[Dict[str, Any]] = []
            for point in raw_data:
                # v1 format: [[date_str, value], ...]
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        normalized.append(
                            {
                                "period": str(point[0]),
                                "value": float(point[1])
                                if point[1] is not None
                                else None,
                            }
                        )
                    except (ValueError, TypeError):
                        continue
            return normalized

        except Exception:
            logger.exception("v1 fallback failed for series {}", series_id)
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _store_report(self, report: Dict[str, Any]) -> None:
        """Persist the EIA report as a SentimentRecord row.

        The ``text`` field contains a human-readable summary, and the
        ``sentiment_score`` is set to the composite surprise value so
        that downstream aggregation can consume it directly.
        """
        composite = report.get("composite_surprise", 0.0)
        report_date = report.get("report_date", "unknown")

        # Build a readable summary
        lines = [f"EIA Weekly Petroleum Status Report ({report_date})"]
        for label in _SERIES:
            entry = report.get(label)
            if entry is None:
                continue
            lines.append(
                f"  {label}: actual={entry['actual']:.1f}  "
                f"change={entry['change']:+.1f}  "
                f"surprise={entry['surprise']:+.2f}σ"
            )
        lines.append(f"  Composite surprise: {composite:+.3f}σ")
        text = "\n".join(lines)

        # Determine a unique source_id
        source_id = f"eia_weekly_{report_date}"

        session = get_session()
        try:
            # Map composite surprise to [0,1] positive/negative split
            if composite >= 0:
                positive = min(abs(composite) / 3.0, 1.0)
                negative = 0.0
            else:
                positive = 0.0
                negative = min(abs(composite) / 3.0, 1.0)
            neutral = max(0.0, 1.0 - positive - negative)

            record = SentimentRecord(
                source="eia",
                source_id=source_id,
                text=text,
                sentiment_score=float(np.clip(composite, -1.0, 1.0)),
                positive=positive,
                negative=negative,
                neutral=neutral,
                timestamp=datetime.now(timezone.utc),
                author="EIA",
                url=f"https://www.eia.gov/petroleum/supply/weekly/",
            )
            session.merge(record)
            session.commit()
            logger.debug("Stored EIA report as SentimentRecord: {}", source_id)
        except Exception:
            session.rollback()
            logger.exception("Failed to store EIA report")
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_report_time(now: datetime, tz: ZoneInfo) -> datetime:
        """Compute the next Wednesday 10:30 AM ET from *now*.

        If *now* is already past Wednesday 10:30 this week, returns next
        Wednesday.
        """
        # Wednesday = 2 in Python's weekday()
        target_weekday = 2
        days_ahead = (target_weekday - now.weekday()) % 7

        candidate = now.replace(
            hour=settings.EIA_REPORT_HOUR,
            minute=settings.EIA_REPORT_MINUTE,
            second=0,
            microsecond=0,
        )

        if days_ahead == 0 and now >= candidate:
            days_ahead = 7

        from datetime import timedelta

        target = candidate + timedelta(days=days_ahead)
        return target
