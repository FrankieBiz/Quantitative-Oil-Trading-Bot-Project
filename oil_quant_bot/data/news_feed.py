"""
News data feed via newsapi-python and feedparser.

Collects articles from the NewsAPI search endpoint and from RSS feeds
(Reuters Energy, EIA Today-in-Energy).  Articles are classified as
``news_major`` (Reuters, Bloomberg, AP) or ``news_minor``, deduplicated
by URL, and persisted to the SentimentRecord database table.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
import feedparser
from newsapi import NewsApiClient
from loguru import logger

from config import settings
from db.models import SentimentRecord, get_session


def _classify_source(source_name: str, url: str) -> str:
    """Return ``news_major`` or ``news_minor`` based on the source.

    Major outlets: Reuters, Bloomberg, Associated Press (AP).
    Everything else is treated as minor.
    """
    name_lower = (source_name or "").lower()
    domain = urlparse(url).netloc.lower() if url else ""
    for major in settings.MAJOR_NEWS_SOURCES:
        if major in name_lower or major in domain:
            return "news_major"
    return "news_minor"


class NewsFeed:
    """Asynchronous news collector combining NewsAPI and RSS feeds.

    Usage::

        feed = NewsFeed()
        articles = await feed.fetch_all()
    """

    KEYWORDS: List[str] = settings.NEWS_KEYWORDS
    RSS_FEEDS: Dict[str, str] = settings.RSS_FEEDS

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or settings.NEWSAPI_KEY
        if not self._api_key:
            logger.warning(
                "NEWSAPI_KEY is empty. NewsAPI searches will be skipped."
            )
        self._newsapi = (
            NewsApiClient(api_key=self._api_key) if self._api_key else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_all(self) -> List[Dict]:
        """Run NewsAPI and RSS fetches concurrently, merge and persist.

        Returns
        -------
        list[dict]
            Each dict contains ``text``, ``timestamp``, ``author``,
            ``source_id`` (URL), ``source`` (``news_major``/``news_minor``),
            ``title``, ``url``.
        """
        existing_urls = self._load_existing_urls()

        newsapi_task = asyncio.ensure_future(
            self.fetch_newsapi(existing_urls)
        )
        rss_task = asyncio.ensure_future(self.fetch_rss(existing_urls))

        newsapi_results, rss_results = await asyncio.gather(
            newsapi_task, rss_task, return_exceptions=True
        )

        articles: List[Dict] = []
        if isinstance(newsapi_results, list):
            articles.extend(newsapi_results)
        else:
            logger.exception("NewsAPI fetch failed: {}", newsapi_results)

        if isinstance(rss_results, list):
            articles.extend(rss_results)
        else:
            logger.exception("RSS fetch failed: {}", rss_results)

        # Final dedup across both sources (by URL)
        seen_urls: Set[str] = set(existing_urls)
        deduped: List[Dict] = []
        for art in articles:
            url = art.get("url") or art.get("source_id", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(art)

        if deduped:
            self._store_records(deduped)

        logger.info(
            "NewsFeed: collected {} new articles (NewsAPI + RSS)", len(deduped)
        )
        return deduped

    # ------------------------------------------------------------------
    # NewsAPI
    # ------------------------------------------------------------------

    async def fetch_newsapi(
        self, existing_urls: Optional[Set[str]] = None
    ) -> List[Dict]:
        """Search NewsAPI ``everything`` endpoint for each keyword.

        Parameters
        ----------
        existing_urls:
            Set of URLs already stored.  Passed in to avoid re-querying DB.

        Returns
        -------
        list[dict]
        """
        if self._newsapi is None:
            logger.debug("NewsAPI client not initialized; skipping.")
            return []

        existing_urls = existing_urls if existing_urls is not None else self._load_existing_urls()
        seen: Set[str] = set(existing_urls)
        results: List[Dict] = []
        loop = asyncio.get_running_loop()

        for keyword in self.KEYWORDS:
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda kw=keyword: self._newsapi.get_everything(
                        q=kw,
                        language="en",
                        sort_by="publishedAt",
                        page_size=100,
                    ),
                )

                for article in response.get("articles", []):
                    url = article.get("url", "")
                    if not url or url in seen:
                        continue
                    seen.add(url)

                    source_name = (
                        article.get("source", {}).get("name", "") or ""
                    )
                    classification = _classify_source(source_name, url)

                    published = article.get("publishedAt")
                    ts = self._parse_timestamp(published)

                    title = article.get("title", "") or ""
                    description = article.get("description", "") or ""
                    text = f"{title}. {description}".strip(". ")

                    results.append(
                        {
                            "text": text,
                            "timestamp": ts,
                            "author": article.get("author") or source_name,
                            "source_id": url,
                            "source": classification,
                            "title": title,
                            "url": url,
                        }
                    )

            except Exception:
                logger.exception("NewsAPI error for keyword '{}'", keyword)

        logger.debug("NewsAPI returned {} unique articles", len(results))
        return results

    # ------------------------------------------------------------------
    # RSS Feeds
    # ------------------------------------------------------------------

    async def fetch_rss(
        self, existing_urls: Optional[Set[str]] = None
    ) -> List[Dict]:
        """Fetch and parse configured RSS feeds asynchronously.

        Each feed is downloaded with ``aiohttp`` and parsed with
        ``feedparser``.  Reuters is classified as ``news_major``; EIA as
        ``news_minor``.

        Returns
        -------
        list[dict]
        """
        existing_urls = existing_urls if existing_urls is not None else self._load_existing_urls()
        seen: Set[str] = set(existing_urls)
        results: List[Dict] = []

        async with aiohttp.ClientSession() as http:
            for feed_name, feed_url in self.RSS_FEEDS.items():
                try:
                    async with http.get(feed_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            logger.warning(
                                "RSS feed '{}' returned HTTP {}", feed_name, resp.status
                            )
                            continue
                        raw = await resp.text()

                    parsed = feedparser.parse(raw)

                    for entry in parsed.entries:
                        link = entry.get("link", "")
                        if not link or link in seen:
                            continue
                        seen.add(link)

                        source_name = feed_name
                        classification = _classify_source(source_name, link)

                        title = entry.get("title", "")
                        summary = entry.get("summary", "")
                        text = f"{title}. {summary}".strip(". ")

                        published = entry.get("published_parsed") or entry.get(
                            "updated_parsed"
                        )
                        if published:
                            ts = datetime(*published[:6], tzinfo=timezone.utc)
                        else:
                            ts = datetime.now(timezone.utc)

                        results.append(
                            {
                                "text": text,
                                "timestamp": ts,
                                "author": entry.get("author", source_name),
                                "source_id": link,
                                "source": classification,
                                "title": title,
                                "url": link,
                            }
                        )

                except Exception:
                    logger.exception("Error fetching RSS feed '{}'", feed_name)

        logger.debug("RSS returned {} unique articles", len(results))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_existing_urls(self) -> Set[str]:
        """Return URLs already persisted in the SentimentRecord table."""
        session = get_session()
        try:
            rows = (
                session.query(SentimentRecord.source_id)
                .filter(
                    SentimentRecord.source.in_(["news_major", "news_minor"])
                )
                .filter(SentimentRecord.source_id.isnot(None))
                .all()
            )
            return {r[0] for r in rows}
        except Exception:
            logger.exception("Failed to load existing news URLs")
            return set()
        finally:
            session.close()

    def _store_records(self, articles: List[Dict]) -> None:
        """Persist articles to the SentimentRecord table.

        Sentiment scores are initialized to zero and will be populated
        by the downstream sentiment engine after FinBERT inference.
        """
        session = get_session()
        try:
            for art in articles:
                record = SentimentRecord(
                    source=art["source"],
                    source_id=art.get("url") or art.get("source_id"),
                    text=art["text"],
                    sentiment_score=0.0,
                    positive=0.0,
                    negative=0.0,
                    neutral=0.0,
                    timestamp=art["timestamp"],
                    author=art.get("author"),
                    url=art.get("url"),
                )
                session.merge(record)
            session.commit()
            logger.debug("Stored {} news records", len(articles))
        except Exception:
            session.rollback()
            logger.exception("Failed to store news records")
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(value) -> datetime:
        """Best-effort parsing of a timestamp string to a tz-aware datetime."""
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                continue
        return datetime.now(timezone.utc)
