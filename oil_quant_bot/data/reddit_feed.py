"""
Reddit oil-sentiment feed — **no API key required**.

Uses Reddit's public JSON endpoints (``old.reddit.com/…/.json``) to fetch
recent posts from oil/energy subreddits.  Posts are deduplicated by Reddit
ID, persisted to the SentimentRecord table (source ``reddit``), and returned
for downstream FinBERT sentiment analysis.

This module is a free replacement for the Twitter feed — no account,
no OAuth tokens, no rate-limit tiers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import aiohttp
from loguru import logger

from config import settings
from db.models import SentimentRecord, get_session


# Subreddits with active oil / energy / commodities discussion
_DEFAULT_SUBREDDITS = [
    "oil",
    "energy",
    "commodities",
    "wallstreetbets",
    "investing",
    "stocks",
]

# Keywords to filter posts (must contain at least one)
_OIL_KEYWORDS = {
    "oil", "crude", "wti", "brent", "opec", "petroleum", "barrel",
    "uso", "xle", "refinery", "pipeline", "eia", "energy",
    "gasoline", "diesel", "lng", "natural gas", "sanctions",
}

# Polite user-agent (Reddit blocks default python-requests UA)
_USER_AGENT = "OilQuantBot/1.0 (oil sentiment research; async)"


class RedditFeed:
    """Fetch oil-related posts from Reddit — no API key needed.

    Usage::

        feed = RedditFeed()
        posts = await feed.fetch_posts()
    """

    def __init__(
        self,
        subreddits: Optional[List[str]] = None,
        max_posts_per_sub: int = 25,
    ) -> None:
        self._subreddits = subreddits or _DEFAULT_SUBREDDITS
        self._max_posts = max_posts_per_sub

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_posts(self) -> List[Dict]:
        """Fetch recent posts from all configured subreddits.

        Returns
        -------
        list[dict]
            Each dict has ``text``, ``timestamp``, ``author``,
            ``source_id`` (Reddit post ID), ``url``, ``score``.
        """
        existing_ids = self._load_existing_ids()
        all_posts: List[Dict] = []

        headers = {"User-Agent": _USER_AGENT}
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as http:
            tasks = [
                self._fetch_subreddit(http, sub, existing_ids)
                for sub in self._subreddits
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, list):
                    all_posts.extend(result)
                else:
                    logger.warning(
                        "Reddit fetch failed for r/{}: {}",
                        self._subreddits[i],
                        result,
                    )

        # Deduplicate across subreddits (cross-posts)
        seen: Set[str] = set(existing_ids)
        deduped: List[Dict] = []
        for post in all_posts:
            if post["source_id"] not in seen:
                seen.add(post["source_id"])
                deduped.append(post)

        if deduped:
            self._store_records(deduped)

        logger.info("RedditFeed: collected {} new posts", len(deduped))
        return deduped

    # ------------------------------------------------------------------
    # Subreddit fetching
    # ------------------------------------------------------------------

    async def _fetch_subreddit(
        self,
        http: aiohttp.ClientSession,
        subreddit: str,
        existing_ids: Set[str],
    ) -> List[Dict]:
        """Fetch and filter posts from a single subreddit."""
        url = f"https://old.reddit.com/r/{subreddit}/new/.json"
        params = {"limit": str(self._max_posts), "raw_json": "1"}

        try:
            async with http.get(url, params=params) as resp:
                if resp.status == 429:
                    logger.warning("Reddit rate-limited on r/{}", subreddit)
                    return []
                if resp.status != 200:
                    logger.warning(
                        "Reddit r/{} returned HTTP {}", subreddit, resp.status
                    )
                    return []
                data = await resp.json()
        except Exception:
            logger.exception("Error fetching r/{}", subreddit)
            return []

        posts: List[Dict] = []
        children = data.get("data", {}).get("children", [])

        for child in children:
            post_data = child.get("data", {})
            post_id = post_data.get("id", "")

            if not post_id or post_id in existing_ids:
                continue

            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            text = f"{title}. {selftext}".strip(". ") if selftext else title

            # Filter: must mention oil-related keywords
            text_lower = text.lower()
            if not any(kw in text_lower for kw in _OIL_KEYWORDS):
                continue

            created_utc = post_data.get("created_utc", 0)
            timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)

            posts.append({
                "text": text[:2000],  # Cap length for FinBERT
                "timestamp": timestamp,
                "author": post_data.get("author", "unknown"),
                "source_id": post_id,
                "url": f"https://reddit.com{post_data.get('permalink', '')}",
                "score": post_data.get("score", 0),
                "subreddit": subreddit,
            })

        logger.debug("r/{}: {} oil-related posts", subreddit, len(posts))
        return posts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_existing_ids(self) -> Set[str]:
        """Return Reddit post IDs already stored in the database."""
        session = get_session()
        try:
            rows = (
                session.query(SentimentRecord.source_id)
                .filter(SentimentRecord.source == "reddit")
                .filter(SentimentRecord.source_id.isnot(None))
                .all()
            )
            return {r[0] for r in rows}
        except Exception:
            logger.exception("Failed to load existing Reddit IDs")
            return set()
        finally:
            session.close()

    def _store_records(self, posts: List[Dict]) -> None:
        """Persist Reddit posts to the SentimentRecord table."""
        session = get_session()
        try:
            for post in posts:
                record = SentimentRecord(
                    source="reddit",
                    source_id=post["source_id"],
                    text=post["text"],
                    sentiment_score=0.0,
                    positive=0.0,
                    negative=0.0,
                    neutral=0.0,
                    timestamp=post["timestamp"],
                    author=post["author"],
                    url=post["url"],
                )
                session.merge(record)
            session.commit()
            logger.debug("Stored {} Reddit records", len(posts))
        except Exception:
            session.rollback()
            logger.exception("Failed to store Reddit records")
        finally:
            session.close()
