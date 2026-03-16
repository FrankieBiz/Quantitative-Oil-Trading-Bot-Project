"""
Twitter / X data feed via tweepy (v2 API).

Searches oil-related keywords, deduplicates by tweet ID, filters by
minimum follower count, and persists results to the SentimentRecord
database table.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import aiohttp
import tweepy
from tweepy import asynchronous as tweepy_async
from loguru import logger

from config import settings
from db.models import SentimentRecord, get_session


class TwitterFeed:
    """Asynchronous Twitter data collector for oil-market sentiment.

    Authenticates with a Bearer token (Twitter API v2) and searches for
    tweets matching preconfigured energy/oil keywords.  Results are
    deduplicated, filtered by minimum follower count, stored in the DB,
    and returned as a list of plain dicts for downstream processing.

    Usage::

        feed = TwitterFeed()
        tweets = await feed.fetch_tweets()
    """

    # Keywords to search
    KEYWORDS: List[str] = settings.TWITTER_KEYWORDS
    MAX_RESULTS_PER_KEYWORD: int = settings.TWITTER_MAX_RESULTS_PER_KEYWORD
    MIN_FOLLOWERS: int = settings.TWITTER_MIN_FOLLOWERS

    def __init__(self, bearer_token: Optional[str] = None) -> None:
        self._bearer_token = bearer_token or settings.TWITTER_BEARER_TOKEN
        if not self._bearer_token:
            logger.warning(
                "TWITTER_BEARER_TOKEN is empty. TwitterFeed will not be able to fetch data."
            )
        self._client = tweepy.Client(
            bearer_token=self._bearer_token,
            wait_on_rate_limit=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_tweets(self) -> List[Dict]:
        """Search tweets for every keyword, deduplicate, filter, persist.

        Returns
        -------
        list[dict]
            Each dict contains ``text``, ``timestamp``, ``author``,
            ``source_id``, ``author_followers``.
        """
        seen_ids: Set[str] = set()
        collected: List[Dict] = []

        # Load already-stored tweet IDs so we skip re-processing
        existing_ids = self._load_existing_ids()
        seen_ids.update(existing_ids)

        for keyword in self.KEYWORDS:
            try:
                results = await self._search_keyword(keyword, seen_ids)
                collected.extend(results)
            except Exception:
                logger.exception("Error searching Twitter for '{}'", keyword)

        if collected:
            self._store_records(collected)

        logger.info(
            "TwitterFeed: collected {} new tweets across {} keywords",
            len(collected),
            len(self.KEYWORDS),
        )
        return collected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _search_keyword(
        self, keyword: str, seen_ids: Set[str]
    ) -> List[Dict]:
        """Search a single keyword and return filtered, deduplicated dicts.

        The Twitter v2 ``search_recent_tweets`` endpoint is called in a
        thread-pool executor because ``tweepy.Client`` is synchronous.
        """
        loop = asyncio.get_running_loop()

        def _do_search() -> tweepy.Response | None:
            return self._client.search_recent_tweets(
                query=f'"{keyword}" -is:retweet lang:en',
                max_results=min(self.MAX_RESULTS_PER_KEYWORD, 100),
                tweet_fields=["created_at", "author_id", "public_metrics", "text"],
                user_fields=["username", "public_metrics"],
                expansions=["author_id"],
            )

        response: tweepy.Response | None = await loop.run_in_executor(
            None, _do_search
        )

        if response is None or response.data is None:
            return []

        # Build author lookup from includes
        authors: Dict[str, dict] = {}
        if response.includes and "users" in response.includes:
            for user in response.includes["users"]:
                authors[str(user.id)] = {
                    "username": user.username,
                    "followers": user.public_metrics.get("followers_count", 0)
                    if user.public_metrics
                    else 0,
                }

        results: List[Dict] = []
        for tweet in response.data:
            tweet_id = str(tweet.id)

            # Deduplicate
            if tweet_id in seen_ids:
                continue
            seen_ids.add(tweet_id)

            # Follower filter
            author_info = authors.get(str(tweet.author_id), {})
            followers = author_info.get("followers", 0)
            if followers < self.MIN_FOLLOWERS:
                continue

            created_at = tweet.created_at
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            results.append(
                {
                    "text": tweet.text,
                    "timestamp": created_at or datetime.now(timezone.utc),
                    "author": author_info.get("username", "unknown"),
                    "source_id": tweet_id,
                    "author_followers": followers,
                }
            )

        logger.debug(
            "Keyword '{}': {} tweets after dedup/filter", keyword, len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_existing_ids(self) -> Set[str]:
        """Return the set of tweet source_ids already in the database."""
        session = get_session()
        try:
            rows = (
                session.query(SentimentRecord.source_id)
                .filter(SentimentRecord.source == "twitter")
                .filter(SentimentRecord.source_id.isnot(None))
                .all()
            )
            return {r[0] for r in rows}
        except Exception:
            logger.exception("Failed to load existing tweet IDs")
            return set()
        finally:
            session.close()

    def _store_records(self, tweets: List[Dict]) -> None:
        """Persist tweet records into the SentimentRecord table.

        Sentiment scores are initialized to zero; the downstream
        sentiment engine will update them after FinBERT inference.
        """
        session = get_session()
        try:
            for tw in tweets:
                record = SentimentRecord(
                    source="twitter",
                    source_id=tw["source_id"],
                    text=tw["text"],
                    sentiment_score=0.0,
                    positive=0.0,
                    negative=0.0,
                    neutral=0.0,
                    timestamp=tw["timestamp"],
                    author=tw["author"],
                    url=None,
                )
                session.merge(record)  # merge handles duplicate source_id
            session.commit()
            logger.debug("Stored {} tweet records", len(tweets))
        except Exception:
            session.rollback()
            logger.exception("Failed to store tweet records")
        finally:
            session.close()
