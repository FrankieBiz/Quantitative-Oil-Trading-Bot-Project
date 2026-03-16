"""
Sentiment Engine for the Oil Quantitative Trading Bot.

Analyzes text using FinBERT and computes a Composite Sentiment Signal (CSS)
by weighting sources, applying recency decay, and tracking momentum.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from transformers import pipeline

from config import settings
from db.models import SentimentRecord, CompositeSentiment, get_session


class SentimentEngine:
    """Runs FinBERT sentiment analysis and computes Composite Sentiment Signal."""

    def __init__(self) -> None:
        logger.info("Initialising SentimentEngine ...")
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=settings.FINBERT_MODEL,
            tokenizer=settings.FINBERT_MODEL,
            truncation=True,
            max_length=settings.FINBERT_MAX_LENGTH,
            return_all_scores=True,
        )
        self._label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
        logger.info("FinBERT model loaded successfully.")

    # ------------------------------------------------------------------
    # Single-text analysis
    # ------------------------------------------------------------------
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyse a single text string and return sentiment scores.

        Returns
        -------
        dict
            Keys: positive, negative, neutral, sentiment_score.
            sentiment_score = positive - negative in [-1, 1].
        """
        if not text or not text.strip():
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "sentiment_score": 0.0}

        results = self._pipeline(text[:settings.FINBERT_MAX_LENGTH * 5])[0]
        scores: Dict[str, float] = {}
        for entry in results:
            label = entry["label"].lower()
            scores[label] = float(entry["score"])

        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        neutral = scores.get("neutral", 0.0)

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "sentiment_score": positive - negative,
        }

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """Analyse a list of texts in batches for efficiency.

        Parameters
        ----------
        texts : list[str]
            Raw text strings.
        batch_size : int
            Number of texts per forward pass.

        Returns
        -------
        list[dict]
            One sentiment dict per input text.
        """
        if not texts:
            return []

        # Truncate to a character limit that safely fits 512 tokens
        truncated = [t[:settings.FINBERT_MAX_LENGTH * 5] if t else "" for t in texts]

        all_results: List[Dict[str, float]] = []
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i : i + batch_size]
            logger.debug("Processing sentiment batch {}-{} / {}", i, i + len(batch), len(truncated))
            batch_outputs = self._pipeline(batch)
            for output in batch_outputs:
                scores: Dict[str, float] = {}
                for entry in output:
                    label = entry["label"].lower()
                    scores[label] = float(entry["score"])
                positive = scores.get("positive", 0.0)
                negative = scores.get("negative", 0.0)
                neutral = scores.get("neutral", 0.0)
                all_results.append({
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "sentiment_score": positive - negative,
                })

        return all_results

    # ------------------------------------------------------------------
    # Composite Sentiment Signal
    # ------------------------------------------------------------------
    def compute_css(self, now: Optional[datetime] = None) -> float:
        """Compute the Composite Sentiment Signal (CSS).

        CSS = sum(score_i * weight_i * decay_i) / sum(weight_i)

        Reads SentimentRecord rows from the last 48 hours, applies per-source
        weighting and exponential recency decay.

        Parameters
        ----------
        now : datetime, optional
            Reference time.  Defaults to utcnow().

        Returns
        -------
        float
            CSS value in [-1, 1].
        """
        if now is None:
            now = datetime.utcnow()

        cutoff = now - timedelta(hours=48)
        session = get_session()
        try:
            records: List[SentimentRecord] = (
                session.query(SentimentRecord)
                .filter(SentimentRecord.timestamp >= cutoff)
                .order_by(SentimentRecord.timestamp.desc())
                .all()
            )

            if not records:
                logger.warning("No sentiment records found in the last 48 h; CSS = 0.0")
                return 0.0

            weighted_sum = 0.0
            weight_sum = 0.0

            for rec in records:
                age_hours = max((now - rec.timestamp).total_seconds() / 3600.0, 0.0)
                decay = math.exp(-settings.RECENCY_DECAY_LAMBDA * age_hours)
                source_weight = settings.SENTIMENT_WEIGHTS.get(rec.source, 0.0)
                if source_weight == 0.0:
                    continue
                weighted_sum += rec.sentiment_score * source_weight * decay
                weight_sum += source_weight

            if weight_sum == 0.0:
                return 0.0

            css = weighted_sum / weight_sum
            css = max(-1.0, min(1.0, css))
            logger.debug("CSS computed: {:.4f} from {} records", css, len(records))
            return css
        finally:
            session.close()

    # ------------------------------------------------------------------
    # CSS momentum
    # ------------------------------------------------------------------
    def compute_css_momentum(self, now: Optional[datetime] = None) -> float:
        """Compute CSS momentum: delta_CSS = CSS_t - CSS_(t - 1h).

        Parameters
        ----------
        now : datetime, optional
            Reference time.  Defaults to utcnow().

        Returns
        -------
        float
            Change in CSS over the last hour.
        """
        if now is None:
            now = datetime.utcnow()

        css_now = self.compute_css(now=now)
        css_prev = self.compute_css(now=now - timedelta(hours=1))
        momentum = css_now - css_prev
        logger.debug("CSS momentum: {:.4f} (now={:.4f}, prev={:.4f})", momentum, css_now, css_prev)
        return momentum

    # ------------------------------------------------------------------
    # Aggregated signal
    # ------------------------------------------------------------------
    def get_sentiment_signal(self, now: Optional[datetime] = None) -> Dict[str, float]:
        """Return a full sentiment signal and persist a CompositeSentiment record.

        Returns
        -------
        dict
            css_score  : float in [-1, 1]
            css_momentum : float
            css_volume : int  (number of records in the 48 h window)
        """
        if now is None:
            now = datetime.utcnow()

        css_score = self.compute_css(now=now)
        css_momentum = self.compute_css_momentum(now=now)

        # Count records in the window
        cutoff = now - timedelta(hours=48)
        session = get_session()
        try:
            css_volume: int = (
                session.query(SentimentRecord)
                .filter(SentimentRecord.timestamp >= cutoff)
                .count()
            )

            # Persist composite record
            record = CompositeSentiment(
                timestamp=now,
                css_score=css_score,
                css_momentum=css_momentum,
                css_volume=css_volume,
            )
            session.add(record)
            session.commit()
            logger.info(
                "Sentiment signal: css={:.4f}, momentum={:.4f}, volume={}",
                css_score,
                css_momentum,
                css_volume,
            )
        except Exception:
            session.rollback()
            logger.exception("Failed to persist CompositeSentiment record")
        finally:
            session.close()

        return {
            "css_score": css_score,
            "css_momentum": css_momentum,
            "css_volume": css_volume,
        }
