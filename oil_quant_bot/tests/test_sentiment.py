"""
Comprehensive unit tests for the SentimentEngine class.

Mocks the FinBERT pipeline and database sessions so tests are fully
self-contained and require no model downloads or live database.
"""

import math
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup – make oil_quant_bot importable
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/user/Quantitative-Oil-Trading-Bot-Project/oil_quant_bot")


# ---------------------------------------------------------------------------
# Helpers: fake FinBERT pipeline output
# ---------------------------------------------------------------------------
def _finbert_output(positive: float, negative: float, neutral: float):
    """Build a single-text FinBERT pipeline output (list of label dicts)."""
    return [
        {"label": "positive", "score": positive},
        {"label": "negative", "score": negative},
        {"label": "neutral", "score": neutral},
    ]


def _make_pipeline_fn(positive=0.7, negative=0.1, neutral=0.2):
    """Return a callable that mimics the transformers sentiment pipeline.

    For a single text the real pipeline returns [[{...}, ...]], for a batch
    it returns [[ ... ], [ ... ], ...].  The SentimentEngine calls it as:
        self._pipeline(text)[0]          # single text
        self._pipeline(batch)            # batch
    So this mock returns a list-of-lists.
    """

    def _fake_pipeline(input_text, **kwargs):
        if isinstance(input_text, list):
            return [_finbert_output(positive, negative, neutral)] * len(input_text)
        # single string
        return [_finbert_output(positive, negative, neutral)]

    return _fake_pipeline


# ---------------------------------------------------------------------------
# Fake SentimentRecord
# ---------------------------------------------------------------------------
class FakeSentimentRecord:
    """Lightweight stand-in for db.models.SentimentRecord."""

    def __init__(self, source: str, sentiment_score: float, timestamp: datetime):
        self.source = source
        self.sentiment_score = sentiment_score
        self.timestamp = timestamp


# ---------------------------------------------------------------------------
# Fixture: build a SentimentEngine with mocked pipeline + settings
# ---------------------------------------------------------------------------
@pytest.fixture()
def engine():
    """Create a SentimentEngine with the FinBERT pipeline mocked out."""
    with patch("signals.sentiment.pipeline", return_value=_make_pipeline_fn()), \
         patch("signals.sentiment.settings") as mock_settings:
        mock_settings.FINBERT_MODEL = "mock-model"
        mock_settings.FINBERT_MAX_LENGTH = 512
        mock_settings.RECENCY_DECAY_LAMBDA = 0.1
        mock_settings.SENTIMENT_WEIGHTS = {
            "twitter": 0.20,
            "news_major": 0.45,
            "news_minor": 0.20,
            "eia": 0.15,
        }

        from signals.sentiment import SentimentEngine
        eng = SentimentEngine()
        # Replace the internal pipeline with a controlled mock
        eng._pipeline = _make_pipeline_fn(positive=0.7, negative=0.1, neutral=0.2)
    return eng


# ===================================================================
# 1. Sentiment score calculation: positive - negative in [-1, 1]
# ===================================================================
class TestAnalyzeText:
    """Tests for SentimentEngine.analyze_text."""

    def test_score_is_positive_minus_negative(self, engine):
        engine._pipeline = _make_pipeline_fn(positive=0.8, negative=0.1, neutral=0.1)
        result = engine.analyze_text("Oil prices surge")
        assert result["sentiment_score"] == pytest.approx(0.7)

    def test_score_range_positive(self, engine):
        engine._pipeline = _make_pipeline_fn(positive=1.0, negative=0.0, neutral=0.0)
        result = engine.analyze_text("Market is booming")
        assert result["sentiment_score"] == pytest.approx(1.0)
        assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_score_range_negative(self, engine):
        engine._pipeline = _make_pipeline_fn(positive=0.0, negative=1.0, neutral=0.0)
        result = engine.analyze_text("Market crashes badly")
        assert result["sentiment_score"] == pytest.approx(-1.0)
        assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_score_neutral(self, engine):
        engine._pipeline = _make_pipeline_fn(positive=0.3, negative=0.3, neutral=0.4)
        result = engine.analyze_text("Prices stayed flat")
        assert result["sentiment_score"] == pytest.approx(0.0)

    def test_returns_all_keys(self, engine):
        result = engine.analyze_text("test")
        assert set(result.keys()) == {"positive", "negative", "neutral", "sentiment_score"}

    def test_empty_string_returns_neutral(self, engine):
        result = engine.analyze_text("")
        assert result == {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "sentiment_score": 0.0}

    def test_whitespace_only_returns_neutral(self, engine):
        result = engine.analyze_text("   ")
        assert result["sentiment_score"] == 0.0
        assert result["neutral"] == 1.0

    def test_none_text_returns_neutral(self, engine):
        result = engine.analyze_text(None)
        assert result["sentiment_score"] == 0.0


# ===================================================================
# 2. Batch analysis
# ===================================================================
class TestAnalyzeBatch:
    def test_batch_returns_correct_count(self, engine):
        texts = ["text1", "text2", "text3"]
        results = engine.analyze_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "sentiment_score" in r

    def test_batch_empty_list(self, engine):
        assert engine.analyze_batch([]) == []

    def test_batch_respects_batch_size(self, engine):
        """Verify batching produces correct results regardless of batch_size."""
        texts = ["a", "b", "c", "d", "e"]
        results = engine.analyze_batch(texts, batch_size=2)
        assert len(results) == 5


# ===================================================================
# 3. Recency decay: e^(-0.1 * age_in_hours)
# ===================================================================
class TestRecencyDecay:
    """Verify the exponential decay applied during CSS computation."""

    def _run_css_with_records(self, engine, records, now):
        """Helper: patch get_session to feed fake records into compute_css."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = records

        with patch("signals.sentiment.get_session", return_value=mock_session), \
             patch("signals.sentiment.settings") as s:
            s.RECENCY_DECAY_LAMBDA = 0.1
            s.SENTIMENT_WEIGHTS = {
                "twitter": 0.20,
                "news_major": 0.45,
                "news_minor": 0.20,
                "eia": 0.15,
            }
            return engine.compute_css(now=now)

    def test_newer_items_decay_less(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        rec_new = FakeSentimentRecord("news_major", 0.5, now - timedelta(hours=1))
        rec_old = FakeSentimentRecord("news_major", 0.5, now - timedelta(hours=10))

        # CSS with only the new record
        css_new = self._run_css_with_records(engine, [rec_new], now)
        # CSS with only the old record
        css_old = self._run_css_with_records(engine, [rec_old], now)

        # Newer record should have higher effective contribution (less decay)
        assert abs(css_new) > abs(css_old)

    def test_zero_age_has_no_decay(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        rec = FakeSentimentRecord("news_major", 0.8, now)  # age = 0
        css = self._run_css_with_records(engine, [rec], now)
        # decay = e^0 = 1.0, so css = score * weight * 1 / weight = score
        assert css == pytest.approx(0.8)

    def test_decay_formula_exact(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        age_hours = 5.0
        rec = FakeSentimentRecord("news_major", 1.0, now - timedelta(hours=age_hours))
        css = self._run_css_with_records(engine, [rec], now)

        expected_decay = math.exp(-0.1 * age_hours)
        # single record: css = score * weight * decay / weight = score * decay
        assert css == pytest.approx(1.0 * expected_decay)

    def test_decay_24h_old(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        rec = FakeSentimentRecord("eia", 0.6, now - timedelta(hours=24))
        css = self._run_css_with_records(engine, [rec], now)
        expected = 0.6 * math.exp(-0.1 * 24)
        assert css == pytest.approx(expected)


# ===================================================================
# 4. CSS computation: weighted average with source weights
# ===================================================================
class TestComputeCSS:
    """Verify CSS = sum(score_i * weight_i * decay_i) / sum(weight_i)."""

    def _css_with_records(self, engine, records, now):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = records

        with patch("signals.sentiment.get_session", return_value=mock_session), \
             patch("signals.sentiment.settings") as s:
            s.RECENCY_DECAY_LAMBDA = 0.1
            s.SENTIMENT_WEIGHTS = {
                "twitter": 0.20,
                "news_major": 0.45,
                "news_minor": 0.20,
                "eia": 0.15,
            }
            return engine.compute_css(now=now)

    def test_weighted_average_multiple_sources(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        records = [
            FakeSentimentRecord("twitter", 0.5, now),
            FakeSentimentRecord("news_major", 0.8, now),
            FakeSentimentRecord("news_minor", -0.3, now),
            FakeSentimentRecord("eia", 0.2, now),
        ]
        css = self._css_with_records(engine, records, now)

        # All at time=now so decay=1
        weights = {"twitter": 0.20, "news_major": 0.45, "news_minor": 0.20, "eia": 0.15}
        expected_num = (
            0.5 * weights["twitter"]
            + 0.8 * weights["news_major"]
            + (-0.3) * weights["news_minor"]
            + 0.2 * weights["eia"]
        )
        expected_den = sum(weights.values())
        expected_css = expected_num / expected_den
        assert css == pytest.approx(expected_css)

    def test_css_clamped_to_range(self, engine):
        """Even with extreme scores CSS should stay in [-1, 1]."""
        now = datetime(2025, 6, 1, 12, 0, 0)
        records = [FakeSentimentRecord("news_major", 1.0, now)]
        css = self._css_with_records(engine, records, now)
        assert -1.0 <= css <= 1.0

    def test_css_negative_score(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        records = [FakeSentimentRecord("news_major", -0.9, now)]
        css = self._css_with_records(engine, records, now)
        assert css == pytest.approx(-0.9)
        assert css < 0


# ===================================================================
# 5. CSS momentum: delta_CSS = CSS_t - CSS_(t-1h)
# ===================================================================
class TestCSSMomentum:

    def test_momentum_positive_when_improving(self, engine):
        """If current CSS > previous CSS, momentum should be positive."""
        with patch.object(engine, "compute_css", side_effect=[0.5, 0.2]):
            momentum = engine.compute_css_momentum(now=datetime(2025, 6, 1, 12, 0, 0))
        assert momentum == pytest.approx(0.3)

    def test_momentum_negative_when_declining(self, engine):
        with patch.object(engine, "compute_css", side_effect=[0.1, 0.6]):
            momentum = engine.compute_css_momentum(now=datetime(2025, 6, 1, 12, 0, 0))
        assert momentum == pytest.approx(-0.5)

    def test_momentum_zero_when_stable(self, engine):
        with patch.object(engine, "compute_css", side_effect=[0.4, 0.4]):
            momentum = engine.compute_css_momentum(now=datetime(2025, 6, 1, 12, 0, 0))
        assert momentum == pytest.approx(0.0)

    def test_momentum_calls_css_with_correct_times(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        with patch.object(engine, "compute_css", return_value=0.0) as mock_css:
            engine.compute_css_momentum(now=now)
        calls = mock_css.call_args_list
        assert calls[0].kwargs["now"] == now
        assert calls[1].kwargs["now"] == now - timedelta(hours=1)


# ===================================================================
# 6. CSS volume count
# ===================================================================
class TestCSSVolume:

    def test_volume_counts_records(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.count.return_value = 42

        with patch.object(engine, "compute_css", return_value=0.3), \
             patch.object(engine, "compute_css_momentum", return_value=0.05), \
             patch("signals.sentiment.get_session", return_value=mock_session), \
             patch("signals.sentiment.settings") as s:
            s.SENTIMENT_WEIGHTS = {"twitter": 0.20, "news_major": 0.45, "news_minor": 0.20, "eia": 0.15}
            s.RECENCY_DECAY_LAMBDA = 0.1
            result = engine.get_sentiment_signal(now=now)

        assert result["css_volume"] == 42

    def test_volume_zero_no_records(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.count.return_value = 0

        with patch.object(engine, "compute_css", return_value=0.0), \
             patch.object(engine, "compute_css_momentum", return_value=0.0), \
             patch("signals.sentiment.get_session", return_value=mock_session), \
             patch("signals.sentiment.settings") as s:
            s.SENTIMENT_WEIGHTS = {"twitter": 0.20, "news_major": 0.45, "news_minor": 0.20, "eia": 0.15}
            s.RECENCY_DECAY_LAMBDA = 0.1
            result = engine.get_sentiment_signal(now=now)

        assert result["css_volume"] == 0


# ===================================================================
# 7. Source weight mapping
# ===================================================================
class TestSourceWeights:

    def _css_with_records(self, engine, records, now):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = records

        with patch("signals.sentiment.get_session", return_value=mock_session), \
             patch("signals.sentiment.settings") as s:
            s.RECENCY_DECAY_LAMBDA = 0.1
            s.SENTIMENT_WEIGHTS = {
                "twitter": 0.20,
                "news_major": 0.45,
                "news_minor": 0.20,
                "eia": 0.15,
            }
            return engine.compute_css(now=now)

    def test_twitter_weight_is_0_20(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        rec = FakeSentimentRecord("twitter", 1.0, now)
        css = self._css_with_records(engine, [rec], now)
        # single source: css = score * w * decay / w = score * decay = 1.0
        assert css == pytest.approx(1.0)

    def test_news_major_weight_dominates(self, engine):
        """news_major (0.45) should have a stronger pull than twitter (0.20)."""
        now = datetime(2025, 6, 1, 12, 0, 0)
        records = [
            FakeSentimentRecord("twitter", -1.0, now),
            FakeSentimentRecord("news_major", 1.0, now),
        ]
        css = self._css_with_records(engine, records, now)
        # css = (-1*0.20 + 1*0.45) / (0.20+0.45) = 0.25/0.65
        expected = 0.25 / 0.65
        assert css == pytest.approx(expected)
        assert css > 0  # news_major dominates

    def test_unknown_source_ignored(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        records = [
            FakeSentimentRecord("unknown_source", 1.0, now),
        ]
        css = self._css_with_records(engine, records, now)
        assert css == pytest.approx(0.0)

    def test_eia_weight_is_0_15(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        records = [
            FakeSentimentRecord("eia", 1.0, now),
            FakeSentimentRecord("twitter", 1.0, now),
        ]
        css = self._css_with_records(engine, records, now)
        # both score=1, decay=1: css = (1*0.15 + 1*0.20)/(0.15+0.20) = 1.0
        assert css == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        weights = {"twitter": 0.20, "news_major": 0.45, "news_minor": 0.20, "eia": 0.15}
        assert sum(weights.values()) == pytest.approx(1.0)


# ===================================================================
# 8. Edge cases
# ===================================================================
class TestEdgeCases:

    def _css_with_records(self, engine, records, now):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = records

        with patch("signals.sentiment.get_session", return_value=mock_session), \
             patch("signals.sentiment.settings") as s:
            s.RECENCY_DECAY_LAMBDA = 0.1
            s.SENTIMENT_WEIGHTS = {
                "twitter": 0.20,
                "news_major": 0.45,
                "news_minor": 0.20,
                "eia": 0.15,
            }
            return engine.compute_css(now=now)

    def test_empty_records_returns_zero(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        css = self._css_with_records(engine, [], now)
        assert css == 0.0

    def test_single_record(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        rec = FakeSentimentRecord("news_minor", 0.6, now)
        css = self._css_with_records(engine, [rec], now)
        assert css == pytest.approx(0.6)

    def test_all_same_source(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        records = [
            FakeSentimentRecord("eia", 0.4, now),
            FakeSentimentRecord("eia", 0.8, now),
            FakeSentimentRecord("eia", -0.2, now),
        ]
        css = self._css_with_records(engine, records, now)
        # All same weight, all decay=1: css = (0.4+0.8-0.2)*w / (3*w) = 1.0/3
        expected = (0.4 * 0.15 + 0.8 * 0.15 + (-0.2) * 0.15) / (3 * 0.15)
        assert css == pytest.approx(expected)

    def test_all_same_source_same_score(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        records = [FakeSentimentRecord("twitter", 0.5, now)] * 5
        css = self._css_with_records(engine, records, now)
        assert css == pytest.approx(0.5)

    def test_mixed_ages_weighted_correctly(self, engine):
        """Two records of different ages should produce the decay-weighted average."""
        now = datetime(2025, 6, 1, 12, 0, 0)
        rec1 = FakeSentimentRecord("news_major", 1.0, now)                        # age=0h
        rec2 = FakeSentimentRecord("news_major", -1.0, now - timedelta(hours=10))  # age=10h
        css = self._css_with_records(engine, [rec1, rec2], now)

        d1 = math.exp(-0.1 * 0)
        d2 = math.exp(-0.1 * 10)
        w = 0.45
        expected = (1.0 * w * d1 + (-1.0) * w * d2) / (2 * w)
        assert css == pytest.approx(expected)


# ===================================================================
# 9. get_sentiment_signal integration
# ===================================================================
class TestGetSentimentSignal:

    def test_returns_expected_keys(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 10

        with patch.object(engine, "compute_css", return_value=0.3), \
             patch.object(engine, "compute_css_momentum", return_value=0.05), \
             patch("signals.sentiment.get_session", return_value=mock_session), \
             patch("signals.sentiment.settings") as s:
            s.SENTIMENT_WEIGHTS = {"twitter": 0.20, "news_major": 0.45, "news_minor": 0.20, "eia": 0.15}
            s.RECENCY_DECAY_LAMBDA = 0.1
            result = engine.get_sentiment_signal(now=now)

        assert set(result.keys()) == {"css_score", "css_momentum", "css_volume"}
        assert result["css_score"] == pytest.approx(0.3)
        assert result["css_momentum"] == pytest.approx(0.05)
        assert result["css_volume"] == 10

    def test_persists_composite_record(self, engine):
        now = datetime(2025, 6, 1, 12, 0, 0)
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 5

        with patch.object(engine, "compute_css", return_value=0.1), \
             patch.object(engine, "compute_css_momentum", return_value=-0.02), \
             patch("signals.sentiment.get_session", return_value=mock_session), \
             patch("signals.sentiment.settings") as s:
            s.SENTIMENT_WEIGHTS = {"twitter": 0.20, "news_major": 0.45, "news_minor": 0.20, "eia": 0.15}
            s.RECENCY_DECAY_LAMBDA = 0.1
            engine.get_sentiment_signal(now=now)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
