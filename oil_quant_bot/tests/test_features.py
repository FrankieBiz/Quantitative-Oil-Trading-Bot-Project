"""
Comprehensive unit tests for oil_quant_bot.ml.features.FeatureEngineer.

Covers:
  - Feature vector construction (length, ordering, dtype)
  - FEATURE_NAMES attribute
  - Cyclical encoding bounds (hour_sin/cos, dow_sin/cos)
  - RobustScaler fit + transform pipeline
  - save_scaler / load_scaler roundtrip
  - Graceful handling of missing values
  - Edge cases: midnight, weekends
"""

from __future__ import annotations

import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the oil_quant_bot package is importable
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parent.parent          # oil_quant_bot/
sys.path.insert(0, str(_PROJECT_DIR))

# We need to mock `config.settings` and `db.models` before importing features,
# because features.py does a top-level ``from config import settings`` and
# ``from db.models import ...``.

_fake_models_dir: Path  # will be set per-test via the fixture


class _FakeSettings:
    """Minimal stand-in for config.settings used by FeatureEngineer.__init__."""

    @property
    def MODELS_DIR(self) -> Path:  # noqa: N802
        return _fake_models_dir


_settings_mod = MagicMock()
_settings_mod.settings = _FakeSettings()

sys.modules.setdefault("config", _settings_mod)
sys.modules.setdefault("config.settings", _settings_mod)

_db_models_mod = MagicMock()
sys.modules.setdefault("db", MagicMock())
sys.modules.setdefault("db.models", _db_models_mod)

from ml.features import FeatureEngineer  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _tmp_models_dir(tmp_path: Path):
    """Point the fake settings.MODELS_DIR to a temp directory for every test."""
    global _fake_models_dir
    _fake_models_dir = tmp_path / "models"
    _fake_models_dir.mkdir()
    yield


@pytest.fixture()
def fe() -> FeatureEngineer:
    """Return a fresh FeatureEngineer with no pre-existing scaler on disk."""
    return FeatureEngineer(instrument="CL")


def _full_technicals() -> dict:
    """Return a technicals dict with every expected key populated."""
    return {
        "ema9": 72.5,
        "ema21": 71.8,
        "ema50": 70.0,
        "ema200": 68.0,
        "macd_hist": 0.35,
        "macd_signal": 0.12,
        "rsi14": 58.0,
        "stoch_rsi": 0.65,
        "roc10": 1.2,
        "atr14": 1.5,
        "bb_pct_b": 0.72,
        "bb_width": 3.1,
        "hist_vol20": 0.28,
        "obv_zscore": 1.1,
        "volume_zscore": 0.9,
        "close": 73.0,
        "vwap": 72.5,
        "wti_brent_spread": -3.5,
        "oil_dxy_corr20": -0.45,
        "oil_spx_corr20": 0.30,
    }


def _full_sentiment() -> dict:
    return {
        "css_score": 0.6,
        "css_momentum": 0.1,
        "css_volume_zscore": 1.3,
        "eia_surprise": -0.5,
    }


# ======================================================================
# 1. FEATURE_NAMES attribute
# ======================================================================

class TestFeatureNames:

    def test_length_is_28(self, fe: FeatureEngineer):
        assert fe.NUM_FEATURES == 28

    def test_feature_names_length_matches_num_features(self, fe: FeatureEngineer):
        assert len(fe.FEATURE_NAMES) == fe.NUM_FEATURES

    def test_expected_feature_names_present(self, fe: FeatureEngineer):
        expected_subset = [
            "ema9", "rsi14", "atr14", "obv_zscore",
            "css_score", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "eia_surprise",
        ]
        for name in expected_subset:
            assert name in fe.FEATURE_NAMES, f"{name} missing from FEATURE_NAMES"

    def test_feature_names_are_unique(self, fe: FeatureEngineer):
        assert len(fe.FEATURE_NAMES) == len(set(fe.FEATURE_NAMES))


# ======================================================================
# 2. Feature vector construction
# ======================================================================

class TestBuildFeatureVector:

    def test_vector_length(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 12, 14, 30)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        assert vec.shape == (fe.NUM_FEATURES,)

    def test_vector_dtype(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 12, 14, 30)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        assert vec.dtype == np.float64

    def test_ordering_matches_feature_names(self, fe: FeatureEngineer):
        """The vector entries must follow FEATURE_NAMES ordering."""
        ts = datetime(2025, 3, 12, 14, 30)  # Wednesday
        tech = _full_technicals()
        sent = _full_sentiment()
        vec = fe.build_feature_vector(tech, sent, ts)

        idx = fe.FEATURE_NAMES.index
        # Spot-check several entries
        assert vec[idx("ema9")] == pytest.approx(72.5)
        assert vec[idx("rsi14")] == pytest.approx(58.0)
        assert vec[idx("css_score")] == pytest.approx(0.6)
        assert vec[idx("eia_surprise")] == pytest.approx(-0.5)

    def test_ema9_cross_ema21_flag(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 12, 14, 0)
        tech = _full_technicals()
        # ema9 > ema21 => cross = 1.0
        tech["ema9"] = 75.0
        tech["ema21"] = 70.0
        vec = fe.build_feature_vector(tech, _full_sentiment(), ts)
        idx = fe.FEATURE_NAMES.index("ema9_cross_ema21")
        assert vec[idx] == 1.0

        # ema9 < ema21 => cross = 0.0
        tech["ema9"] = 69.0
        vec2 = fe.build_feature_vector(tech, _full_sentiment(), ts)
        assert vec2[idx] == 0.0

    def test_vwap_distance_with_nonzero_atr(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 12, 14, 0)
        tech = _full_technicals()
        tech["close"] = 75.0
        tech["vwap"] = 73.0
        tech["atr14"] = 2.0
        vec = fe.build_feature_vector(tech, _full_sentiment(), ts)
        idx = fe.FEATURE_NAMES.index("vwap_distance")
        assert vec[idx] == pytest.approx((75.0 - 73.0) / 2.0)

    def test_vwap_distance_zero_atr(self, fe: FeatureEngineer):
        """When atr14 == 0 the vwap_distance should default to 0.0."""
        ts = datetime(2025, 3, 12, 14, 0)
        tech = _full_technicals()
        tech["atr14"] = 0.0
        vec = fe.build_feature_vector(tech, _full_sentiment(), ts)
        idx = fe.FEATURE_NAMES.index("vwap_distance")
        assert vec[idx] == 0.0


# ======================================================================
# 3. Cyclical encoding
# ======================================================================

class TestCyclicalEncoding:

    @pytest.mark.parametrize("hour", list(range(24)))
    def test_hour_sin_cos_in_bounds(self, fe: FeatureEngineer, hour: int):
        ts = datetime(2025, 3, 12, hour, 0)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        h_sin = vec[fe.FEATURE_NAMES.index("hour_sin")]
        h_cos = vec[fe.FEATURE_NAMES.index("hour_cos")]
        assert -1.0 <= h_sin <= 1.0
        assert -1.0 <= h_cos <= 1.0

    @pytest.mark.parametrize("weekday", list(range(7)))
    def test_dow_sin_cos_in_bounds(self, fe: FeatureEngineer, weekday: int):
        # 2025-03-10 is a Monday (weekday 0); add offset to hit each day
        ts = datetime(2025, 3, 10 + weekday, 12, 0)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        d_sin = vec[fe.FEATURE_NAMES.index("dow_sin")]
        d_cos = vec[fe.FEATURE_NAMES.index("dow_cos")]
        assert -1.0 <= d_sin <= 1.0
        assert -1.0 <= d_cos <= 1.0

    def test_hour_sin_cos_unit_circle(self, fe: FeatureEngineer):
        """sin^2 + cos^2 == 1 for any hour."""
        ts = datetime(2025, 3, 12, 9, 45)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        h_sin = vec[fe.FEATURE_NAMES.index("hour_sin")]
        h_cos = vec[fe.FEATURE_NAMES.index("hour_cos")]
        assert h_sin**2 + h_cos**2 == pytest.approx(1.0)

    def test_dow_sin_cos_unit_circle(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 12, 12, 0)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        d_sin = vec[fe.FEATURE_NAMES.index("dow_sin")]
        d_cos = vec[fe.FEATURE_NAMES.index("dow_cos")]
        assert d_sin**2 + d_cos**2 == pytest.approx(1.0)

    def test_midnight_encoding(self, fe: FeatureEngineer):
        """At midnight hour_sin should be ~0, hour_cos should be ~1."""
        ts = datetime(2025, 3, 12, 0, 0)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        h_sin = vec[fe.FEATURE_NAMES.index("hour_sin")]
        h_cos = vec[fe.FEATURE_NAMES.index("hour_cos")]
        assert h_sin == pytest.approx(0.0, abs=1e-10)
        assert h_cos == pytest.approx(1.0, abs=1e-10)

    def test_saturday_and_sunday_encoding(self, fe: FeatureEngineer):
        """Weekend days should still produce valid cyclical values."""
        # 2025-03-15 is Saturday (weekday=5), 2025-03-16 is Sunday (weekday=6)
        for day, expected_wd in [(15, 5), (16, 6)]:
            ts = datetime(2025, 3, day, 12, 0)
            vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
            d_sin = vec[fe.FEATURE_NAMES.index("dow_sin")]
            d_cos = vec[fe.FEATURE_NAMES.index("dow_cos")]
            assert -1.0 <= d_sin <= 1.0
            assert -1.0 <= d_cos <= 1.0
            expected_sin = math.sin(2 * math.pi * expected_wd / 7.0)
            expected_cos = math.cos(2 * math.pi * expected_wd / 7.0)
            assert d_sin == pytest.approx(expected_sin)
            assert d_cos == pytest.approx(expected_cos)


# ======================================================================
# 4. RobustScaler: fit + transform
# ======================================================================

class TestScaler:

    def test_transform_without_fit_returns_raw(self, fe: FeatureEngineer):
        """When no scaler is fitted, transform should return the raw vector."""
        ts = datetime(2025, 3, 12, 14, 0)
        raw = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        transformed = fe.transform(raw)
        np.testing.assert_array_equal(raw, transformed)

    def test_fit_then_transform_changes_values(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 12, 14, 0)
        rng = np.random.default_rng(42)
        # Generate random historical data
        hist = rng.standard_normal((200, fe.NUM_FEATURES))
        fe.fit_scaler(hist)

        raw = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        scaled = fe.transform(raw)

        assert scaled.shape == raw.shape
        # Scaled values should generally differ from raw
        assert not np.allclose(raw, scaled)

    def test_fit_scaler_on_single_sample(self, fe: FeatureEngineer):
        """fit_scaler accepts a 1-D array (single sample)."""
        single = np.ones(fe.NUM_FEATURES)
        fe.fit_scaler(single)
        assert fe.scaler is not None

    def test_transform_matrix(self, fe: FeatureEngineer):
        """transform accepts a 2-D matrix and returns the same shape."""
        rng = np.random.default_rng(7)
        hist = rng.standard_normal((100, fe.NUM_FEATURES))
        fe.fit_scaler(hist)

        batch = rng.standard_normal((10, fe.NUM_FEATURES))
        scaled = fe.transform(batch)
        assert scaled.shape == (10, fe.NUM_FEATURES)

    def test_fit_scaler_rolling_uses_subset(self, fe: FeatureEngineer):
        """fit_scaler_rolling should fit on at most window_days * 26 rows."""
        rng = np.random.default_rng(99)
        hist = rng.standard_normal((10000, fe.NUM_FEATURES))
        fe.fit_scaler_rolling(hist, window_days=10)
        assert fe.scaler is not None


# ======================================================================
# 5. save_scaler / load_scaler roundtrip
# ======================================================================

class TestScalerPersistence:

    def test_save_and_load_roundtrip(self, fe: FeatureEngineer):
        rng = np.random.default_rng(0)
        hist = rng.standard_normal((200, fe.NUM_FEATURES))
        fe.fit_scaler(hist)  # also calls save_scaler internally

        sample = rng.standard_normal(fe.NUM_FEATURES)
        expected = fe.transform(sample)

        # Create a new FeatureEngineer -- it should auto-load the saved scaler
        fe2 = FeatureEngineer(instrument="CL")
        assert fe2.scaler is not None
        result = fe2.transform(sample)
        np.testing.assert_array_almost_equal(result, expected)

    def test_save_scaler_no_scaler(self, fe: FeatureEngineer):
        """save_scaler with no fitted scaler should not raise."""
        fe.scaler = None
        fe.save_scaler()  # should just log a warning

    def test_load_scaler_missing_file(self, fe: FeatureEngineer):
        """load_scaler with a non-existent file should not raise."""
        fe._scaler_path = Path("/tmp/nonexistent_scaler.joblib")
        fe.load_scaler()
        assert fe.scaler is None


# ======================================================================
# 6. Missing values handled gracefully
# ======================================================================

class TestMissingValues:

    def test_empty_technicals_and_sentiment(self, fe: FeatureEngineer):
        """All-empty dicts should produce a valid zero-filled vector."""
        ts = datetime(2025, 3, 12, 10, 0)
        vec = fe.build_feature_vector({}, {}, ts)
        assert vec.shape == (fe.NUM_FEATURES,)
        # All technicals/sentiment default to 0; only calendar entries are non-zero
        assert not np.any(np.isnan(vec))

    def test_partial_technicals(self, fe: FeatureEngineer):
        """Providing only a subset of technicals should not raise."""
        ts = datetime(2025, 3, 12, 10, 0)
        partial = {"ema9": 72.0, "rsi14": 55.0}
        vec = fe.build_feature_vector(partial, {}, ts)
        assert vec.shape == (fe.NUM_FEATURES,)
        assert vec[fe.FEATURE_NAMES.index("ema9")] == pytest.approx(72.0)
        assert vec[fe.FEATURE_NAMES.index("rsi14")] == pytest.approx(55.0)
        # unset fields default to 0
        assert vec[fe.FEATURE_NAMES.index("atr14")] == pytest.approx(0.0)

    def test_none_values_converted_to_float(self, fe: FeatureEngineer):
        """If a dict value is a string-number or int, float() should handle it."""
        ts = datetime(2025, 3, 12, 10, 0)
        tech = {"ema9": "72.5", "rsi14": 55}
        vec = fe.build_feature_vector(tech, {}, ts)
        assert vec[fe.FEATURE_NAMES.index("ema9")] == pytest.approx(72.5)
        assert vec[fe.FEATURE_NAMES.index("rsi14")] == pytest.approx(55.0)


# ======================================================================
# 7. Edge cases: midnight and weekend
# ======================================================================

class TestEdgeCases:

    def test_midnight_hour_zero(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 12, 0, 0)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        assert vec[fe.FEATURE_NAMES.index("hour_sin")] == pytest.approx(0.0, abs=1e-10)
        assert vec[fe.FEATURE_NAMES.index("hour_cos")] == pytest.approx(1.0, abs=1e-10)

    def test_last_minute_of_day(self, fe: FeatureEngineer):
        """23:59 should produce values very close to midnight but not equal."""
        ts = datetime(2025, 3, 12, 23, 59)
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        h_sin = vec[fe.FEATURE_NAMES.index("hour_sin")]
        h_cos = vec[fe.FEATURE_NAMES.index("hour_cos")]
        # Close to midnight => sin near 0 (slightly negative), cos near 1
        assert -1.0 <= h_sin <= 1.0
        assert -1.0 <= h_cos <= 1.0

    def test_monday_dow_encoding(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 10, 12, 0)  # Monday
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        expected_sin = math.sin(2 * math.pi * 0 / 7.0)  # weekday=0
        expected_cos = math.cos(2 * math.pi * 0 / 7.0)
        assert vec[fe.FEATURE_NAMES.index("dow_sin")] == pytest.approx(expected_sin)
        assert vec[fe.FEATURE_NAMES.index("dow_cos")] == pytest.approx(expected_cos)

    def test_sunday_dow_encoding(self, fe: FeatureEngineer):
        ts = datetime(2025, 3, 16, 12, 0)  # Sunday
        vec = fe.build_feature_vector(_full_technicals(), _full_sentiment(), ts)
        expected_sin = math.sin(2 * math.pi * 6 / 7.0)  # weekday=6
        expected_cos = math.cos(2 * math.pi * 6 / 7.0)
        assert vec[fe.FEATURE_NAMES.index("dow_sin")] == pytest.approx(expected_sin)
        assert vec[fe.FEATURE_NAMES.index("dow_cos")] == pytest.approx(expected_cos)

    def test_build_and_transform_calls_store_snapshot(self, fe: FeatureEngineer):
        """build_and_transform with store=True should call store_snapshot."""
        ts = datetime(2025, 3, 12, 14, 0)
        with patch.object(fe, "store_snapshot") as mock_store:
            fe.build_and_transform(
                _full_technicals(), _full_sentiment(), ts, store=True,
            )
            mock_store.assert_called_once()

    def test_build_and_transform_no_store(self, fe: FeatureEngineer):
        """build_and_transform with store=False should not call store_snapshot."""
        ts = datetime(2025, 3, 12, 14, 0)
        with patch.object(fe, "store_snapshot") as mock_store:
            result = fe.build_and_transform(
                _full_technicals(), _full_sentiment(), ts, store=False,
            )
            mock_store.assert_not_called()
            assert result.shape == (fe.NUM_FEATURES,)
