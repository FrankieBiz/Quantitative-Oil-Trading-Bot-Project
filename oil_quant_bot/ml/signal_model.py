"""
XGBoost 3-class signal model for the Oil Quantitative Trading Bot.

Predicts the next-4-bar directional move (LONG / FLAT / SHORT) and outputs
calibrated class probabilities.  Supports walk-forward validation, versioned
model persistence, incremental warm-start updates, and full metric logging.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from config import settings
from db.models import ModelVersion, get_session
from ml.features import FeatureEngineer


# Label mapping: 0 → short (-1), 1 → flat (0), 2 → long (+1)
_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
_INV_LABEL_MAP = {v: k for k, v in _LABEL_MAP.items()}
_CLASS_NAMES = ["short", "flat", "long"]


class SignalModel:
    """XGBoost 3-class classifier that drives directional trading signals."""

    def __init__(self, instrument: str = "CL") -> None:
        self.instrument = instrument
        self._model_dir = settings.MODELS_DIR / f"xgb_{instrument}"
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[XGBClassifier] = None
        self._version: Optional[str] = None

    # -------------------------------------------------------------- labels
    @staticmethod
    def _compute_labels(close_series: pd.Series, forward_bars: int = None) -> np.ndarray:
        """Compute target labels from a close-price series.

        y =  1  if return(t + forward_bars) >  +0.4 %
        y = -1  if return(t + forward_bars) <  -0.4 %
        y =  0  otherwise

        Returns mapped labels suitable for XGBoost (0, 1, 2).
        """
        if forward_bars is None:
            forward_bars = settings.FORWARD_BARS

        fwd_return = close_series.shift(-forward_bars) / close_series - 1.0
        raw_labels = np.where(
            fwd_return > settings.LONG_THRESHOLD,
            1,
            np.where(fwd_return < settings.SHORT_THRESHOLD, -1, 0),
        )
        mapped = np.array([_LABEL_MAP[int(l)] for l in raw_labels], dtype=np.int32)
        return mapped

    # ------------------------------------------------- prepare_training_data
    def prepare_training_data(
        self,
        bars_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build aligned X, y arrays from bar data and feature matrices.

        Parameters
        ----------
        bars_df : pd.DataFrame
            Must contain at least a ``close`` column, indexed or aligned with
            ``features_df``.
        features_df : pd.DataFrame
            Columns correspond to ``FeatureEngineer.FEATURE_NAMES``.

        Returns
        -------
        X : np.ndarray  (n_valid, n_features)
        y : np.ndarray  (n_valid,)  values in {0, 1, 2}
        """
        y_all = self._compute_labels(bars_df["close"])

        # Drop rows where the label is undefined (last *forward_bars* rows)
        valid_mask = ~np.isnan(bars_df["close"].shift(-settings.FORWARD_BARS).values)
        X = features_df.values[valid_mask].astype(np.float64)
        y = y_all[valid_mask]

        logger.info(
            "Prepared training data: {} samples, label distribution: short={}, flat={}, long={}",
            len(y),
            int((y == 0).sum()),
            int((y == 1).sum()),
            int((y == 2).sum()),
        )
        return X, y

    # ----------------------------------------------------------- train
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train with walk-forward expanding-window validation.

        Walk-forward scheme:
            - Expanding training window starting at 60 days
            - 10-day test window
            - Step forward by 10 days each fold

        The final model is trained on the **entire** dataset.

        Returns
        -------
        dict
            Aggregate validation metrics across all folds.
        """
        bars_per_day = 26  # approx 15-min bars in a trading day
        train_size_init = settings.WALK_FORWARD_TRAIN_DAYS * bars_per_day
        test_size = settings.WALK_FORWARD_TEST_DAYS * bars_per_day

        n = len(y)
        if n < train_size_init + test_size:
            logger.warning(
                "Insufficient data for walk-forward ({} samples). "
                "Training on all data without validation.",
                n,
            )
            self.model = self._build_classifier()
            self.model.fit(X, y)
            return {"note": "no_validation", "samples": n}

        fold_metrics: List[Dict[str, float]] = []
        start = train_size_init

        while start + test_size <= n:
            X_train, y_train = X[:start], y[:start]
            X_test, y_test = X[start: start + test_size], y[start: start + test_size]

            clf = self._build_classifier()
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = clf.predict(X_test)
            metrics = self._calc_metrics(y_test, y_pred)
            fold_metrics.append(metrics)

            logger.debug(
                "WF fold train=0:{} test={}:{} acc={:.4f}",
                start, start, start + test_size, metrics["accuracy"],
            )
            start += test_size

        # Aggregate walk-forward metrics
        agg = self._aggregate_metrics(fold_metrics)
        self._log_metrics(agg, prefix="Walk-Forward")

        # Final model on all data
        self.model = self._build_classifier()
        self.model.fit(X, y)
        logger.info("Final model trained on {} samples for instrument={}", n, self.instrument)

        # Feature importances
        importances = dict(
            zip(FeatureEngineer.FEATURE_NAMES, self.model.feature_importances_)
        )
        sorted_imp = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        logger.info("Top-10 feature importances:")
        for feat, imp in sorted_imp[:10]:
            logger.info("  {}: {:.4f}", feat, imp)

        # Version and persist
        version_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._version = f"xgb_{self.instrument}_{version_ts}"
        self.save_model()
        self._store_model_version(agg, importances, n)
        self._prune_old_versions()

        return agg

    # -------------------------------------------------------- predict
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """Predict class probabilities and derive a trading signal.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_features,)`` — a single scaled feature vector.

        Returns
        -------
        dict
            ``P_long``, ``P_flat``, ``P_short``, ``signal_strength``,
            ``trade`` (bool — True only when max(P_long, P_short) > threshold).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load_model() first.")

        x = features.reshape(1, -1)
        proba = self.model.predict_proba(x)[0]  # [P_short, P_flat, P_long]

        p_short, p_flat, p_long = float(proba[0]), float(proba[1]), float(proba[2])
        signal_strength = p_long - p_short
        trade = max(p_long, p_short) > settings.CONFIDENCE_THRESHOLD

        return {
            "P_long": p_long,
            "P_flat": p_flat,
            "P_short": p_short,
            "signal_strength": signal_strength,
            "trade": trade,
        }

    # ------------------------------------------------- incremental_update
    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """Warm-start an incremental update on new data.

        Uses ``xgb_model`` parameter to continue boosting from the existing
        model, adding a small number of additional rounds.
        """
        if self.model is None:
            logger.warning("No existing model — performing full fit instead of incremental update.")
            self.model = self._build_classifier()
            self.model.fit(X_new, y_new)
            return

        incremental_rounds = 50
        booster = self.model.get_booster()

        new_clf = self._build_classifier()
        new_clf.set_params(n_estimators=incremental_rounds)
        new_clf.fit(X_new, y_new, xgb_model=booster, verbose=False)

        # Merge: replace internal booster
        self.model = new_clf
        logger.info(
            "Incremental update complete: +{} rounds on {} new samples",
            incremental_rounds,
            len(y_new),
        )
        self.save_model()

    # ------------------------------------------------- save / load
    def save_model(self) -> Path:
        """Save the current model to a versioned file, return its path."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        if self._version is None:
            self._version = f"xgb_{self.instrument}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        path = self._model_dir / f"{self._version}.joblib"
        joblib.dump(self.model, path)
        logger.info("Model saved to {}", path)
        return path

    def load_model(self, path: Path) -> None:
        """Load a specific model file."""
        self.model = joblib.load(path)
        self._version = path.stem
        logger.info("Model loaded from {} (version={})", path, self._version)

    def load_latest_model(self) -> bool:
        """Load the most recent versioned model from disk.

        Returns True if a model was loaded, False otherwise.
        """
        model_files = sorted(self._model_dir.glob("xgb_*.joblib"))
        if not model_files:
            logger.warning("No saved models found in {}", self._model_dir)
            return False
        latest = model_files[-1]
        self.load_model(latest)
        return True

    # ================================================ private helpers ====
    @staticmethod
    def _build_classifier() -> XGBClassifier:
        params = dict(settings.XGBOOST_PARAMS)
        params["objective"] = "multi:softprob"
        params["num_class"] = 3
        return XGBClassifier(**params)

    @staticmethod
    def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        labels = [0, 1, 2]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0.0)
        rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0.0)
        return {
            "accuracy": acc,
            "precision_short": prec[0],
            "recall_short": rec[0],
            "f1_short": f1[0],
            "precision_flat": prec[1],
            "recall_flat": rec[1],
            "f1_flat": f1[1],
            "precision_long": prec[2],
            "recall_long": rec[2],
            "f1_long": f1[2],
        }

    @staticmethod
    def _aggregate_metrics(folds: List[Dict[str, float]]) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        if not folds:
            return agg
        for key in folds[0]:
            vals = [f[key] for f in folds]
            agg[key] = float(np.mean(vals))
        return agg

    @staticmethod
    def _log_metrics(m: Dict[str, float], prefix: str = "") -> None:
        tag = f"[{prefix}] " if prefix else ""
        logger.info("{}accuracy:        {:.4f}", tag, m.get("accuracy", 0))
        for cls in _CLASS_NAMES:
            logger.info(
                "{}class={:<5s}  precision={:.4f}  recall={:.4f}  f1={:.4f}",
                tag,
                cls,
                m.get(f"precision_{cls}", 0),
                m.get(f"recall_{cls}", 0),
                m.get(f"f1_{cls}", 0),
            )

    def _store_model_version(
        self,
        metrics: Dict[str, float],
        importances: Dict[str, float],
        training_samples: int,
    ) -> None:
        """Write a ModelVersion row to the database."""
        session = get_session()
        try:
            mv = ModelVersion(
                model_type="xgboost",
                version=self._version,
                file_path=str(self._model_dir / f"{self._version}.joblib"),
                training_samples=training_samples,
                accuracy=metrics.get("accuracy"),
                precision_long=metrics.get("precision_long"),
                recall_long=metrics.get("recall_long"),
                f1_long=metrics.get("f1_long"),
                precision_short=metrics.get("precision_short"),
                recall_short=metrics.get("recall_short"),
                f1_short=metrics.get("f1_short"),
                feature_importances=importances,
                is_active=True,
            )
            # Deactivate previous versions
            session.query(ModelVersion).filter(
                ModelVersion.model_type == "xgboost",
                ModelVersion.is_active == True,  # noqa: E712
            ).update({"is_active": False})

            session.add(mv)
            session.commit()
            logger.info("ModelVersion stored: version={}", self._version)
        except Exception:
            session.rollback()
            logger.exception("Failed to store ModelVersion")
        finally:
            session.close()

    def _prune_old_versions(self) -> None:
        """Keep only the last MAX_MODEL_VERSIONS model files on disk."""
        model_files = sorted(self._model_dir.glob("xgb_*.joblib"))
        max_keep = settings.MAX_MODEL_VERSIONS
        if len(model_files) > max_keep:
            to_delete = model_files[: len(model_files) - max_keep]
            for f in to_delete:
                f.unlink(missing_ok=True)
                logger.info("Pruned old model file: {}", f)
