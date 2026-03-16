"""
Reinforcement-learning position sizer for the Oil Quantitative Trading Bot.

Provides a custom Gymnasium environment (OilTradingEnv) and a PPO-based
position-sizing agent (RLPositionSizer) that outputs continuous actions in
[-1, 1] mapped to long / hold / short with proportional sizing.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import settings
from db.models import ModelVersion, get_session
from ml.features import FeatureEngineer


# ---------------------------------------------------------------------------
# Custom Gymnasium environment
# ---------------------------------------------------------------------------

class OilTradingEnv(gym.Env):
    """Simulated oil-trading environment for RL position sizing.

    State
    -----
    Concatenation of:
        - feature vector          (NUM_FEATURES,)
        - current_position        scalar in [-1, 1]
        - unrealized_pnl          scalar (fraction of portfolio)
        - portfolio_heat          scalar in [0, 1]

    Action
    ------
    Continuous scalar in [-1, 1]:
        action >  0.2  →  go / stay long  (size = action)
        action < -0.2  →  go / stay short (size = |action|)
        else           →  hold / flatten

    Reward
    ------
    r = (realized_pnl / portfolio_value)
        - 0.5 * max_drawdown_penalty
        - 0.1 * transaction_cost
        + 0.2 * sharpe_bonus
        - 0.3 * margin_call_penalty
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        instrument: str = "CL",
        initial_balance: float = 100_000.0,
    ) -> None:
        super().__init__()

        self.features = features.astype(np.float64)
        self.prices = prices.astype(np.float64)
        self.instrument = instrument
        self.initial_balance = initial_balance

        n_features = self.features.shape[1]
        # state = features + [position, unrealized_pnl, portfolio_heat]
        self._obs_size = n_features + 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_size,), dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64,
        )

        # Episode bookkeeping (set in reset)
        self._step_idx: int = 0
        self._position: float = 0.0  # in [-1, 1]
        self._entry_price: float = 0.0
        self._balance: float = initial_balance
        self._peak_balance: float = initial_balance
        self._returns: List[float] = []
        self._prev_balance: float = initial_balance

    # ------------------------------------------------ gym API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_idx = 0
        self._position = 0.0
        self._entry_price = 0.0
        self._balance = self.initial_balance
        self._peak_balance = self.initial_balance
        self._prev_balance = self.initial_balance
        self._returns = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_val = float(np.clip(action[0], -1.0, 1.0))
        price = self.prices[self._step_idx]
        next_price = self.prices[min(self._step_idx + 1, len(self.prices) - 1)]

        # ---- Determine desired position ------------------------------------
        threshold = settings.RL_ACTION_THRESHOLD
        if action_val > threshold:
            desired_pos = action_val
        elif action_val < -threshold:
            desired_pos = action_val
        else:
            desired_pos = 0.0

        # ---- Transaction cost on position change ---------------------------
        pos_delta = abs(desired_pos - self._position)
        txn_cost = pos_delta * settings.SLIPPAGE_PCT * self._balance

        # ---- Realised PnL from holding previous position -------------------
        price_return = (next_price - price) / price if price != 0 else 0.0
        position_pnl = self._position * price_return * self._balance

        # ---- Update balance ------------------------------------------------
        self._balance += position_pnl - txn_cost
        step_return = (self._balance - self._prev_balance) / self._prev_balance if self._prev_balance != 0 else 0.0
        self._returns.append(step_return)
        self._prev_balance = self._balance

        # ---- Drawdown ------------------------------------------------------
        self._peak_balance = max(self._peak_balance, self._balance)
        drawdown = (self._peak_balance - self._balance) / self._peak_balance if self._peak_balance != 0 else 0.0

        # ---- Sharpe bonus (rolling) ----------------------------------------
        if len(self._returns) >= 20:
            recent = np.array(self._returns[-20:])
            mean_r = np.mean(recent)
            std_r = np.std(recent)
            sharpe = mean_r / std_r if std_r > 1e-9 else 0.0
        else:
            sharpe = 0.0

        # ---- Margin-call penalty -------------------------------------------
        margin_call = 1.0 if self._balance < self.initial_balance * 0.5 else 0.0

        # ---- Composite reward ----------------------------------------------
        pnl_frac = position_pnl / self.initial_balance if self.initial_balance != 0 else 0.0
        reward = (
            pnl_frac
            - settings.REWARD_DRAWDOWN_PENALTY * drawdown
            - settings.REWARD_TRANSACTION_COST * (txn_cost / self.initial_balance)
            + settings.REWARD_SHARPE_BONUS * max(sharpe, 0.0)
            - settings.REWARD_MARGIN_CALL_PENALTY * margin_call
        )

        # ---- Advance -------------------------------------------------------
        self._position = desired_pos
        self._entry_price = price if pos_delta > 0.01 else self._entry_price
        self._step_idx += 1

        terminated = self._step_idx >= len(self.prices) - 1
        truncated = self._balance <= 0.0

        info = {
            "balance": self._balance,
            "position": self._position,
            "drawdown": drawdown,
            "sharpe": sharpe,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------ helpers
    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_idx, len(self.features) - 1)
        feat = self.features[idx]

        unrealized_pnl = 0.0
        if self._position != 0.0 and self._entry_price != 0.0:
            current_price = self.prices[idx]
            unrealized_pnl = (
                self._position
                * (current_price - self._entry_price)
                / self._entry_price
            )

        portfolio_heat = abs(self._position)

        obs = np.concatenate([
            feat,
            np.array([self._position, unrealized_pnl, portfolio_heat], dtype=np.float64),
        ])
        return obs


# ---------------------------------------------------------------------------
# RL Position Sizer
# ---------------------------------------------------------------------------

class RLPositionSizer:
    """PPO-based position sizer with per-instrument models.

    Wraps Stable-Baselines3 PPO and manages training, prediction, and
    model persistence for each traded instrument.
    """

    def __init__(self, instruments: Optional[List[str]] = None) -> None:
        if instruments is None:
            instruments = list(settings.ALL_INSTRUMENTS.keys())
        self.instruments = instruments
        self._models: Dict[str, PPO] = {}
        self._model_dir = settings.MODELS_DIR / "rl"
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Try to load existing models
        for inst in self.instruments:
            self._try_load(inst)

    # ------------------------------------------------------------ train
    def train(
        self,
        instrument: str,
        historical_features: np.ndarray,
        historical_prices: np.ndarray,
        total_timesteps: int = 50_000,
    ) -> None:
        """Train (or retrain) a PPO agent for *instrument*.

        Parameters
        ----------
        instrument : str
            E.g. ``"CL"``, ``"BZ"``, ``"USO"``.
        historical_features : np.ndarray
            Shape ``(n_bars, n_features)`` — last 90 days of scaled features.
        historical_prices : np.ndarray
            Shape ``(n_bars,)`` — close prices aligned with features.
        total_timesteps : int
            PPO training budget.
        """
        logger.info(
            "Training RL agent for instrument={} on {} bars, {} timesteps",
            instrument,
            len(historical_prices),
            total_timesteps,
        )

        env = DummyVecEnv([
            lambda: OilTradingEnv(
                features=historical_features,
                prices=historical_prices,
                instrument=instrument,
            )
        ])

        ppo_kwargs = dict(settings.PPO_PARAMS)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            **ppo_kwargs,
        )
        model.learn(total_timesteps=total_timesteps)

        self._models[instrument] = model
        self.save_model(instrument)
        self._store_model_version(instrument, len(historical_prices))
        logger.info("RL agent training complete for instrument={}", instrument)

    # ---------------------------------------------------------- predict
    def predict(self, instrument: str, state: np.ndarray) -> float:
        """Return a position size in [-1, 1] for the given state.

        Parameters
        ----------
        instrument : str
        state : np.ndarray
            Observation vector (features + position + unrealized_pnl +
            portfolio_heat).

        Returns
        -------
        float
            Desired position size.  Values between
            (-ACTION_THRESHOLD, +ACTION_THRESHOLD) mean "hold / flat".
        """
        model = self._models.get(instrument)
        if model is None:
            logger.warning(
                "No RL model for instrument={}; returning 0.0 (flat)", instrument,
            )
            return 0.0

        action, _ = model.predict(state, deterministic=True)
        position_size = float(np.clip(action[0], -1.0, 1.0))
        return position_size

    # ------------------------------------------------- save / load
    def save_model(self, instrument: str) -> Path:
        """Save the PPO model for *instrument* to disk."""
        model = self._models.get(instrument)
        if model is None:
            raise RuntimeError(f"No model to save for instrument={instrument}")
        path = self._model_dir / f"ppo_{instrument}"
        model.save(str(path))
        logger.info("RL model saved: {}", path)
        return path

    def load_model(self, instrument: str, path: Optional[Path] = None) -> bool:
        """Load a PPO model for *instrument*.

        If *path* is None the default location is used.
        Returns True on success.
        """
        if path is None:
            path = self._model_dir / f"ppo_{instrument}.zip"
        if not path.exists():
            logger.warning("RL model file not found: {}", path)
            return False
        self._models[instrument] = PPO.load(str(path))
        logger.info("RL model loaded: {}", path)
        return True

    # ------------------------------------------------ private helpers
    def _try_load(self, instrument: str) -> None:
        """Silently attempt to load a saved model for *instrument*."""
        path = self._model_dir / f"ppo_{instrument}.zip"
        if path.exists():
            try:
                self.load_model(instrument, path)
            except Exception:
                logger.debug("Could not load RL model for {}", instrument)

    def _store_model_version(self, instrument: str, training_samples: int) -> None:
        """Record the RL model version in the database."""
        version = f"rl_{instrument}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        session = get_session()
        try:
            mv = ModelVersion(
                model_type="rl",
                version=version,
                file_path=str(self._model_dir / f"ppo_{instrument}.zip"),
                training_samples=training_samples,
                is_active=True,
            )
            # Deactivate previous RL versions for this instrument
            session.query(ModelVersion).filter(
                ModelVersion.model_type == "rl",
                ModelVersion.version.like(f"rl_{instrument}_%"),
                ModelVersion.is_active == True,  # noqa: E712
            ).update({"is_active": False}, synchronize_session="fetch")

            session.add(mv)
            session.commit()
            logger.info("RL ModelVersion stored: version={}", version)
        except Exception:
            session.rollback()
            logger.exception("Failed to store RL ModelVersion")
        finally:
            session.close()
