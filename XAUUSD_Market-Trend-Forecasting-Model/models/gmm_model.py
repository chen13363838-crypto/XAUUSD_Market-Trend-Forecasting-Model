"""
jud_model/models/gmm_model.py
GaussianMixture (GMM) Regime Detection Model -- Framework Version

Uses GMM to soft-cluster feature vectors into market states.
Each component represents a market regime (trend / oscillation).
States are auto-labeled based on learned feature means.

Dependencies: scikit-learn, joblib
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from core.analyzer import AnalysisResult, Regime
from core.features import (
    extract_features,
    IDX_VOLATILITY, IDX_ADX_NORM, IDX_ATR_RATIO, IDX_BB_WIDTH,
)
from models.base_model import BaseRegimeModel
from utils.logger import get_logger

_log = get_logger(__name__)


class GMMRegimeModel(BaseRegimeModel):
    """
    GMM-based regime detection model.

    Parameters
    ----------
    n_states        : number of mixture components
    covariance_type : 'full' / 'diag' / 'tied' / 'spherical'
    n_init          : random initialization count
    max_iter        : EM max iterations
    window          : recent bars for prediction (sequence awareness)
    """

    def __init__(self, n_states: int = 2, covariance_type: str = "full",
                 n_init: int = 10, max_iter: int = 500,
                 window: int = 30, random_state: int = 42):
        self.n_states = n_states
        self.window = window
        self._gmm = GaussianMixture(
            n_components=n_states, covariance_type=covariance_type,
            n_init=n_init, max_iter=max_iter, random_state=random_state,
        )
        self._scaler = StandardScaler()
        self._state_map: Dict[int, str] = {}
        self._means_orig: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "GMMRegimeModel":
        """Train GMM from historical K-line data."""
        X, valid_mask = extract_features(df)
        X_valid = X[valid_mask]
        if len(X_valid) < max(50, self.n_states * 10):
            raise ValueError(f"Insufficient samples: {len(X_valid)}")
        X_scaled = self._scaler.fit_transform(X_valid)
        self._gmm.fit(X_scaled)
        self._is_fitted = True
        self._means_orig = self._scaler.inverse_transform(self._gmm.means_)
        self._auto_label_states()
        self._log_state_info()
        return self

    def _auto_label_states(self) -> None:
        """
        Auto-label states by trend strength score.
        Score = 0.35*vol + 0.35*ADX + 0.20*ATR + 0.10*BB
        Higher -> TREND, lower -> OSCILLATION.
        """
        means = self._means_orig
        if means is None:
            return
        def _n(a):
            r = a.max() - a.min()
            return (a - a.min()) / r if r > 1e-10 else np.zeros_like(a)
        ts = (_n(means[:, IDX_VOLATILITY]) * 0.35 + _n(means[:, IDX_ADX_NORM]) * 0.35 +
              _n(means[:, IDX_ATR_RATIO]) * 0.20 + _n(means[:, IDX_BB_WIDTH]) * 0.10)
        ranked = np.argsort(ts)
        self._state_map.clear()
        if self.n_states == 2:
            self._state_map[int(ranked[0])] = Regime.OSCILLATION
            self._state_map[int(ranked[1])] = Regime.TREND
        elif self.n_states == 3:
            self._state_map[int(ranked[0])] = Regime.OSCILLATION
            self._state_map[int(ranked[1])] = Regime.UNKNOWN
            self._state_map[int(ranked[2])] = Regime.TREND
        else:
            for i, sid in enumerate(ranked):
                self._state_map[int(sid)] = (Regime.OSCILLATION if i == 0 else
                    Regime.TREND if i == len(ranked) - 1 else Regime.UNKNOWN)

    def _log_state_info(self) -> None:
        if self._means_orig is None:
            return
        for sid, regime in self._state_map.items():
            _log.info(f"  Component-{sid} -> {regime}")

    def predict(self, df: pd.DataFrame) -> AnalysisResult:
        """Predict using time-decay weighted average of recent bars."""
        if not self._is_fitted:
            raise RuntimeError("Not trained")
        X, valid_mask = extract_features(df)
        X_valid = X[valid_mask]
        if len(X_valid) < 5:
            return AnalysisResult(regime=Regime.UNKNOWN, confidence=0.0, score=0.0)
        X_recent = X_valid[-self.window:]
        X_scaled = self._scaler.transform(X_recent)
        proba = self._gmm.predict_proba(X_scaled)
        n = len(proba)
        w = np.exp(np.linspace(-1, 0, n))
        w /= w.sum()
        avg = (proba * w[:, None]).sum(axis=0)
        best = int(np.argmax(avg))
        conf = float(avg[best])
        regime = self._state_map.get(best, Regime.UNKNOWN)
        score = conf if regime == Regime.TREND else (-conf if regime == Regime.OSCILLATION else 0.0)
        lf = X[-1]
        def _s(a, i):
            return float(a[i]) if not np.isnan(a[i]) else 0.0
        return AnalysisResult(regime=regime, confidence=conf, score=score,
            adx_value=_s(lf, IDX_ADX_NORM) * 100,
            atr_ratio=_s(lf, IDX_ATR_RATIO) or 1.0,
            bb_width_pct=_s(lf, IDX_BB_WIDTH))

    def decode_history(self, df: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Not trained")
        X, valid_mask = extract_features(df)
        labels = np.full(len(df), None, dtype=object)
        if valid_mask.sum() < 5:
            return pd.Series(labels, index=df.index)
        X_scaled = self._scaler.transform(X[valid_mask])
        states = self._gmm.predict(X_scaled)
        for idx, state in zip(np.where(valid_mask)[0], states):
            labels[idx] = self._state_map.get(int(state), Regime.UNKNOWN)
        return pd.Series(labels, index=df.index)

    def partial_fit(self, df: pd.DataFrame) -> "GMMRegimeModel":
        if not self._is_fitted:
            return self.fit(df)
        X, valid_mask = extract_features(df)
        X_valid = X[valid_mask]
        if len(X_valid) < 30:
            return self
        warm = GaussianMixture(n_components=self.n_states,
            covariance_type=self._gmm.covariance_type,
            max_iter=50, n_init=1, warm_start=False,
            means_init=self._gmm.means_, random_state=42)
        warm.fit(self._scaler.transform(X_valid))
        self._gmm = warm
        self._means_orig = self._scaler.inverse_transform(self._gmm.means_)
        self._auto_label_states()
        return self

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"gmm": self._gmm, "scaler": self._scaler,
            "state_map": self._state_map, "n_states": self.n_states,
            "window": self.window, "means_orig": self._means_orig}, path)
        _log.info(f"[GMM] Saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "GMMRegimeModel":
        import joblib
        data = joblib.load(path)
        obj = cls(n_states=data["n_states"], window=data.get("window", 30))
        obj._gmm = data["gmm"]
        obj._scaler = data["scaler"]
        obj._state_map = data["state_map"]
        obj._means_orig = data.get("means_orig")
        obj._is_fitted = True
        return obj
