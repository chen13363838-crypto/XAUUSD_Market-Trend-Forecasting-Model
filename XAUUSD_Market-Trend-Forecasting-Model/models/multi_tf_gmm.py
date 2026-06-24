"""
jud_model/models/multi_tf_gmm.py
Multi-Timeframe GMM Regime Detection Model -- Framework Version

Input: M30/M15/M5 fused features (21 dims).
State labeling focuses on M30 dimensions.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from core.analyzer import AnalysisResult, Regime, OscillationRange
from core.features import IDX_VOLATILITY, IDX_ADX_NORM, IDX_ATR_RATIO, IDX_BB_WIDTH
from core.multi_tf_features import extract_multi_tf_features, N_MTF_FEATURES
from models.base_model import BaseRegimeModel
from utils.logger import get_logger

_log = get_logger(__name__)

M30_BASE = 0
IDX_MTF_VOL = M30_BASE + IDX_VOLATILITY
IDX_MTF_ADX = M30_BASE + IDX_ADX_NORM
IDX_MTF_ATR = M30_BASE + IDX_ATR_RATIO
IDX_MTF_BB  = M30_BASE + IDX_BB_WIDTH


class MultiTFGMMModel(BaseRegimeModel):
    def __init__(self, n_states: int = 3, covariance_type: str = "full",
                 n_init: int = 10, max_iter: int = 500,
                 window: int = 30, max_fuzzy_bars: int = 8, random_state: int = 42):
        self.n_states = n_states
        self.window = window
        self.max_fuzzy_bars = max_fuzzy_bars
        self._gmm = GaussianMixture(n_components=n_states, covariance_type=covariance_type,
            n_init=n_init, max_iter=max_iter, random_state=random_state)
        self._scaler = StandardScaler()
        self._state_map: Dict[int, str] = {}
        self._means_orig: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, tf_data: Dict[str, pd.DataFrame]) -> "MultiTFGMMModel":
        X, vm = extract_multi_tf_features(tf_data)
        Xv = X[vm]
        if len(Xv) < max(50, self.n_states * 10):
            raise ValueError(f"Insufficient: {len(Xv)}")
        self._gmm.fit(self._scaler.fit_transform(Xv))
        self._is_fitted = True
        self._means_orig = self._scaler.inverse_transform(self._gmm.means_)
        self._auto_label_states()
        return self

    def _auto_label_states(self) -> None:
        m = self._means_orig
        if m is None:
            return
        def _n(a):
            r = a.max() - a.min()
            return (a - a.min()) / r if r > 1e-10 else np.zeros_like(a)
        ts = _n(m[:, IDX_MTF_VOL]) * 0.35 + _n(m[:, IDX_MTF_ADX]) * 0.35 + _n(m[:, IDX_MTF_ATR]) * 0.20 + _n(m[:, IDX_MTF_BB]) * 0.10
        ranked = np.argsort(ts)
        self._state_map.clear()
        if self.n_states == 2:
            self._state_map[int(ranked[0])] = Regime.OSCILLATION
            self._state_map[int(ranked[1])] = Regime.TREND
        elif self.n_states >= 3:
            self._state_map[int(ranked[0])] = Regime.OSCILLATION
            for i in range(1, self.n_states - 1):
                self._state_map[int(ranked[i])] = Regime.FUZZY
            self._state_map[int(ranked[-1])] = Regime.TREND

    def predict(self, tf_data: Dict[str, pd.DataFrame]) -> AnalysisResult:
        if not self._is_fitted:
            raise RuntimeError("Not trained")
        X, vm = extract_multi_tf_features(tf_data)
        Xv = X[vm]
        if len(Xv) < 5:
            return AnalysisResult(regime=Regime.UNKNOWN, confidence=0.0, score=0.0)
        Xr = Xv[-self.window:]
        Xs = self._scaler.transform(Xr)
        proba = self._gmm.predict_proba(Xs)
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
            adx_value=_s(lf, IDX_MTF_ADX) * 100,
            atr_ratio=_s(lf, IDX_MTF_ATR) or 1.0,
            bb_width_pct=_s(lf, IDX_MTF_BB))

    def decode_history(self, tf_data: Dict[str, pd.DataFrame], cap_fuzzy: bool = True) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Not trained")
        X, vm = extract_multi_tf_features(tf_data)
        labels = np.full(X.shape[0], None, dtype=object)
        if vm.sum() < 5:
            return pd.Series(labels)
        Xs = self._scaler.transform(X[vm])
        states = self._gmm.predict(Xs)
        for vi, st in zip(np.where(vm)[0], states):
            labels[vi] = self._state_map.get(int(st), Regime.UNKNOWN)
        return pd.Series(labels)

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"gmm": self._gmm, "scaler": self._scaler, "state_map": self._state_map,
            "n_states": self.n_states, "window": self.window,
            "max_fuzzy_bars": self.max_fuzzy_bars, "means_orig": self._means_orig}, path)

    @classmethod
    def load(cls, path: str) -> "MultiTFGMMModel":
        import joblib
        data = joblib.load(path)
        obj = cls(n_states=data["n_states"], window=data.get("window", 30), max_fuzzy_bars=data.get("max_fuzzy_bars", 8))
        obj._gmm = data["gmm"]
        obj._scaler = data["scaler"]
        obj._state_map = data["state_map"]
        obj._means_orig = data.get("means_orig")
        obj._is_fitted = True
        return obj
