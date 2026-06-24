"""
jud_model/models/hmm_model.py
GaussianHMM Regime Detection Model -- Framework Version

Dependencies: pip install hmmlearn scikit-learn
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.analyzer import AnalysisResult, Regime
from core.features import extract_features, IDX_VOLATILITY, IDX_ADX_NORM, IDX_ATR_RATIO, IDX_BB_WIDTH
from models.base_model import BaseRegimeModel
from utils.logger import get_logger

_log = get_logger(__name__)


class HMMRegimeModel(BaseRegimeModel):
    def __init__(self, n_states: int = 2, covariance_type: str = "full", n_iter: int = 300, random_state: int = 42):
        try:
            from hmmlearn import hmm
        except ImportError:
            raise ImportError("pip install hmmlearn")
        from sklearn.preprocessing import StandardScaler
        self.n_states = n_states
        self._hmm_cls = hmm.GaussianHMM
        self._model_kwargs = dict(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, random_state=random_state, verbose=False)
        self._model = None
        self._scaler = StandardScaler()
        self._state_map: Dict[int, str] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "HMMRegimeModel":
        X, vm = extract_features(df)
        Xv = X[vm]
        if len(Xv) < max(100, self.n_states * 20):
            raise ValueError(f"Insufficient: {len(Xv)}")
        self._model = self._hmm_cls(**self._model_kwargs)
        self._model.fit(self._scaler.fit_transform(Xv))
        self._is_fitted = True
        self._auto_label_states()
        return self

    def _auto_label_states(self) -> None:
        m = self._scaler.inverse_transform(self._model.means_)
        def _n(a):
            r = a.max() - a.min()
            return (a - a.min()) / r if r > 1e-10 else a * 0
        ts = _n(m[:, IDX_VOLATILITY]) * 0.35 + _n(m[:, IDX_ADX_NORM]) * 0.35 + _n(m[:, IDX_ATR_RATIO]) * 0.20 + _n(m[:, IDX_BB_WIDTH]) * 0.10
        ranked = np.argsort(ts)
        self._state_map.clear()
        if self.n_states == 2:
            self._state_map[int(ranked[0])] = Regime.OSCILLATION
            self._state_map[int(ranked[1])] = Regime.TREND
        elif self.n_states == 3:
            self._state_map[int(ranked[0])] = Regime.OSCILLATION
            self._state_map[int(ranked[1])] = Regime.UNKNOWN
            self._state_map[int(ranked[2])] = Regime.TREND

    def predict(self, df: pd.DataFrame) -> AnalysisResult:
        if not self._is_fitted:
            raise RuntimeError("Not trained")
        X, vm = extract_features(df)
        Xv = X[vm]
        if len(Xv) < 10:
            return AnalysisResult(regime=Regime.UNKNOWN, confidence=0.0, score=0.0)
        Xs = self._scaler.transform(Xv)
        states = self._model.predict(Xs)
        posts = self._model.predict_proba(Xs)
        s = int(states[-1])
        c = float(posts[-1, s])
        regime = self._state_map.get(s, Regime.UNKNOWN)
        score = c if regime == Regime.TREND else (-c if regime == Regime.OSCILLATION else 0.0)
        lf = X[-1]
        return AnalysisResult(regime=regime, confidence=c, score=score,
            adx_value=float(lf[IDX_ADX_NORM]) * 100 if not np.isnan(lf[IDX_ADX_NORM]) else 0.0,
            atr_ratio=float(lf[IDX_ATR_RATIO]) if not np.isnan(lf[IDX_ATR_RATIO]) else 1.0,
            bb_width_pct=float(lf[IDX_BB_WIDTH]) if not np.isnan(lf[IDX_BB_WIDTH]) else 0.0)

    def decode_history(self, df: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Not trained")
        X, vm = extract_features(df)
        labels = np.full(len(df), None, dtype=object)
        if vm.sum() < 10:
            return pd.Series(labels, index=df.index)
        Xs = self._scaler.transform(X[vm])
        states = self._model.predict(Xs)
        for idx, st in zip(np.where(vm)[0], states):
            labels[idx] = self._state_map.get(int(st), Regime.UNKNOWN)
        return pd.Series(labels, index=df.index)

    def partial_fit(self, df: pd.DataFrame) -> "HMMRegimeModel":
        if not self._is_fitted:
            return self.fit(df)
        X, vm = extract_features(df)
        Xv = X[vm]
        if len(Xv) < 50:
            return self
        warm = self._hmm_cls(**{**self._model_kwargs, "n_iter": 50, "init_params": ""})
        warm.startprob_ = self._model.startprob_
        warm.transmat_ = self._model.transmat_
        warm.means_ = self._model.means_
        warm.covars_ = self._model.covars_
        warm.fit(self._scaler.transform(Xv))
        self._model = warm
        self._auto_label_states()
        return self

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"model": self._model, "scaler": self._scaler, "state_map": self._state_map, "n_states": self.n_states, "model_kwargs": self._model_kwargs}, path)

    @classmethod
    def load(cls, path: str) -> "HMMRegimeModel":
        import joblib
        data = joblib.load(path)
        obj = cls(n_states=data["n_states"])
        obj._model = data["model"]
        obj._scaler = data["scaler"]
        obj._state_map = data["state_map"]
        obj._model_kwargs = data.get("model_kwargs", obj._model_kwargs)
        obj._is_fitted = True
        return obj
