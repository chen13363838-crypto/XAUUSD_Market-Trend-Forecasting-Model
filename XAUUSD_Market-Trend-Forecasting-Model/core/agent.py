"""
jud_model/core/agent.py
JudgeAgent -- Regime Detection Agent (supports rule + learn + hybrid modes)

Modes
-----
    "rule"   : Pure rule engine (fixed thresholds, no training needed)
    "learn"  : Pure ML model (requires train() first)
    "hybrid" : Model first, fall back to rules if confidence is low
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import AgentConfig, DEFAULT_CONFIG
from core.analyzer import RegimeAnalyzer, AnalysisResult, Regime
from data.feed import DataFeeder
from models.base_model import BaseRegimeModel
from utils.logger import get_logger


class JudgeAgent:
    """
    Regime Detection Agent.

    Parameters
    ----------
    config      : AgentConfig, optional
    mode        : str, 'rule' | 'learn' | 'hybrid'
    buffer_size : int, rolling buffer size for online mode
    """

    def __init__(self, config: Optional[AgentConfig] = None, mode: str = "hybrid", buffer_size: int = 500):
        assert mode in ("rule", "learn", "hybrid")
        self.cfg = config or DEFAULT_CONFIG
        self.mode = mode
        self._log = get_logger(__name__, self.cfg.log_level)
        self._rule_analyzer = RegimeAnalyzer(self.cfg)
        self._learn_model: Optional[BaseRegimeModel] = None
        self._feeder = DataFeeder(buffer_size=buffer_size)
        self._history: List[AnalysisResult] = []

    def train(self, df: pd.DataFrame, model_type: str = "gmm", n_states: int = 2, **model_kwargs) -> "JudgeAgent":
        """Train ML model with historical K-line data."""
        model = self._build_model(model_type, n_states, **model_kwargs)
        model.fit(df)
        self._learn_model = model
        if self.mode == "rule":
            self.mode = "hybrid"
        return self

    def partial_fit(self, df: pd.DataFrame) -> "JudgeAgent":
        """Incremental model update with new data."""
        if self._learn_model is None:
            return self.train(df)
        if hasattr(self._learn_model, "partial_fit"):
            self._learn_model.partial_fit(df)
        else:
            self._learn_model.fit(df)
        return self

    def run(self, df: pd.DataFrame) -> AnalysisResult:
        """Analyze current regime based on mode."""
        result = self._dispatch(df)
        self._history.append(result)
        return result

    def tick(self, bar: Dict[str, Any]) -> Optional[AnalysisResult]:
        """Feed single bar in real-time."""
        df = self._feeder.feed_single_bar(bar)
        if len(df) < self.cfg.min_bars:
            return None
        result = self._dispatch(df)
        self._history.append(result)
        return result

    def decode_history(self, df: pd.DataFrame) -> pd.Series:
        """Label each historical bar with regime (for backtesting)."""
        if self._learn_model is None:
            raise RuntimeError("Call train() first")
        return self._learn_model.decode_history(df)

    def save_model(self, path: str) -> None:
        if self._learn_model is None:
            raise RuntimeError("No model to save")
        self._learn_model.save(path)

    def load_model(self, path: str, model_type: str = "gmm") -> "JudgeAgent":
        model_cls = self._model_class(model_type)
        self._learn_model = model_cls.load(path)
        if self.mode == "rule":
            self.mode = "hybrid"
        return self

    @property
    def last_result(self) -> Optional[AnalysisResult]:
        return self._history[-1] if self._history else None

    @property
    def history(self) -> List[AnalysisResult]:
        return list(self._history)

    @property
    def has_model(self) -> bool:
        return self._learn_model is not None and self._learn_model.is_fitted

    def clear_history(self) -> None:
        self._history.clear()
        self._feeder.clear_buffer()

    def _dispatch(self, df: pd.DataFrame) -> AnalysisResult:
        if self.mode == "rule" or not self.has_model:
            return self._rule_analyzer.analyze(df)
        if self.mode == "learn":
            return self._learn_model.predict(df)
        result = self._learn_model.predict(df)
        if result.confidence < 0.50:
            result = self._rule_analyzer.analyze(df)
        return result

    @staticmethod
    def _model_class(model_type: str):
        if model_type in ("gmm", "default"):
            from models.gmm_model import GMMRegimeModel
            return GMMRegimeModel
        if model_type == "hmm":
            from models.hmm_model import HMMRegimeModel
            return HMMRegimeModel
        raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _build_model(model_type: str, n_states: int, **kwargs) -> BaseRegimeModel:
        cls = JudgeAgent._model_class(model_type)
        return cls(n_states=n_states, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str, model_type: str = "gmm", **kwargs) -> "JudgeAgent":
        agent = cls(mode="learn", **kwargs)
        agent.load_model(model_path, model_type)
        return agent
