"""
jud_model/core/multi_tf_agent.py
MultiTFAgent -- Multi-Timeframe Regime Detection Agent

Input: {"M30": df_m30, "M15": df_m15, "M5": df_m5}
Internally aligns bars across timeframes, extracts fused features, runs GMM.
Output: AnalysisResult (same interface as single-timeframe JudgeAgent)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from core.analyzer import AnalysisResult, RegimeAnalyzer, Regime, OscillationRange
from config.settings import AgentConfig, DEFAULT_CONFIG
from models.multi_tf_gmm import MultiTFGMMModel
from utils.logger import get_logger


class MultiTFAgent:
    """
    Multi-Timeframe Regime Detection Agent.

    Parameters
    ----------
    config : AgentConfig, optional
    mode   : 'learn' | 'rule' | 'hybrid'
    """

    def __init__(self, config: Optional[AgentConfig] = None, mode: str = "hybrid"):
        assert mode in ("rule", "learn", "hybrid")
        self.cfg = config or DEFAULT_CONFIG
        self.mode = mode
        self._log = get_logger(__name__, self.cfg.log_level)
        self._rule_analyzer = RegimeAnalyzer(self.cfg)
        self._model: Optional[MultiTFGMMModel] = None
        self._history: List[AnalysisResult] = []

    def train(self, tf_data: Dict[str, pd.DataFrame], n_states: int = 2, **model_kwargs) -> "MultiTFAgent":
        """Train with multi-timeframe historical data."""
        model = MultiTFGMMModel(n_states=n_states, **model_kwargs)
        model.fit(tf_data)
        self._model = model
        if self.mode == "rule":
            self.mode = "hybrid"
        return self

    def run(self, tf_data: Dict[str, pd.DataFrame]) -> AnalysisResult:
        """Multi-timeframe regime detection."""
        result = self._dispatch(tf_data)
        self._history.append(result)
        return result

    def decode_history(self, tf_data: Dict[str, pd.DataFrame]) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Call train() first")
        return self._model.decode_history(tf_data)

    def save_model(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        self._model.save(path)

    def load_model(self, path: str) -> "MultiTFAgent":
        self._model = MultiTFGMMModel.load(path)
        if self.mode == "rule":
            self.mode = "hybrid"
        return self

    @property
    def last_result(self) -> Optional[AnalysisResult]:
        return self._history[-1] if self._history else None

    @property
    def has_model(self) -> bool:
        return self._model is not None and self._model._is_fitted

    def _dispatch(self, tf_data: Dict[str, pd.DataFrame]) -> AnalysisResult:
        if self.mode == "rule" or not self.has_model:
            result = self._rule_analyzer.analyze(tf_data["M30"])
            if result.regime == Regime.OSCILLATION and "M30" in tf_data:
                result.osc_range = self._simple_m30_range(tf_data["M30"])
            return result
        if self.mode == "learn":
            return self._model.predict(tf_data)
        # hybrid
        result = self._model.predict(tf_data)
        if result.confidence < 0.50:
            result = self._rule_analyzer.analyze(tf_data["M30"])
        return result

    @staticmethod
    def _simple_m30_range(df_m30: pd.DataFrame, lookback: int = 30) -> Optional[OscillationRange]:
        if len(df_m30) < 4:
            return None
        seg = df_m30.tail(min(lookback, len(df_m30)))
        seg_high = float(seg["high"].max())
        seg_low = float(seg["low"].min())
        center = (seg_high + seg_low) / 2.0
        spread = seg_high - seg_low
        return OscillationRange(
            high=seg_high, low=seg_low, center=center, spread=spread,
            spread_pct=spread / center * 100 if center > 0 else 0.0,
            m30_bars=len(seg),
        )

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "MultiTFAgent":
        agent = cls(mode="learn", **kwargs)
        agent.load_model(model_path)
        return agent
