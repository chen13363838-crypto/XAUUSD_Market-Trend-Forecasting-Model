"""
jud_model/core/analyzer.py
多因子盘型评分引擎（框架版）

盘型输出：
    TREND        — 趋势突破盘
    OSCILLATION  — 震荡盘
    UNKNOWN      — 信号不明确（置信度不足）

设计思路
--------
每个因子产出 score ∈ [-1.0, +1.0]：
    +1.0  → 强烈趋势信号
    -1.0  → 强烈震荡信号
     0.0  → 中性

加权综合 score + 一致性置信度 → 最终判断。

⚠️ 注意：本文件为开源框架版，评分函数提供了基础实现作为参考。
    你可以根据自己的策略调整阈值和权重。
    可参考的因子包括：ADX 趋势强度、BB 宽度分位、ATR 波动比值、区间突破等。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd

from config.settings import AgentConfig, DEFAULT_CONFIG
from core.indicators import (
    calc_atr,
    calc_adx,
    calc_bb,
    calc_percentile_rank,
    calc_range_break,
)
from utils.logger import get_logger


# ── 枚举 ───────────────────────────────────────────────────────────────────────

class Regime:
    TREND       = "TREND"
    FUZZY       = "FUZZY"        # 震荡 <-> 趋势过渡态
    OSCILLATION = "OSCILLATION"
    UNKNOWN     = "UNKNOWN"


# ── 结果数据类 ─────────────────────────────────────────────────────────────────

@dataclass
class OscillationRange:
    """震荡区间数据"""

    high: float                  # 区间最高价（阻力位）
    low:  float                  # 区间最低价（支撑位）
    center: float                # 区间中枢 (high+low)/2
    spread: float                # 区间振幅 (high-low)
    spread_pct: float            # 区间振幅百分比

    m30_bars: int = 0
    m15_bars: int = 0
    m5_bars:  int = 0

    bb_upper: float = 0.0
    bb_lower: float = 0.0


@dataclass
class AnalysisResult:
    """盘型分析输出结果"""

    regime: str                  # Regime.TREND / OSCILLATION / UNKNOWN
    confidence: float            # 0.0 ~ 1.0
    score: float                 # 综合分数 [-1, +1], 正->趋势，负->震荡

    adx_score: float = 0.0
    bb_width_score: float = 0.0
    atr_ratio_score: float = 0.0
    range_break_score: float = 0.0

    adx_value: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    bb_width_pct: float = 0.0
    atr_ratio: float = 1.0

    fast_path: bool = False
    osc_range: Optional[OscillationRange] = None

    def __str__(self) -> str:
        filled = int(self.confidence * 10)
        bars = "#" * filled + "-" * (10 - filled)
        base = (
            f"[{self.regime:<12}]  score={self.score:+.3f}  "
            f"conf={self.confidence:.2f} {bars}\n"
            f"  ADX={self.adx_value:.1f}(score={self.adx_score:+.2f})  "
            f"BB%={self.bb_width_pct:.2f}(score={self.bb_width_score:+.2f})  "
            f"ATR_ratio={self.atr_ratio:.2f}(score={self.atr_ratio_score:+.2f})  "
            f"range_break(score={self.range_break_score:+.2f})"
            + ("  [FAST]" if self.fast_path else "")
        )
        if self.osc_range is not None:
            base += "\n" + str(self.osc_range)
        return base


# ══════════════════════════════════════════════════════════════════════════════
#  RegimeAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class RegimeAnalyzer:
    """
    盘型分析器（规则引擎）。

    Usage
    -----
    analyzer = RegimeAnalyzer()
    result   = analyzer.analyze(df)   # df 含 open/high/low/close 列
    print(result)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.cfg = config or DEFAULT_CONFIG
        self._ic  = self.cfg.indicator
        self._sc  = self.cfg.scoring
        self._log = get_logger(__name__, self.cfg.log_level)

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """
        输入标准化 OHLCV DataFrame，输出盘型分析结果。

        Parameters
        ----------
        df : pd.DataFrame
            必须包含列：open, high, low, close
            可选列   ：volume, time
            按时间升序排列（最新数据在末尾）
        """
        n = len(df)
        if n < self.cfg.min_bars:
            self._log.warning(f"Insufficient data: {n} < min_bars={self.cfg.min_bars}")
            return AnalysisResult(regime=Regime.UNKNOWN, confidence=0.0, score=0.0)

        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        # 指标计算
        adx_arr, pdi_arr, mdi_arr = calc_adx(high, low, close, self._ic.adx_period)
        _, _, _, bb_width_arr     = calc_bb(close, self._ic.bb_period, self._ic.bb_std_mult)
        atr_arr                   = calc_atr(high, low, close, self._ic.atr_period)

        latest_adx  = float(adx_arr[-1])
        latest_pdi  = float(pdi_arr[-1])
        latest_mdi  = float(mdi_arr[-1])

        # 快速通道（极端 ADX 直接判断）
        if self.cfg.enable_fast_path:
            if latest_adx > self._sc.fast_adx_trend:
                return AnalysisResult(
                    regime=Regime.TREND,
                    confidence=min(0.95, 0.7 + (latest_adx - self._sc.fast_adx_trend) / 100),
                    score=1.0, adx_score=1.0,
                    adx_value=latest_adx, plus_di=latest_pdi, minus_di=latest_mdi,
                    fast_path=True,
                )
            if latest_adx < self._sc.fast_adx_osc:
                return AnalysisResult(
                    regime=Regime.OSCILLATION,
                    confidence=min(0.95, 0.7 + (self._sc.fast_adx_osc - latest_adx) / 40),
                    score=-1.0, adx_score=-1.0,
                    adx_value=latest_adx, plus_di=latest_pdi, minus_di=latest_mdi,
                    fast_path=True,
                )

        # 多因子评分（以下为参考实现，请根据你的策略调整）
        adx_score        = self._score_adx(latest_adx)
        bb_width_score, bb_pct = self._score_bb_width(bb_width_arr)
        atr_ratio_score, atr_ratio = self._score_atr_ratio(atr_arr)
        range_score      = self._score_range_break(close, high, low)

        # 加权综合
        w = self._sc.weights
        composite = (
            w["adx"]         * adx_score +
            w["bb_width"]    * bb_width_score +
            w["atr_ratio"]   * atr_ratio_score +
            w["range_break"] * range_score
        )
        composite = float(np.clip(composite, -1.0, 1.0))

        # 置信度
        factor_scores = np.array([adx_score, bb_width_score, atr_ratio_score, range_score])
        confidence = self._calc_confidence(factor_scores, composite)

        # 最终判断
        regime = self._classify(composite, confidence)

        result = AnalysisResult(
            regime=regime, confidence=confidence, score=composite,
            adx_score=adx_score, bb_width_score=bb_width_score,
            atr_ratio_score=atr_ratio_score, range_break_score=range_score,
            adx_value=latest_adx, plus_di=latest_pdi, minus_di=latest_mdi,
            bb_width_pct=bb_pct, atr_ratio=atr_ratio,
        )
        self._log.info(str(result))
        return result

    # ── 因子评分函数（参考实现，可自行替换）──────────────────────────

    def _score_adx(self, adx: float) -> float:
        """ADX -> [-1, +1]。ADX >= trend_th -> 趋势, ADX <= osc_th -> 震荡。"""
        osc_th   = self._ic.adx_osc_threshold
        trend_th = self._ic.adx_trend_threshold
        if adx >= trend_th:
            return min(1.0, (adx - trend_th) / 15.0)
        if adx <= osc_th:
            return max(-1.0, -(osc_th - adx) / 10.0)
        mid = (osc_th + trend_th) / 2.0
        return (adx - mid) / ((trend_th - osc_th) / 2.0) * 0.5

    def _score_bb_width(self, bb_width: np.ndarray) -> tuple[float, float]:
        """BB 宽度分位数 -> [-1, +1]。宽 -> 趋势, 窄 -> 震荡。"""
        pct = calc_percentile_rank(bb_width, self._ic.bb_width_lookback)
        score = (pct - 0.5) * 2.0
        return float(np.clip(score, -1.0, 1.0)), pct

    def _score_atr_ratio(self, atr: np.ndarray) -> tuple[float, float]:
        """当前 ATR / 历史均值 -> [-1, +1]。扩张 -> 趋势, 收缩 -> 震荡。"""
        valid = atr[atr > 1e-10]
        if len(valid) < self._ic.atr_lookback:
            return 0.0, 1.0
        hist = valid[-self._ic.atr_lookback - 1 : -1]
        if len(hist) == 0:
            return 0.0, 1.0
        hist_mean = float(np.mean(hist))
        current   = float(valid[-1])
        ratio = current / hist_mean if hist_mean > 1e-10 else 1.0
        high_r = self._ic.atr_high_ratio
        low_r  = self._ic.atr_low_ratio
        if ratio >= high_r:
            return min(1.0, (ratio - high_r) / 0.5), ratio
        if ratio <= low_r:
            return max(-1.0, -(low_r - ratio) / 0.3), ratio
        mid = (high_r + low_r) / 2.0
        score = (ratio - mid) / ((high_r - low_r) / 2.0) * 0.5
        return float(np.clip(score, -1.0, 1.0)), ratio

    def _score_range_break(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
        """区间突破 -> [-1, 0, +1]。突破 = 趋势, 区间内 = 震荡。"""
        result = calc_range_break(close, high, low, self._ic.range_lookback)
        return 1.0 if result != 0.0 else -0.5

    @staticmethod
    def _calc_confidence(factor_scores: np.ndarray, composite: float) -> float:
        """置信度 = 因子一致性 + 综合分数绝对值。"""
        if abs(composite) < 1e-6:
            return 0.0
        sign = np.sign(composite)
        agreement = float(np.mean(np.sign(factor_scores) == sign))
        magnitude = min(1.0, abs(composite) * 1.5)
        confidence = 0.5 * agreement + 0.5 * magnitude
        return float(np.clip(confidence, 0.0, 1.0))

    def _classify(self, score: float, confidence: float) -> str:
        """根据综合分数和置信度输出最终盘型。"""
        if confidence < self._sc.confidence_threshold:
            return Regime.UNKNOWN
        if score > self._sc.trend_score_threshold:
            return Regime.TREND
        if score < -self._sc.osc_score_threshold:
            return Regime.OSCILLATION
        return Regime.UNKNOWN
