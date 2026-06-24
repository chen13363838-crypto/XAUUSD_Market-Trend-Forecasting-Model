"""
jud_model/config/settings.py
盘型判断 Agent 全局参数配置

所有阈值和权重均为示例占位值。
实际使用时应根据你的品种和回测结果进行调优。
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class IndicatorConfig:
    """技术指标参数（示例值，需根据品种调优）"""

    # ADX
    adx_period: int = 14
    adx_trend_threshold: float = 25.0   # ADX > 此值 -> 趋势信号
    adx_osc_threshold: float = 20.0     # ADX < 此值 -> 震荡信号

    # 布林带
    bb_period: int = 20
    bb_std_mult: float = 2.0
    bb_width_lookback: int = 100

    # ATR
    atr_period: int = 14
    atr_lookback: int = 50
    atr_high_ratio: float = 1.2
    atr_low_ratio: float = 0.8

    # 价格区间突破
    range_lookback: int = 50


@dataclass
class ScoringConfig:
    """评分与置信度参数（示例值，需根据策略调优）"""

    # 各因子权重（加权求和，总和须等于 1.0）
    weights: Dict[str, float] = field(default_factory=lambda: {
        "adx":         0.40,
        "bb_width":    0.30,
        "atr_ratio":   0.20,
        "range_break": 0.10,
    })

    # 综合分数阈值
    trend_score_threshold: float = 0.25
    osc_score_threshold: float = 0.25

    # 置信度低于此值时输出 UNKNOWN
    confidence_threshold: float = 0.55

    # 快速判断阈值
    fast_adx_trend: float = 32.0
    fast_adx_osc: float = 15.0


@dataclass
class AgentConfig:
    """Agent 全局参数"""

    min_bars: int = 120
    enable_fast_path: bool = True
    log_level: str = "INFO"

    indicator: IndicatorConfig = field(default_factory=IndicatorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


DEFAULT_CONFIG = AgentConfig()
