"""
jud_model/core/indicators.py
纯 NumPy 实现的技术指标库（零 ta-lib 依赖）

提供函数：
    calc_atr(high, low, close, period) -> np.ndarray
    calc_adx(high, low, close, period) -> (adx, plus_di, minus_di)
    calc_bb (close, period, std_mult)  -> (upper, mid, lower, width)
    calc_percentile_rank(series, lookback) -> float
    calc_range_break(close, high, low, lookback) -> float
    calc_cmf(high, low, close, volume, period) -> np.ndarray
    calc_macd(close, fast, slow, signal) -> (macd_line, signal, histogram)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ══════════════════════════════════════════════════════════════════
#  ATR (Average True Range)
# ══════════════════════════════════════════════════════════════════

def calc_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Wilder 平滑 ATR。

    Returns
    -------
    atr : np.ndarray, shape (n,)
        前 period-1 根为 0（数据不足）
    """
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.zeros(n)
    if n < period:
        return atr

    atr[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha

    return atr


# ══════════════════════════════════════════════════════════════════
#  ADX (+DI / -DI)
# ══════════════════════════════════════════════════════════════════

def calc_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wilder ADX。

    Returns
    -------
    adx      : np.ndarray  -- 趋势强度 0~100
    plus_di  : np.ndarray  -- +DI
    minus_di : np.ndarray  -- -DI
    """
    n = len(close)
    tr       = np.zeros(n)
    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)

    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        up_move   = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    alpha = 1.0 / period
    s_tr  = np.zeros(n)
    s_pdm = np.zeros(n)
    s_mdm = np.zeros(n)

    if n <= period:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    s_tr[period]  = np.sum(tr[1 : period + 1])
    s_pdm[period] = np.sum(plus_dm[1 : period + 1])
    s_mdm[period] = np.sum(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        s_tr[i]  = s_tr[i - 1]  * (1 - alpha) + tr[i]
        s_pdm[i] = s_pdm[i - 1] * (1 - alpha) + plus_dm[i]
        s_mdm[i] = s_mdm[i - 1] * (1 - alpha) + minus_dm[i]

    plus_di  = np.zeros(n)
    minus_di = np.zeros(n)
    dx       = np.zeros(n)

    for i in range(period, n):
        if s_tr[i] > 1e-10:
            plus_di[i]  = 100.0 * s_pdm[i] / s_tr[i]
            minus_di[i] = 100.0 * s_mdm[i] / s_tr[i]
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    adx = np.zeros(n)
    start = 2 * period
    if n <= start:
        return adx, plus_di, minus_di

    adx[start - 1] = np.mean(dx[period:start])
    for i in range(start, n):
        adx[i] = adx[i - 1] * (1 - alpha) + dx[i] * alpha

    return adx, plus_di, minus_di


# ══════════════════════════════════════════════════════════════════
#  Bollinger Bands
# ══════════════════════════════════════════════════════════════════

def calc_bb(
    close: np.ndarray,
    period: int = 20,
    std_mult: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    布林带 + 归一化宽度。

    Returns
    -------
    upper, mid, lower : np.ndarray
    width             : np.ndarray  -- (upper-lower)/mid
    """
    n = len(close)
    upper = np.full(n, np.nan)
    mid   = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    width = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        m = float(np.mean(window))
        s = float(np.std(window, ddof=0))
        mid[i]   = m
        upper[i] = m + std_mult * s
        lower[i] = m - std_mult * s
        if m > 1e-10:
            width[i] = (upper[i] - lower[i]) / m

    return upper, mid, lower, width


# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def calc_percentile_rank(series: np.ndarray, lookback: int) -> float:
    """
    计算 series 最后一个值在过去 lookback 根内的百分位排名。

    Returns
    -------
    float : 0.0 ~ 1.0
    """
    valid = series[~np.isnan(series)]
    if len(valid) < 2:
        return 0.5

    window = valid[-lookback:]
    current = window[-1]
    rank = float(np.sum(window <= current)) / len(window)
    return rank


def calc_cmf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """
    Chaikin Money Flow (CMF)。

    Returns
    -------
    cmf : np.ndarray, range [-1, +1]
        > 0 = 资金净流入，< 0 = 净流出
    """
    n = len(close)
    cmf = np.full(n, np.nan)
    if n < period or period < 1:
        return cmf

    denom = high - low
    with np.errstate(divide='ignore', invalid='ignore'):
        mfm = np.where(
            denom > 1e-10,
            ((close - low) - (high - close)) / denom,
            0.0,
        )

    mf_volume = mfm * volume.astype(float)

    for i in range(period - 1, n):
        start = i - period + 1
        vol_sum = float(np.sum(volume[start : i + 1]))
        if vol_sum > 1e-10:
            cmf[i] = float(np.sum(mf_volume[start : i + 1])) / vol_sum
        else:
            cmf[i] = 0.0

    return cmf


def calc_range_break(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 50,
) -> float:
    """
    判断当前收盘价是否突破近期高低区间。

    Returns
    -------
    float : +1.0 向上突破, 0.0 区间内, -1.0 向下突破
    """
    if len(close) < lookback + 1:
        return 0.0

    hist_high = np.max(high[-lookback - 1 : -1])
    hist_low  = np.min(low[-lookback - 1 : -1])
    current   = close[-1]

    if current > hist_high:
        return 1.0
    if current < hist_low:
        return -1.0
    return 0.0


# ══════════════════════════════════════════════════════════════════
#  MACD
# ══════════════════════════════════════════════════════════════════

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """计算 EMA，前 period-1 根返回 NaN。支持输入含 NaN。"""
    n = len(arr)
    ema = np.full(n, np.nan)
    if n < period:
        return ema

    first_valid = 0
    while first_valid < n and np.isnan(arr[first_valid]):
        first_valid += 1
    if first_valid + period > n:
        return ema

    end_init = first_valid + period
    ema[end_init - 1] = float(np.mean(arr[first_valid:end_init]))
    alpha = 2.0 / (period + 1.0)
    for i in range(end_init, n):
        if not np.isnan(arr[i]):
            ema[i] = arr[i] * alpha + ema[i - 1] * (1.0 - alpha)
        else:
            ema[i] = ema[i - 1]
    return ema


def calc_macd(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD -- 纯 NumPy 实现。

    Returns
    -------
    macd_line  : EMA(fast) - EMA(slow)
    signal     : EMA of macd_line
    histogram  : macd_line - signal
    """
    fast_ema  = _ema(close, fast_period)
    slow_ema  = _ema(close, slow_period)
    macd_line  = fast_ema - slow_ema
    signal     = _ema(macd_line, signal_period)
    histogram  = macd_line - signal
    return macd_line, signal, histogram


def calc_macd_hist_norm(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> np.ndarray:
    """MACD Histogram 归一化（除以收盘价，使不同价格级别可比）。"""
    _, _, hist = calc_macd(close, fast_period, slow_period, signal_period)
    denom = np.where(close > 1e-10, close, 1.0)
    return hist / denom
