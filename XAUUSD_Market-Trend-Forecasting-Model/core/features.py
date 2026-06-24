"""
jud_model/core/features.py
Feature Engineering -- extract feature vectors from OHLCV for ML models

Feature dimensions (9 total):
    [0] log_return      Log return
    [1] volatility      Rolling std of log returns
    [2] bb_width        Normalized Bollinger Band width
    [3] atr_ratio       Current ATR / historical ATR mean
    [4] adx_norm        ADX / 100
    [5] di_diff         (+DI - -DI) / 100
    [6] range_position  Close position within recent high-low range
    [7] cmf_norm        Chaikin Money Flow (-1 ~ +1)
    [8] macd_hist_norm  Normalized MACD histogram
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from core.indicators import calc_atr, calc_adx, calc_bb, calc_cmf, calc_macd_hist_norm

IDX_LOG_RETURN     = 0
IDX_VOLATILITY     = 1
IDX_BB_WIDTH       = 2
IDX_ATR_RATIO      = 3
IDX_ADX_NORM       = 4
IDX_DI_DIFF        = 5
IDX_RANGE_POSITION = 6
IDX_CMF_NORM       = 7
IDX_MACD_HIST_NORM = 8

FEATURE_NAMES = [
    "log_return", "volatility", "bb_width", "atr_ratio",
    "adx_norm", "di_diff", "range_position", "cmf_norm", "macd_hist_norm",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(
    df: pd.DataFrame,
    vol_window: int = 20,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    atr_lookback: int = 50,
    adx_period: int = 14,
    range_lookback: int = 50,
    cmf_period: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix from standardized OHLCV DataFrame."""
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    n     = len(close)

    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(np.maximum(close[1:] / close[:-1], 1e-10))

    volatility = np.full(n, np.nan)
    for i in range(vol_window, n):
        volatility[i] = float(np.std(log_ret[i - vol_window + 1 : i + 1], ddof=1))

    _, _, _, bb_width = calc_bb(close, bb_period, bb_std)

    atr = calc_atr(high, low, close, atr_period)
    atr_ratio = np.full(n, np.nan)
    for i in range(atr_lookback, n):
        hist = atr[i - atr_lookback : i]
        hist_mean = float(np.mean(hist[hist > 1e-10]))
        if hist_mean > 1e-10 and atr[i] > 1e-10:
            atr_ratio[i] = atr[i] / hist_mean
        else:
            atr_ratio[i] = 1.0

    adx_arr, pdi_arr, mdi_arr = calc_adx(high, low, close, adx_period)
    adx_norm = adx_arr / 100.0
    di_diff  = (pdi_arr - mdi_arr) / 100.0

    range_pos = np.full(n, np.nan)
    for i in range(range_lookback, n):
        h_max = np.max(high[i - range_lookback : i + 1])
        l_min = np.min(low[i - range_lookback : i + 1])
        span  = h_max - l_min
        range_pos[i] = (close[i] - l_min) / span if span > 1e-10 else 0.5

    if "volume" in df.columns:
        vol = df["volume"].values.astype(float)
    else:
        vol = np.ones(n)
    cmf_norm = calc_cmf(high, low, close, vol, period=cmf_period)

    macd_hist_norm = calc_macd_hist_norm(
        close, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal
    )

    X = np.column_stack([
        log_ret, volatility, bb_width, atr_ratio,
        adx_norm, di_diff, range_pos, cmf_norm, macd_hist_norm,
    ])

    valid_mask = ~np.any(np.isnan(X), axis=1)
    return X, valid_mask


def features_to_dataframe(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=FEATURE_NAMES)
