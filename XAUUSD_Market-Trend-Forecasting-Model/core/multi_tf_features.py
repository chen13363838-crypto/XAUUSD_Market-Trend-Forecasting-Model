"""
jud_model/core/multi_tf_features.py
Multi-Timeframe Feature Fusion -- align and merge M5/M15/M30 features

Alignment strategy (M30 as base):
    For each M30 bar[i] covering window [M30.time[i-1], M30.time[i]]:
        1. Take M30 features directly (9 dims)
        2. Aggregate M15 bars in window -> mean (6 dims)
        3. Aggregate M5 bars in window -> mean (6 dims)

Final feature vector: 9 + 6 + 6 = 21 dims

Note: This is the framework version. The specific feature selection
and aggregation strategy can be customized.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.features import (
    extract_features,
    FEATURE_NAMES,
    N_FEATURES,
    IDX_LOG_RETURN,
    IDX_VOLATILITY,
    IDX_ADX_NORM,
    IDX_BB_WIDTH,
    IDX_ATR_RATIO,
    IDX_MACD_HIST_NORM,
)

# Features suitable for cross-timeframe aggregation
AGG_FEATURE_INDICES = [
    IDX_LOG_RETURN, IDX_VOLATILITY, IDX_ADX_NORM,
    IDX_BB_WIDTH, IDX_ATR_RATIO, IDX_MACD_HIST_NORM,
]
AGG_FEATURE_NAMES = [FEATURE_NAMES[i] for i in AGG_FEATURE_INDICES]
N_AGG = len(AGG_FEATURE_INDICES)

MTF_FEATURE_NAMES = (
    [f"m30_{n}" for n in FEATURE_NAMES] +
    [f"m15_{n}_mean" for n in AGG_FEATURE_NAMES] +
    [f"m5_{n}_mean" for n in AGG_FEATURE_NAMES]
)
N_MTF_FEATURES = N_FEATURES + N_AGG + N_AGG  # 21


def extract_multi_tf_features(
    tf_data: Dict[str, pd.DataFrame],
    feature_windows: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fused multi-timeframe feature matrix.

    Parameters
    ----------
    tf_data : dict
        Keys: "M5"/"M15"/"M30", values: standard OHLCV DataFrames.
        Must contain "M30" (base timeframe).
        DataFrames must have a "datetime" column.

    Returns
    -------
    X         : np.ndarray, shape (n_m30, N_MTF_FEATURES)
    valid_mask : np.ndarray of bool
    """
    fw = feature_windows or {}

    if "M30" not in tf_data:
        raise ValueError("tf_data must contain M30 (base timeframe)")

    # Extract features for each timeframe
    feat_dfs: Dict[str, pd.DataFrame] = {}
    for tf_label, df in tf_data.items():
        X, _ = extract_features(df, **fw)
        feat_dfs[tf_label] = pd.DataFrame(
            X, index=pd.to_datetime(df["datetime"]), columns=FEATURE_NAMES
        )

    # Build fused features per M30 bar
    base_df  = tf_data["M30"]
    base_idx = pd.to_datetime(base_df["datetime"])

    all_rows: List[List[float]] = []

    for i in range(len(base_idx)):
        if i == 0:
            all_rows.append([np.nan] * N_MTF_FEATURES)
            continue

        t_start = base_idx[i - 1]
        t_end   = base_idx[i]

        m30_feat = feat_dfs["M30"].iloc[i].values.copy()

        if "M15" in feat_dfs:
            m15_slice = feat_dfs["M15"].loc[
                (feat_dfs["M15"].index > t_start) &
                (feat_dfs["M15"].index <= t_end)
            ]
            m15_agg = m15_slice.iloc[:, AGG_FEATURE_INDICES].mean(axis=0).values if len(m15_slice) > 0 else np.full(N_AGG, np.nan)
        else:
            m15_agg = np.full(N_AGG, np.nan)

        if "M5" in feat_dfs:
            m5_slice = feat_dfs["M5"].loc[
                (feat_dfs["M5"].index > t_start) &
                (feat_dfs["M5"].index <= t_end)
            ]
            m5_agg = m5_slice.iloc[:, AGG_FEATURE_INDICES].mean(axis=0).values if len(m5_slice) > 0 else np.full(N_AGG, np.nan)
        else:
            m5_agg = np.full(N_AGG, np.nan)

        fused = np.concatenate([m30_feat, m15_agg, m5_agg])
        all_rows.append(fused)

    X = np.array(all_rows)
    valid_mask = ~np.any(np.isnan(X), axis=1)
    return X, valid_mask


def mtf_features_to_dataframe(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=MTF_FEATURE_NAMES)
