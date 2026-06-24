"""
jud_model/data/feed.py
Data Feeder -- 数据投喂入口

提供多种数据注入方式，统一转换为标准 DataFrame 后交给 Agent 使用。

支持的投喂方式
--------------
1. feed_from_list()      -- 字典列表（最通用）
2. feed_from_dataframe() -- pandas DataFrame
3. feed_from_csv()       -- CSV 文件（MT5 导出 / 自定义格式）
4. feed_from_mt5()       -- 直接从 MT5 终端拉取（需安装 MetaTrader5 包）
5. feed_single_bar()     -- 实时逐根投喂（在线模式）

标准列名
--------
    time    : datetime64[ns] 或 str（可选，有则自动转换）
    open    : float
    high    : float
    low     : float
    close   : float
    volume  : float（可选）

列名大小写不敏感，会自动规范化。
"""

from __future__ import annotations

import os
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils.logger import get_logger

_log = get_logger(__name__)

# 必须包含的列
_REQUIRED_COLS = {"open", "high", "low", "close"}
# 自动识别的列名别名
_COL_ALIASES: Dict[str, str] = {
    "date":         "time",
    "datetime":     "time",
    "timestamp":    "time",
    "bar_time":     "time",
    "o":            "open",
    "h":            "high",
    "l":            "low",
    "c":            "close",
    "last":         "close",
    "vol":          "volume",
    "tick_volume":  "volume",
    "real_volume":  "volume",
    "v":            "volume",
}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """规范化列名 + 数据类型，返回干净的 DataFrame。"""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.rename(columns=_COL_ALIASES, inplace=True)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Current columns: {list(df.columns)}"
        )

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.sort_values("time", inplace=True)

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


class DataFeeder:
    """
    数据投喂器 -- 数据注入的统一入口。

    Parameters
    ----------
    buffer_size : int
        实时模式下内部滚动缓冲区的最大容量（根 K 线数量）。
        0 = 不限制（全量保存）。
    """

    def __init__(self, buffer_size: int = 0):
        self._buffer_size = buffer_size
        self._live_buffer: deque[Dict[str, Any]] = deque(
            maxlen=buffer_size if buffer_size > 0 else None
        )

    def feed_from_list(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """投喂字典列表，每个字典代表一根 K 线。"""
        if not data:
            raise ValueError("data list is empty")
        df = pd.DataFrame(data)
        result = _normalize_df(df)
        _log.info(f"[feed_from_list] Fed {len(result)} bars")
        return result

    def feed_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """投喂已有的 pandas DataFrame。列名不敏感，支持别名。"""
        result = _normalize_df(df)
        _log.info(f"[feed_from_dataframe] Fed {len(result)} bars")
        return result

    def feed_from_csv(
        self,
        file_path: str,
        sep: str = ",",
        encoding: str = "utf-8",
        **kwargs,
    ) -> pd.DataFrame:
        """
        从 CSV 文件投喂。支持 MT5 导出格式（制表符分隔）和标准 CSV。

        Parameters
        ----------
        file_path : str
        sep       : str, 分隔符。MT5 通常为 '\\t'；标准 CSV 为 ','。
        encoding  : str, 默认 utf-8，若乱码可尝试 'gbk'。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path, sep=sep, encoding=encoding, **kwargs)
        result = _normalize_df(df)
        _log.info(f"[feed_from_csv] Fed {len(result)} bars from {file_path}")
        return result

    def feed_from_mt5(
        self,
        symbol: str = "XAUUSD",
        timeframe: int = None,
        bars: int = 500,
        from_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        直接从本地 MT5 终端拉取历史 K 线。

        依赖: pip install MetaTrader5
        仅支持 Windows 且已安装 MT5 终端。
        """
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install MetaTrader5: pip install MetaTrader5\n"
                "(Windows only, requires MT5 terminal)"
            )

        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15

        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

        try:
            if from_date is not None:
                rates = mt5.copy_rates_from(symbol, timeframe, from_date, bars)
            else:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

            if rates is None or len(rates) == 0:
                raise RuntimeError(f"Failed to get {symbol} data: {mt5.last_error()}")

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            result = _normalize_df(df)
            _log.info(f"[feed_from_mt5] Pulled {len(result)} {symbol} bars from MT5")
            return result

        finally:
            mt5.shutdown()

    def feed_single_bar(self, bar: Dict[str, Any]) -> pd.DataFrame:
        """实时投喂单根 K 线（追加到内部缓冲区）。"""
        self._live_buffer.append(bar)
        result = self.feed_from_list(list(self._live_buffer))
        _log.debug(f"[feed_single_bar] Buffer size: {len(result)}")
        return result

    def clear_buffer(self) -> None:
        """清空实时缓冲区"""
        self._live_buffer.clear()

    @property
    def buffer_length(self) -> int:
        return len(self._live_buffer)

    @staticmethod
    def make_synthetic(
        n_bars: int = 500,
        mode: str = "trend",
        base_price: float = 2000.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        生成合成 K 线数据，用于框架验证（不依赖真实数据源）。

        Parameters
        ----------
        n_bars      : K 线根数
        mode        : 'trend' / 'oscillation' / 'mixed'
        base_price  : 起始价格
        seed        : 随机种子
        """
        rng = np.random.default_rng(seed)
        prices = [base_price]

        if mode == "trend":
            drift, vol = 0.3, 1.5
        elif mode == "oscillation":
            drift, vol = 0.0, 0.8
        else:
            drift, vol = 0.0, 0.8

        for i in range(1, n_bars):
            if mode == "mixed" and i > n_bars // 2:
                drift, vol = 0.25, 1.8
            change = drift + rng.normal(0, vol)
            prices.append(max(prices[-1] + change, 1.0))

        records = []
        times = pd.date_range("2023-01-01", periods=n_bars, freq="15min")
        for i, p in enumerate(prices):
            h = p + abs(rng.normal(0, 0.5))
            l = p - abs(rng.normal(0, 0.5))
            o = p + rng.normal(0, 0.3)
            records.append({
                "time":   times[i],
                "open":   round(o, 2),
                "high":   round(max(o, h, p), 2),
                "low":    round(min(o, l, p), 2),
                "close":  round(p, 2),
                "volume": int(rng.integers(500, 5000)),
            })

        feeder = DataFeeder()
        return feeder.feed_from_list(records)
