"""
jud_model/models/base_model.py
所有可学习盘型模型的抽象基类接口

继承此类并实现全部抽象方法即可接入 Agent 框架。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from core.analyzer import AnalysisResult


class BaseRegimeModel(ABC):
    """
    可学习盘型模型的统一接口。

    所有模型实现（GMM、HMM、KMeans 等）必须继承此类并实现全部抽象方法。
    """

    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseRegimeModel":
        """
        用历史 K 线数据训练模型。

        Parameters
        ----------
        df : pd.DataFrame
            标准化 OHLCV DataFrame（来自 DataFeeder），建议 >= 500 根

        Returns
        -------
        self（支持链式调用）
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> AnalysisResult:
        """
        对当前 K 线序列做盘型推理。

        Returns
        -------
        AnalysisResult
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """将训练好的模型序列化到文件。"""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseRegimeModel":
        """从文件反序列化模型。"""
        ...

    def fit_predict(self, df: pd.DataFrame) -> AnalysisResult:
        """训练 + 立即预测（方便一次性调用）。"""
        self.fit(df)
        return self.predict(df)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}({status})"
