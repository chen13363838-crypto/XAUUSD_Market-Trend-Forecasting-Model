"""
jud_model/utils/logger.py
统一日志工具
"""

import logging
import sys


def get_logger(name: str = "jud_model", level: str = "INFO") -> logging.Logger:
    """
    获取格式化 Logger。

    Parameters
    ----------
    name  : logger 名称，通常传入 __name__
    level : 日志级别字符串 DEBUG / INFO / WARNING / ERROR

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    return logger
