"""统一日志配置（基于 rich，输出彩色、带级别、可读性优先）。"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: int | str = logging.INFO) -> None:
    """全局只配置一次根 logger，使用 RichHandler 输出。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取带统一配置的 logger。"""
    setup_logging()
    return logging.getLogger(name)
