"""OCR 训练/微调入口（阶段三骨架，暂未实现）。"""

from __future__ import annotations

from omegaconf import DictConfig

from ivp.common.logging import get_logger

logger = get_logger(__name__)


def train(cfg: DictConfig) -> None:
    """OCR 管道训练/微调（占位，阶段三实现）。"""
    logger.warning("【骨架占位】OCR 管道将在阶段三实现")
