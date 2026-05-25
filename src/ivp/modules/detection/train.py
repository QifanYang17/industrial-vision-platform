"""目标检测训练入口（阶段二骨架，暂未实现）。"""

from __future__ import annotations

from omegaconf import DictConfig

from ivp.common.logging import get_logger

logger = get_logger(__name__)


def train(cfg: DictConfig) -> None:
    """训练目标检测模型（占位，阶段二实现）。"""
    logger.warning("【骨架占位】目标检测（YOLO26+ByteTrack）将在阶段二实现")
