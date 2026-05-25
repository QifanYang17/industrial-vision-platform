"""设备选择：优先 M4 的 MPS 后端，按可用性回退。

CLAUDE.md 硬件约束：开发机 M4 / 16GB，PyTorch 走 MPS 后端吃 GPU 加速。
torch 为可选依赖，未安装时返回偏好值并提示——保证基础环境下导入不报错。
"""

from __future__ import annotations

from ivp.common.logging import get_logger

logger = get_logger(__name__)

VALID_DEVICES = {"mps", "cuda", "cpu"}


def resolve_device(prefer: str = "mps") -> str:
    """解析实际可用设备字符串。

    Args:
        prefer: 偏好设备（mps/cuda/cpu）。

    Returns:
        实际可用的设备字符串；不可用时按 mps→cuda→cpu 顺序回退。
    """
    if prefer not in VALID_DEVICES:
        raise ValueError(f"未知设备 {prefer!r}，应为 {VALID_DEVICES} 之一")

    try:
        import torch
    except ImportError:
        logger.warning("torch 未安装，无法探测设备可用性，按偏好返回 %r", prefer)
        return prefer

    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer != "cpu":
        logger.warning("偏好设备 %r 不可用，回退到 cpu", prefer)
    return "cpu"
