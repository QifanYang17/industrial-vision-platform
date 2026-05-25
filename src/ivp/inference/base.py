"""统一推理接口（抽象基类）。

为什么要抽象基类：service 层只依赖这个稳定契约，三个模块各自实现，
新增模块/切换 ONNX 不影响调用方——这是「可插拔模块」的接缝所在。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InferenceResult:
    """统一的推理结果容器。

    各模块按需填充：
    - 缺陷检测：score=异常分, label=OK/NG, heatmap=像素级异常图
    - 检测+计数：detections=[(bbox, cls, conf), ...], count=去重计数
    - OCR：text=识别文本, regions=文本框
    """

    label: str | None = None
    score: float | None = None
    latency_ms: float | None = None  # 延迟：每个模型都要报告，对应 CLAUDE.md 指标要求
    extra: dict[str, Any] = field(default_factory=dict)


class BaseInferencer(ABC):
    """所有任务模块推理器的统一接口。"""

    @abstractmethod
    def load(self) -> None:
        """加载权重/模型到内存（含设备搬运）。"""

    @abstractmethod
    def predict(self, image: np.ndarray) -> InferenceResult:
        """对单张图像推理，返回统一结果。"""
