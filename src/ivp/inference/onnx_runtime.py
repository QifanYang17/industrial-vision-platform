"""ONNX Runtime 推理封装（骨架）。

为什么用 ONNX：训练用 torch，部署导出成 ONNX 可获得跨框架、更快、更省内存的
推理，并便于报告「原生 vs ONNX/量化」的延迟对比（CLAUDE.md 指标要求）。
onnxruntime 属 service 可选依赖，故在方法内 guarded import。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ivp.inference.base import BaseInferencer, InferenceResult


class OnnxInferencer(BaseInferencer):
    """通用 ONNX 推理器骨架，各模块可继承后实现前/后处理。"""

    def __init__(self, model_path: str | Path, providers: list[str] | None = None) -> None:
        """记录模型路径与执行 provider，延迟到 load() 才建会话。"""
        self.model_path = Path(model_path)
        # CPUExecutionProvider 在 macOS 通用；CUDA 机器可换 CUDAExecutionProvider
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None

    def load(self) -> None:  # noqa: D102
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)

    def predict(self, image: np.ndarray) -> InferenceResult:  # noqa: D102
        raise NotImplementedError("阶段一骨架：前/后处理待各模块继承实现")
