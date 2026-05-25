"""缺陷检测推理（阶段一骨架）。

待实现：实现 BaseInferencer，加载 anomalib 导出的权重/ONNX，
predict() 返回异常分 + OK/NG 标签 + 像素级 heatmap + 延迟。
"""

from __future__ import annotations

import numpy as np

from ivp.inference.base import BaseInferencer, InferenceResult


class DefectInferencer(BaseInferencer):
    """缺陷检测推理器（占位骨架）。"""

    def load(self) -> None:  # noqa: D102
        raise NotImplementedError("阶段一骨架：待接入 anomalib 权重加载")

    def predict(self, image: np.ndarray) -> InferenceResult:  # noqa: D102
        raise NotImplementedError("阶段一骨架：待实现异常分/heatmap 推理")
