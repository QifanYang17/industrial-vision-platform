"""service 层的请求/响应数据模型（pydantic，属核心依赖可直接导入）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"
    version: str


class PredictResponse(BaseModel):
    """统一推理响应（对应 inference.InferenceResult）。"""

    module: str = Field(description="任务模块：defect/detection/ocr")
    label: str | None = None
    score: float | None = None
    latency_ms: float | None = None
