"""FastAPI 推理服务（阶段一骨架：仅健康检查，推理端点待接入各模块）。

用工厂函数 create_app() 而非模块级 app，便于测试注入与多配置启动。
启动：uvicorn "ivp.service.api:create_app" --factory --reload
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ivp import __version__

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app() -> FastAPI:
    """构建 FastAPI 应用（fastapi 属 service 组，运行期 import）。"""
    from fastapi import FastAPI

    from ivp.service.schemas import HealthResponse

    app = FastAPI(title="工业视觉统一平台 API", version=__version__)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    # TODO: /predict/{module} 端点——加载对应 Inferencer 并返回 PredictResponse

    return app
