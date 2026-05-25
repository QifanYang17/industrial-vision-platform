# 服务镜像（FastAPI + Gradio）。骨架阶段先占位，进入部署阶段再细化。
# 为什么用 uv：镜像内复用同一锁文件，保证「本地=容器」依赖一致、构建快。
FROM python:3.11-slim AS base

# 装 uv（官方静态二进制，体积小）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先只拷依赖声明，命中 Docker 层缓存：依赖没变就不重装
COPY pyproject.toml uv.lock ./
# 只装服务运行所需（service 组）+ 不装项目自身的可编辑包，加快构建
RUN uv sync --frozen --extra service --no-install-project

# 再拷源码并安装项目本身
COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --frozen --extra service

EXPOSE 8000
# 默认起 FastAPI 推理服务；Gradio 演示见 docker-compose 的 demo service
CMD ["uv", "run", "uvicorn", "ivp.service.api:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
