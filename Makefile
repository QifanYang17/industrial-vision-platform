# 常用开发命令的统一入口。`make help` 查看全部。
.DEFAULT_GOAL := help
.PHONY: help setup lint format test train serve-api serve-demo mlflow-ui clean

help:  ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## 创建虚拟环境并安装基础 + 开发依赖
	uv sync --group dev
	uv run pre-commit install

lint:  ## ruff 静态检查
	uv run ruff check .

format:  ## ruff 自动格式化
	uv run ruff format .

test:  ## 运行测试
	uv run pytest

train:  ## 运行训练入口（默认 module=defect；可追加 KEY=VAL 覆盖）
	uv run ivp-train

serve-api:  ## 启动 FastAPI 推理服务（需先 uv sync --extra service）
	uv run uvicorn ivp.service.api:create_app --factory --reload

serve-demo:  ## 启动 Gradio 演示界面（需先 uv sync --extra service）
	uv run python -m ivp.service.demo

mlflow-ui:  ## 打开 MLflow 实验追踪 UI
	uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db

clean:  ## 清理缓存与产物
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs/* mlruns
