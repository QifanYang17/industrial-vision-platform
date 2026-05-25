"""实验追踪（MLflow）骨架封装。

为什么封装而不直接调 mlflow：把「设置后端 URI → 选实验 → 开 run → 记录扁平化
配置」这套样板收敛到一处，各模块训练时一行 `with start_run(cfg):` 即可，
保证每次实验都按统一规范留痕（参数/指标/产物），可追溯、可对比。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import mlflow
from omegaconf import DictConfig

from ivp.common.config import flatten_config
from ivp.common.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def start_run(cfg: DictConfig, run_name: str | None = None) -> Iterator[Any]:
    """以配置驱动的方式开启一次 MLflow run，并自动记录扁平化后的全部配置。

    Args:
        cfg: 组合后的配置，需含 ``cfg.tracking``（enabled/tracking_uri/experiment_name）。
        run_name: 可选的 run 名称。

    Yields:
        活动的 MLflow run 对象（``tracking.enabled=False`` 时为 ``None``，
        训练代码据此可无侵入地跳过追踪）。
    """
    tracking = cfg.tracking
    if not tracking.enabled:
        logger.info("实验追踪已关闭（tracking.enabled=false），跳过 MLflow")
        yield None
        return

    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.set_experiment(tracking.experiment_name)
    logger.info("MLflow: uri=%s experiment=%s", tracking.tracking_uri, tracking.experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_config(cfg))
        logger.info("MLflow run 已启动: run_id=%s", run.info.run_id)
        yield run


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """记录一组指标（如 image_auroc / pixel_auroc / latency_ms）。"""
    if mlflow.active_run() is None:
        logger.debug("无活动 run，跳过 log_metrics")
        return
    mlflow.log_metrics(dict(metrics), step=step)


def log_artifacts(local_path: str) -> None:
    """记录产物文件或目录（如可视化结果、混淆矩阵、ONNX 模型）。"""
    if mlflow.active_run() is None:
        logger.debug("无活动 run，跳过 log_artifacts")
        return
    mlflow.log_artifacts(local_path)
