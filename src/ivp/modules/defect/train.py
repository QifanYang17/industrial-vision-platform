"""缺陷检测训练入口（阶段一骨架，暂不含真实训练逻辑）。

待实现（下一步）：
1. 按 cfg.dataset 构建 anomalib Datamodule（MVTec, category=bottle）。
2. 按 cfg.model 选 PatchCore / EfficientAD，device 走 MPS。
3. fit → test，收集 image/pixel AUROC、PRO，记录延迟/FPS。
4. 用 ivp.common.tracking.log_metrics / log_artifacts 落盘到当前 MLflow run。
"""

from __future__ import annotations

from omegaconf import DictConfig

from ivp.common.logging import get_logger

logger = get_logger(__name__)


def train(cfg: DictConfig) -> None:
    """训练缺陷检测模型（占位）。

    当前仅打印将要执行的配置，证明 Hydra→分发→MLflow 链路已打通；
    真实的 anomalib 训练逻辑在下一步填充。
    """
    logger.warning(
        "【骨架占位】缺陷检测训练尚未实现 —— model=%s dataset=%s/%s",
        cfg.model.name,
        cfg.dataset.name,
        cfg.dataset.category,
    )
    # TODO(阶段一): 接入 anomalib PatchCore 训练与指标评估
