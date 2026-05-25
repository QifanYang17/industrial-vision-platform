"""统一训练入口：把 Hydra（配置）与 MLflow（追踪）接到一起。

运行方式（控制台脚本，见 pyproject [project.scripts]）：
    ivp-train                         # 用默认配置（module=defect）
    ivp-train module=detection seed=1 # 命令行覆盖任意配置项
    ivp-train tracking.enabled=false  # 关闭实验追踪

⚠️ 阶段一（骨架）：本入口已完整打通「组合配置 → 固定种子 → 选设备 → 开
MLflow run → 记录配置 → 分发到模块」的链路，但**各模块的 train() 仍是占位**，
不含真实训练逻辑。进入对应阶段时再填充模块实现即可。
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from ivp.common.config import CONFIG_DIR, register_schemas
from ivp.common.device import resolve_device
from ivp.common.logging import get_logger
from ivp.common.seed import set_seed
from ivp.common.tracking import start_run

logger = get_logger(__name__)

# 模块名 → 训练函数 的分发表（懒加载在 _dispatch 内 import，避免一次性拉重型依赖）
_MODULES = {"defect", "detection", "ocr"}


def _dispatch(name: str, cfg: DictConfig) -> None:
    """按模块名分发到对应的 train()。"""
    if name == "defect":
        from ivp.modules.defect.train import train
    elif name == "detection":
        from ivp.modules.detection.train import train
    elif name == "ocr":
        from ivp.modules.ocr.train import train
    else:
        raise ValueError(f"未知模块 {name!r}，应为 {_MODULES} 之一")
    train(cfg)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def train_entry(cfg: DictConfig) -> None:
    """Hydra 主入口：组合配置后执行通用前置步骤，再分发到模块训练。"""
    register_schemas()
    logger.info("组合后的配置:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed, cfg.hardware.deterministic)
    device = resolve_device(cfg.hardware.device)
    logger.info("实际使用设备: %s", device)

    with start_run(cfg, run_name=f"{cfg.module.name}-{cfg.model.name}"):
        _dispatch(cfg.module.name, cfg)


if __name__ == "__main__":
    train_entry()
