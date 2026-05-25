"""配置框架（Hydra）骨架。

为什么这样设计：
- **结构化配置（dataclass schema）**：用 dataclass 声明配置的「形状」与默认值，
  既是文档，又能在组合时捕获拼写错误/类型错误，比纯 YAML 更安全。
- **YAML 组合**：`configs/` 下按 group（module/model/dataset/tracking/hardware）
  拆分，`config.yaml` 用 defaults 列表组合，命令行可 `key=value` 覆盖。
- **可测试的加载入口**：`load_config()` 用 Hydra 的 compose API，让单测不依赖
  命令行就能加载并校验整棵配置树。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# 仓库根目录下的 configs/ 绝对路径：src/ivp/common/config.py -> parents[3] == 仓库根
CONFIG_DIR: Path = Path(__file__).resolve().parents[3] / "configs"


# ── 结构化配置 schema（默认值即文档）─────────────────────────────────
@dataclass
class TrackingConfig:
    """MLflow 实验追踪配置。"""

    enabled: bool = True
    tracking_uri: str = "file:./mlruns"  # 本地文件后端；接远程改成 http://host:5000
    experiment_name: str = "ivp-default"


@dataclass
class HardwareConfig:
    """硬件 / 可复现性配置。"""

    device: str = "mps"  # mps（M4 GPU）| cpu | cuda
    deterministic: bool = True  # 固定算法、关闭非确定性 kernel，换可复现


@dataclass
class ModuleConfig:
    """任务模块选择。name ∈ {defect, detection, ocr}。"""

    name: str = "defect"


@dataclass
class DatasetConfig:
    """数据集配置（占位，各模块按需扩展）。"""

    name: str = "mvtec_bottle"
    root: str = "data/mvtec"
    category: str = "bottle"


@dataclass
class ModelConfig:
    """模型配置（占位，各模块按需扩展）。"""

    name: str = "patchcore"


@dataclass
class IVPConfig:
    """顶层配置 schema。"""

    seed: int = 42
    module: ModuleConfig = field(default_factory=ModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def register_schemas() -> None:
    """把 dataclass schema 注册进 Hydra ConfigStore，供 YAML 作为类型基底校验。"""
    cs = ConfigStore.instance()
    cs.store(name="ivp_schema", node=IVPConfig)


def load_config(
    overrides: list[str] | None = None,
    config_name: str = "config",
) -> DictConfig:
    """不经命令行加载并组合配置（供脚本/测试调用）。

    Args:
        overrides: Hydra 风格覆盖，如 ``["module=detection", "seed=1"]``。
        config_name: 根配置文件名（不含 .yaml）。

    Returns:
        组合后的 ``DictConfig``。
    """
    register_schemas()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


def flatten_config(cfg: DictConfig | dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """把嵌套配置压平成 ``{"a.b.c": value}``，便于 MLflow ``log_params``。

    MLflow 的 params 是扁平 key→value，因此训练前把整棵配置压平记录下来，
    保证每次 run 的超参可追溯、可对比。
    """
    container = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    items: dict[str, Any] = {}
    for key, value in container.items():  # type: ignore[union-attr]
        full_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_config(value, full_key))
        else:
            items[full_key] = value
    return items
