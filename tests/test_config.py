"""验证 Hydra 配置能正确组合，且命令行式覆盖生效。"""

from __future__ import annotations

from ivp.common.config import flatten_config, load_config


def test_default_config_composes(cfg) -> None:
    """默认配置应组合出完整的五个 group + seed。"""
    assert cfg.seed == 42
    assert cfg.module.name == "defect"
    assert cfg.model.name == "patchcore"
    assert cfg.dataset.category == "bottle"
    assert cfg.tracking.enabled is True
    assert cfg.hardware.device == "mps"


def test_overrides_take_effect() -> None:
    """覆盖 group 与单项均应生效。"""
    cfg = load_config(overrides=["module=detection", "seed=7", "hardware.device=cpu"])
    assert cfg.module.name == "detection"
    assert cfg.seed == 7
    assert cfg.hardware.device == "cpu"


def test_flatten_config_is_flat(cfg) -> None:
    """压平后应为单层 dict，且包含点分 key（供 MLflow log_params）。"""
    flat = flatten_config(cfg)
    assert "tracking.experiment_name" in flat
    assert all("." in k or k == "seed" for k in flat)
