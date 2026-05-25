"""pytest 共享 fixtures。"""

from __future__ import annotations

import pytest
from omegaconf import DictConfig

from ivp.common.config import load_config


@pytest.fixture
def cfg() -> DictConfig:
    """加载默认组合配置（不经命令行），供各测试复用。"""
    return load_config()
