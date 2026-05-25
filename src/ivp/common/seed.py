"""可复现性：统一固定随机种子 + 确定性设置。

为什么重要：面试时要能复现指标。固定种子 + 关闭非确定性算子，才能让
「同样的配置 → 同样的结果」，否则报告的 AUROC 每次都飘，没有说服力。
torch 为可选重型依赖，这里 guarded import：未安装也不报错。
"""

from __future__ import annotations

import os
import random

import numpy as np

from ivp.common.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """固定 random / numpy / torch（若安装）的随机种子。

    Args:
        seed: 随机种子。
        deterministic: 是否开启 torch 确定性算法（牺牲少量速度换可复现）。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # warn_only：遇到无确定性实现的算子时告警而非直接报错，避免训练崩
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        logger.debug("torch 未安装，跳过 torch 种子设置（基础环境属正常情况）")

    logger.info("随机种子已固定: seed=%d, deterministic=%s", seed, deterministic)
