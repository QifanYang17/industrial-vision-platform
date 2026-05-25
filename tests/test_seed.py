"""验证种子设置可复现（torch 缺席时也不报错）。"""

from __future__ import annotations

import random

import numpy as np

from ivp.common.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    a = (random.random(), float(np.random.rand()))
    set_seed(123)
    b = (random.random(), float(np.random.rand()))
    assert a == b
