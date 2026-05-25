"""验证设备解析的回退逻辑与入参校验。"""

from __future__ import annotations

import pytest

from ivp.common.device import VALID_DEVICES, resolve_device


def test_resolve_returns_valid_device() -> None:
    assert resolve_device("cpu") in VALID_DEVICES


def test_invalid_device_raises() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")
