"""数据集获取脚本（骨架）。

约定：数据集不进 git，用 DVC 管理（data/ 下仅提交 .dvc 指针）。
待实现：MVTec AD 下载/校验 + `dvc add data/mvtec` 生成指针。
"""

from __future__ import annotations

from ivp.common.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """下载并用 DVC 跟踪数据集（占位）。"""
    logger.warning("【骨架占位】数据下载脚本待实现（MVTec AD + DVC 指针）")


if __name__ == "__main__":
    main()
