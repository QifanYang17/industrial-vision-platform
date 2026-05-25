"""数据集准备脚本：重建 MVTec AD 目录结构（绕开 anomalib 失效的下载 URL）。

用法：
    uv run python scripts/download_data.py --category bottle --root data/mvtec

约定：数据集不进 git（gitignored），后续可对 data/mvtec 接 DVC 指针做数据版本管理。
实际重建逻辑见 ``ivp.modules.defect.data.prepare_mvtec``。
"""

from __future__ import annotations

import argparse

from ivp.modules.defect.data import prepare_mvtec


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Prepare MVTec AD dataset from HF mirror.")
    parser.add_argument("--category", default="bottle", help="MVTec 类别，如 bottle")
    parser.add_argument("--root", default="data/mvtec", help="数据根目录")
    args = parser.parse_args()
    prepare_mvtec(args.root, args.category)


if __name__ == "__main__":
    main()
