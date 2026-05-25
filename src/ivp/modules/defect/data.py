"""MVTec AD 数据准备：从 HuggingFace parquet 镜像重建 anomalib 期望的目录结构。

为什么需要这个（重要工程决策，面试可讲）：
anomalib 2.4.2 内置的 MVTec 下载地址（mydrive.ch）已失效返回 404。我们改从
HF 上的 parquet 镜像 ``TheoM55/mvtec_all_objects_split`` 拉取——它保留了**原版**
图像 + 像素级 mask（bottle: 209 train / 83 test，缺陷类 broken_large/broken_small/
contamination，与原数据集逐一致），再按 anomalib 的约定重建目录：

    <root>/<category>/
        train/good/<name>.png
        test/<defect>/<name>.png
        ground_truth/<defect>/<stem>_mask.png   # 仅缺陷类有 mask

这样 ``MVTecAD.prepare_data()`` 检测到 ``<root>/<category>`` 已存在即跳过下载。
相比下载完整 5.27GB 原始包，这里只取单类别（~156MB），更快且可复现。
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from ivp.common.logging import get_logger

logger = get_logger(__name__)

# 原版 MVTec AD 的 parquet 镜像（含图像 + 像素 mask）
_HF_REPO = "TheoM55/mvtec_all_objects_split"


def _write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def prepare_mvtec(root: str | Path, category: str) -> Path:
    """确保 ``<root>/<category>`` 存在 anomalib 期望的 MVTec 目录结构。

    幂等：若目标已就绪则直接返回，不重复下载。

    Args:
        root: 数据根目录（如 ``data/mvtec``）。
        category: MVTec 类别（如 ``bottle``）。

    Returns:
        类别目录路径 ``<root>/<category>``。
    """
    base = Path(root) / category
    if (base / "train" / "good").is_dir() and any((base / "train" / "good").iterdir()):
        logger.info("MVTec %s 已就绪，跳过准备: %s", category, base)
        return base

    logger.info("准备 MVTec %s：从 HF 镜像 %s 重建目录结构", category, _HF_REPO)
    n_img = n_mask = 0
    for split in ("train", "test"):
        parquet = hf_hub_download(
            _HF_REPO,
            f"data/{category}.{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(parquet).to_pydict()
        for i in range(len(table["label"])):
            defect = table["defect"][i]
            img = table["image_path"][i]
            name = Path(img["path"]).name
            _write_bytes(base / split / defect / name, img["bytes"])
            n_img += 1
            # 缺陷类带像素 mask；mask 文件名用「图像 stem + _mask.png」匹配 anomalib 约定
            mask = table["mask_path"][i]
            if mask is not None and mask.get("bytes"):
                stem = Path(name).stem
                _write_bytes(base / "ground_truth" / defect / f"{stem}_mask.png", mask["bytes"])
                n_mask += 1

    logger.info("MVTec %s 准备完成：%d 张图像 + %d 张 mask → %s", category, n_img, n_mask, base)
    return base
