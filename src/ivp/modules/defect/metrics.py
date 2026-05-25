"""缺陷检测指标与阈值分析。

约定（贯穿本模块）：
- 异常分 score 越高越「像缺陷」；正类 = NG（有缺陷），label: 1=NG, 0=OK。
- 判定规则：score >= threshold ⇒ 判为 NG。

为什么单独做阈值分析（anomalib 已给 AUROC，为何还要这个）：
AUROC 是「与阈值无关」的排序质量，但工厂真正要落地的是**选一个阈值**，
而漏检（FN：把 NG 放过 → 次品流出，代价极高）和过杀（FP：把 OK 判成 NG →
误废良品，代价相对低）的权衡，正是靠阈值决定的。本模块把这条权衡曲线显式
算出来，并给出几个可解释的工作点，便于在报告里讲清「阈值选择依据」。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class OperatingPoint:
    """某个阈值下的混淆矩阵与关键率。"""

    name: str
    threshold: float
    miss_rate: float  # 漏检率 FNR = FN/(TP+FN)：NG 中被放过的比例（越低越安全）
    overkill_rate: float  # 过杀率 FPR = FP/(FP+TN)：OK 中被误判的比例
    recall: float  # = 1 - miss_rate，NG 召回
    precision: float  # 判为 NG 的样本里真为 NG 的比例
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def _confusion_at(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """返回给定阈值下的 (tp, fp, tn, fn)。"""
    pred_ng = scores >= threshold
    is_ng = labels == 1
    tp = int(np.sum(pred_ng & is_ng))
    fp = int(np.sum(pred_ng & ~is_ng))
    tn = int(np.sum(~pred_ng & ~is_ng))
    fn = int(np.sum(~pred_ng & is_ng))
    return tp, fp, tn, fn


def _operating_point(
    name: str, scores: np.ndarray, labels: np.ndarray, threshold: float
) -> OperatingPoint:
    tp, fp, tn, fn = _confusion_at(scores, labels, threshold)
    miss_rate = fn / (tp + fn) if (tp + fn) else 0.0
    overkill_rate = fp / (fp + tn) if (fp + tn) else 0.0
    recall = 1.0 - miss_rate
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return OperatingPoint(
        name=name,
        threshold=float(threshold),
        miss_rate=miss_rate,
        overkill_rate=overkill_rate,
        recall=recall,
        precision=precision,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """image-level AUROC（与阈值无关的排序质量）。"""
    return float(roc_auc_score(labels, scores))


def select_operating_points(scores: np.ndarray, labels: np.ndarray) -> dict[str, OperatingPoint]:
    """挑选几个可解释的工作点，覆盖「漏检 vs 过杀」权衡的两端与折中。

    返回三个命名工作点：
    - ``f1_optimal``：F1 最大的阈值（精度/召回的平衡点，常作默认）。
    - ``zero_miss``：能做到**零漏检**（FNR=0）的最高阈值，并报告其过杀代价——
      工厂场景常以「绝不漏检」为硬约束，这个点回答「为此要付出多少过杀」。
    - ``youden``：Youden's J（recall - overkill 最大）统计意义上的最佳分离点。

    Args:
        scores: 形如 (N,) 的异常分。
        labels: 形如 (N,) 的真值（1=NG, 0=OK）。

    Returns:
        名称 → OperatingPoint 的字典。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    # 候选阈值：所有不同分值（含略低于最小值的点，保证能取到「全部判 NG」）
    candidates = np.unique(scores)
    candidates = np.concatenate([[candidates[0] - 1e-9], candidates])

    points = [_operating_point("_", scores, labels, t) for t in candidates]

    f1_pt = max(points, key=lambda p: p.f1)
    youden_pt = max(points, key=lambda p: p.recall - p.overkill_rate)
    # 零漏检：miss_rate==0 中阈值最高者（过杀最小的零漏检点）
    zero_miss_candidates = [p for p in points if p.miss_rate == 0.0]
    zero_miss_pt = max(zero_miss_candidates, key=lambda p: p.threshold)

    return {
        "f1_optimal": _operating_point("f1_optimal", scores, labels, f1_pt.threshold),
        "zero_miss": _operating_point("zero_miss", scores, labels, zero_miss_pt.threshold),
        "youden": _operating_point("youden", scores, labels, youden_pt.threshold),
    }


def tradeoff_table(
    scores: np.ndarray, labels: np.ndarray, n_steps: int = 50
) -> list[OperatingPoint]:
    """在分值范围内均匀取阈值，生成漏检率 vs 过杀率权衡表（供画曲线/落 MLflow）。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    lo, hi = float(scores.min()), float(scores.max())
    thresholds = np.linspace(lo, hi, n_steps)
    return [_operating_point(f"t={t:.4f}", scores, labels, t) for t in thresholds]


def save_tradeoff_plot(
    thresholds: Sequence[float],
    miss_rates: Sequence[float],
    overkill_rates: Sequence[float],
    zero_miss_threshold: float,
    out_path: str | Path,
) -> bool:
    """画「漏检率 vs 过杀率」随阈值变化的权衡曲线并存图。

    标签刻意用英文：避免在无 CJK 字体的环境（CI / Docker 的 DejaVu Sans）里
    把中文渲染成方框。matplotlib 缺失时跳过并返回 False。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, miss_rates, "r-o", ms=3, label="Miss rate (FNR, lower is safer)")
    ax.plot(thresholds, overkill_rates, "b-s", ms=3, label="Overkill rate (FPR)")
    ax.axvline(
        zero_miss_threshold,
        color="g",
        ls="--",
        label=f"Zero-miss threshold = {zero_miss_threshold:.3f}",
    )
    ax.set_xlabel("anomaly score threshold")
    ax.set_ylabel("rate")
    ax.set_title("PatchCore / MVTec bottle — Miss rate vs Overkill rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True
