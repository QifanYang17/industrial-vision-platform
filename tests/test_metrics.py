"""缺陷检测阈值分析的单元测试（纯逻辑，不依赖 anomalib）。"""

from __future__ import annotations

import numpy as np

from ivp.modules.defect.metrics import (
    image_auroc,
    select_operating_points,
    tradeoff_table,
)


def test_perfectly_separable() -> None:
    """OK 分低、NG 分高且完全可分时，应能零漏检且零过杀，AUROC=1。"""
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.8])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert image_auroc(scores, labels) == 1.0

    pts = select_operating_points(scores, labels)
    zm = pts["zero_miss"]
    assert zm.miss_rate == 0.0  # 零漏检（定义如此）
    assert zm.overkill_rate == 0.0  # 完全可分 ⇒ 不必牺牲过杀
    assert zm.fn == 0


def test_zero_miss_costs_overkill_when_overlapping() -> None:
    """有重叠时，零漏检必然以一定过杀率为代价。"""
    # 一个 NG 的分值(0.4)低于部分 OK，迫使零漏检阈值压低 → 误判 OK
    scores = np.array([0.1, 0.5, 0.6, 0.4, 0.95, 0.8])
    labels = np.array([0, 0, 0, 1, 1, 1])
    zm = select_operating_points(scores, labels)["zero_miss"]
    assert zm.miss_rate == 0.0
    assert zm.overkill_rate > 0.0  # 为了不漏检，误废了良品


def test_tradeoff_table_is_monotone_ish() -> None:
    """阈值升高时漏检率不降、过杀率不升（单调方向正确）。"""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.3, 0.1, 50), rng.normal(0.7, 0.1, 50)])
    labels = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
    table = tradeoff_table(scores, labels, n_steps=20)
    misses = [p.miss_rate for p in table]
    overkills = [p.overkill_rate for p in table]
    assert misses == sorted(misses)  # 阈值升 ⇒ 漏检率不降
    assert overkills == sorted(overkills, reverse=True)  # 阈值升 ⇒ 过杀率不升
