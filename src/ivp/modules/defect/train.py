"""缺陷检测训练：用 anomalib PatchCore 在 MVTec AD 上跑通并出指标。

为什么是 PatchCore + 无监督：工厂里缺陷样本稀少，只用「正常样品」训练最贴合现实。
PatchCore 用预训练 CNN 抽特征建「记忆库」(memory bank)，推理时按到记忆库的距离打
异常分——无需缺陷标注、无需反向传播（故只跑 1 个 epoch）。

本文件在 ``ivp.cli`` 已开启的 MLflow run 内被调用，因此这里直接 log_* 即可。
重型依赖（torch/anomalib）按既定约定在函数内 import，保证基础环境可导入本模块。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from ivp.common.logging import get_logger
from ivp.common.tracking import log_artifacts, log_metrics
from ivp.modules.defect import metrics as dm

logger = get_logger(__name__)

# Lightning 的 accelerator 命名与我们配置里的 device 命名映射
_ACCELERATOR = {"mps": "mps", "cuda": "gpu", "cpu": "cpu"}


def _build_datamodule(cfg: DictConfig):
    """按配置构建 MVTec AD datamodule（首次运行从 HF 镜像准备数据集）。"""
    from anomalib.data import MVTecAD

    from ivp.modules.defect.data import prepare_mvtec

    ds = cfg.dataset
    # anomalib 内置下载 URL 已失效，改用我们的镜像准备逻辑（幂等）
    prepare_mvtec(ds.root, ds.category)
    return MVTecAD(
        root=ds.root,
        category=ds.category,
        train_batch_size=ds.train_batch_size,
        eval_batch_size=ds.eval_batch_size,
        num_workers=ds.num_workers,
        seed=cfg.seed,
    )


def _build_model(cfg: DictConfig):
    """按配置构建 PatchCore，并挂上 image/pixel AUROC + PRO 评估器。"""
    from anomalib.metrics import AUPRO, AUROC, Evaluator
    from anomalib.models import Patchcore

    # 显式配置评估指标，确保 PRO（AUPRO）也被计算：
    # - image-level：用每图异常分 pred_score 对 gt_label（图级 OK/NG）
    # - pixel-level：用异常图 anomaly_map 对 gt_mask（像素级缺陷掩码）
    evaluator = Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            AUPRO(fields=["anomaly_map", "gt_mask"]),
        ]
    )
    return Patchcore(
        backbone=cfg.model.backbone,
        coreset_sampling_ratio=cfg.model.coreset_sampling_ratio,
        evaluator=evaluator,
    )


def _collect_scores(predictions) -> tuple[np.ndarray, np.ndarray]:
    """从 engine.predict 的输出收集图级异常分与真值标签，转成 numpy。"""
    scores, labels = [], []
    for batch in predictions:
        scores.append(np.asarray(batch.pred_score.cpu()).reshape(-1))
        labels.append(np.asarray(batch.gt_label.cpu()).reshape(-1).astype(int))
    return np.concatenate(scores), np.concatenate(labels)


def _write_threshold_report(
    scores: np.ndarray, labels: np.ndarray, out_dir: Path
) -> dict[str, float]:
    """计算阈值权衡、写 CSV + 曲线图 + Markdown 报告，返回要进 MLflow 的标量指标。"""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    points = dm.select_operating_points(scores, labels)
    table = dm.tradeoff_table(scores, labels, n_steps=50)
    sk_auroc = dm.image_auroc(scores, labels)

    # 0) 持久化原始分值/标签，便于事后廉价地重算报告/重画图，无需重训
    np.savez(out_dir / "scores.npz", scores=scores, labels=labels)

    # 1) 权衡表 CSV（漏检率 vs 过杀率随阈值变化）
    with (out_dir / "threshold_tradeoff.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["threshold", "miss_rate(FNR)", "overkill_rate(FPR)", "recall", "precision", "f1"]
        )
        for p in table:
            w.writerow([p.threshold, p.miss_rate, p.overkill_rate, p.recall, p.precision, p.f1])

    # 2) 权衡曲线图（英文标签，避免无 CJK 字体环境渲染成方框）
    if not dm.save_tradeoff_plot(
        [p.threshold for p in table],
        [p.miss_rate for p in table],
        [p.overkill_rate for p in table],
        points["zero_miss"].threshold,
        out_dir / "threshold_tradeoff.png",
    ):
        logger.warning("matplotlib 未安装，跳过权衡曲线图")

    # 3) Markdown 报告（面试可直接展示）
    lines = [
        "# 缺陷检测指标报告 — PatchCore / MVTec AD (bottle)",
        "",
        f"- image-level AUROC (sklearn 校验): **{sk_auroc:.4f}**",
        "",
        "## 工作点（阈值选择依据）",
        "",
        "| 工作点 | 阈值 | 漏检率FNR | 过杀率FPR | 召回 | 精度 | F1 | TP/FP/TN/FN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("f1_optimal", "zero_miss", "youden"):
        p = points[key]
        lines.append(
            f"| {key} | {p.threshold:.4f} | {p.miss_rate:.3f} | {p.overkill_rate:.3f} "
            f"| {p.recall:.3f} | {p.precision:.3f} | {p.f1:.3f} "
            f"| {p.tp}/{p.fp}/{p.tn}/{p.fn} |"
        )
    zm = points["zero_miss"]
    lines += [
        "",
        "## 解读",
        "",
        f"- **零漏检点**：把阈值设到 {zm.threshold:.4f} 可做到 **0% 漏检**（不放过任何 NG），"
        f"代价是 **{zm.overkill_rate * 100:.1f}% 过杀率**。工厂里漏检（次品流出）代价远高于"
        "过杀（误废良品），因此通常以「零漏检」为硬约束、再压低过杀。",
        "- **F1 最优点**：精度/召回平衡的默认阈值，适合对过杀也敏感的场景。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # 返回要落 MLflow 的标量
    flat: dict[str, float] = {"image_auroc_sklearn": sk_auroc}
    for key, p in points.items():
        flat[f"{key}.miss_rate"] = p.miss_rate
        flat[f"{key}.overkill_rate"] = p.overkill_rate
        flat[f"{key}.f1"] = p.f1
    return flat


def train(cfg: DictConfig) -> None:
    """端到端：fit → test（出 AUROC/PRO）→ predict（出阈值权衡）→ 落 MLflow。"""
    from anomalib.engine import Engine

    accelerator = _ACCELERATOR[cfg.hardware.device]
    logger.info("PatchCore 训练开始：backbone=%s accelerator=%s", cfg.model.backbone, accelerator)

    datamodule = _build_datamodule(cfg)
    model = _build_model(cfg)
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        logger=False,  # 用我们自己的 MLflow，不让 anomalib 另起 logger
        default_root_dir="outputs/anomalib",
    )

    # 1) 建记忆库（PatchCore 只需 1 个 epoch，anomalib 内部已设定）
    engine.fit(model=model, datamodule=datamodule)

    # 2) 测试集指标：image/pixel AUROC + PRO
    test_results = engine.test(model=model, datamodule=datamodule)
    anomalib_metrics: dict[str, float] = {}
    for d in test_results:  # list[dict]
        anomalib_metrics.update({k: float(v) for k, v in d.items()})
    logger.info("anomalib 测试指标: %s", anomalib_metrics)
    log_metrics(anomalib_metrics)

    # 3) 阈值权衡分析（漏检率 vs 过杀率）
    predictions = engine.predict(model=model, datamodule=datamodule)
    scores, labels = _collect_scores(predictions)
    report_dir = Path("outputs/defect_report")
    threshold_metrics = _write_threshold_report(scores, labels, report_dir)
    log_metrics(threshold_metrics)
    log_artifacts(str(report_dir))

    logger.info("阶段一完成，报告已写入 %s 并记录到 MLflow", report_dir)
