"""缺陷检测模块（阶段一，最高优先级）。

路线：无监督异常检测——只用正常样品训练，契合工厂「缺陷样本稀少」的真实约束。
工具：anomalib；方法 PatchCore（首选）对比 EfficientAD。数据集：MVTec AD（先 bottle）。
指标：image/pixel-level AUROC、PRO，重点报告漏检率 vs 过杀率的权衡与阈值依据。
"""
