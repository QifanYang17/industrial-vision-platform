# CLAUDE.md — 工业视觉统一平台

> 这是给 Claude Code 的项目上下文。每次开始工作前请先读完本文件。

## 项目目标

构建一个**工业视觉统一演示平台**作为求职作品集，目标是达到工业级工程标准（不是玩具 demo）。
平台整合三个可插拔的视觉任务模块，共用一套推理服务、演示界面和 MLOps 工程底座。

**关键背景**：这是给面试用的。代码质量和工程规范本身就是卖点。因此：
- 每个重要设计决策都要在代码注释或文档里写清楚「为什么这么做」。
- 我（项目作者）需要能向面试官逐行讲解代码和指标，所以请在生成关键代码时附简短解释，不要做成黑箱。
- 优先工业级实践：配置驱动、可复现、有测试、有部署。

## 三个任务模块与技术选型

1. **缺陷检测**（最高优先级，第一阶段做透）
   - 路线：无监督异常检测（只用正常样品训练，契合工厂「缺陷样本稀少」的真实约束）。
   - 工具：`anomalib`，方法用 PatchCore（首选）和 EfficientAD（对比）。
   - 数据集：MVTec AD，先从单个类别（如 bottle）跑通。
   - 指标：image-level AUROC、pixel-level AUROC、PRO score。重点报告**漏检率 vs 过杀率**的权衡和阈值选择依据。

2. **目标检测 + 计数**
   - 工具：Ultralytics YOLO26（2026年1月发布，端到端 NMS-free、边缘优化）；nano/small 变体起步。
   - 动态计数：YOLO26 检测 + ByteTrack 跟踪，按目标 ID 去重计数。
   - 数据集：Roboflow Universe 上的工业数据集 或 NEU-DET / SKU-110K。

3. **OCR 管道**
   - 路线：两阶段（检测定位铭牌/标签区域 → 文字识别）。
   - 工具：PaddleOCR（注意 Apple Silicon 兼容，见硬件约束）；备选 EasyOCR。
   - 难点处理：旋转、低对比度、金属反光的数据增强。

## 架构（数据流）

```
数据层 → 预处理/增强 → [缺陷检测 | 检测+计数 | OCR]（可插拔模块）
       → 推理服务(FastAPI + ONNX/TensorRT) → 演示界面(Gradio) + 部署(Docker/HF Spaces)
```
MLOps 工程层横跨全流程（不是最后一步）：配置驱动、实验追踪、CI/CD、数据版本。

## 工程规范（工业级标准）

- **配置驱动**：用 Hydra 或 YAML 管理所有超参，禁止硬编码。
- **实验追踪**：MLflow 或 W&B 记录每次训练。
- **可复现**：固定随机种子，确定性设置。
- **数据版本**：DVC（数据集不进 git，用 .gitignore + DVC 指针）。
- **测试**：pytest，至少覆盖前处理/后处理逻辑。
- **CI**：GitHub Actions 做 lint + test + docker build。
- **服务**：FastAPI 异步推理服务 + Gradio 演示界面，docker-compose 一键拉起。
- **指标**：每个模型都要同时报告**精度指标**和**延迟/FPS**（标注硬件），并给出 ONNX/量化后的对比。
- 代码风格：类型注解、docstring、模块化、可读性优先。用 ruff 做 lint。

## 硬件约束（重要）

开发机：2024 MacBook Pro，**M4 芯片，16GB 统一内存**。
- PyTorch 用 **MPS 后端**（`device="mps"`）吃 GPU 加速。
- 16GB 是主要瓶颈：训练时 batch size 调小、用 nano/small 模型、必要时降 imgsz。
- 缺陷检测（PatchCore/EfficientAD）本地可轻松跑通。
- YOLO26 微调本地可行但要克制；**重训练考虑 Colab/Kaggle 免费 GPU**，本地只负责开发调试。
- PaddleOCR 在 Apple Silicon 历史兼容性差：**务必在独立 venv 里按官方 M4 教程安装**，否则改用 EasyOCR 或放进 Docker 跑。

## 分阶段计划

- **阶段一（当前）**：项目骨架 + 缺陷检测线。搭好目录结构、依赖、配置框架；用 Anomalib 在 MVTec bottle 类别上跑通 PatchCore，输出完整指标报告。
- **阶段二**：YOLO26 检测 + ByteTrack 计数。
- **阶段三**：OCR 管道 + 统一 Gradio 界面 + Docker 化 + CI + README/技术文档。

每阶段结束都应是一个可独立展示的里程碑。

## 当前任务

见我在对话里给你的指令。如无特别说明，默认推进**阶段一**。
开始前如对技术选型或目录结构有更优建议，先提出来和我讨论，不要默默改动既定方案。
