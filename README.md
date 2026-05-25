# 工业视觉统一平台 (Industrial Vision Platform, `ivp`)

整合三个可插拔视觉任务模块（缺陷检测 / 目标检测+计数 / OCR），共用一套配置驱动、
实验追踪、推理与服务的 **MLOps 工程底座**。目标是工业级工程标准的求职作品集。

> 项目背景、技术选型与分阶段计划见 [CLAUDE.md](CLAUDE.md)。

## 快速开始

```bash
# 1. 安装基础 + 开发依赖（uv 会按 .python-version 自动拉 Python 3.11）
make setup            # 等价于 uv sync --group dev && pre-commit install

# 2. 跑通工程底座（配置 + 种子 + 设备 + MLflow 链路，暂不含真实训练）
uv run ivp-train                      # 默认 module=defect
uv run ivp-train module=detection     # 命令行覆盖任意配置
uv run ivp-train tracking.enabled=false

# 3. 测试 / lint
make test
make lint

# 4. 查看 MLflow 实验
make mlflow-ui
```

## 分阶段安装重型依赖

基础环境只含 MLOps + 配置骨架，按阶段安装对应 ML 框架：

```bash
uv sync --extra defect      # 阶段一：anomalib + torch（缺陷检测）
uv sync --extra detection   # 阶段二：ultralytics + supervision
uv sync --extra ocr         # 阶段三：easyocr（PaddleOCR 需独立 venv，见 CLAUDE.md）
uv sync --extra service     # 服务层：fastapi + gradio + onnxruntime
```

## 项目结构

详见 [docs/architecture.md](docs/architecture.md)。核心：`configs/`（Hydra 配置）、
`src/ivp/common/`（工程底座）、`src/ivp/modules/`（三模块）、`src/ivp/{inference,service}/`、
`docker/`、`.github/workflows/`。

## 当前进度

- **阶段一（缺陷检测，已完成）**：✅ 项目骨架 + 依赖/配置/追踪底座；✅ Anomalib PatchCore 在 MVTec AD bottle 跑通。
  - 指标（M4 / MPS，seed=42）：**image AUROC 1.000 · pixel AUROC 0.986 · PRO 0.946**。
  - 完整报告（含漏检率 vs 过杀率阈值分析与权衡曲线）：[docs/results/defect-patchcore-bottle](docs/results/defect-patchcore-bottle/report.md)。
  - 复现：`uv sync --extra defect && uv run ivp-train`（数据集自动从 HF 镜像准备）。
- 阶段二：YOLO26 + ByteTrack 检测计数
- 阶段三：OCR 管道 + 统一 Gradio 界面 + Docker + CI
