# 架构说明

> 面向面试讲解的设计文档。随各阶段推进补充指标与决策记录。

## 数据流

```
数据层(DVC) → 预处理/增强 → [缺陷检测 | 检测+计数 | OCR]（可插拔模块）
            → 推理服务(FastAPI + ONNX) → 演示界面(Gradio) + 部署(Docker/HF Spaces)
```

MLOps 工程层（配置驱动 / 实验追踪 / CI / 数据版本）横跨全流程，而非最后一步。

## 分层与目录映射

| 层 | 目录 | 职责 |
|----|------|------|
| 配置 | `configs/` + `ivp/common/config.py` | Hydra 组合式配置，命令行可覆盖 |
| 工程底座 | `ivp/common/` | 日志、随机种子、设备选择、MLflow 封装 |
| 任务模块 | `ivp/modules/{defect,detection,ocr}/` | 三个可插拔视觉任务，约定暴露 `train/infer/metrics` |
| 推理层 | `ivp/inference/` | 统一 `BaseInferencer` 接口 + ONNX 运行时 |
| 服务层 | `ivp/service/` | FastAPI 推理 API + Gradio 演示界面 |
| 编排入口 | `ivp/cli.py` | Hydra 主入口，串起种子/设备/追踪后分发到模块 |

## 关键设计决策（为什么）

- **uv + 锁文件**：跨机器逐位可复现，本地与 Docker 依赖一致。
- **分阶段可选依赖组**（defect/detection/ocr/service）：基础环境轻量，按阶段安装重型 ML 框架；
  其中 OCR 因 PaddleOCR 在 Apple Silicon 的兼容问题，单独隔离（独立 venv 或 Docker）。
- **统一推理接口**：service 层只依赖 `BaseInferencer` 稳定契约，模块可插拔、可替换为 ONNX。
- **配置/种子/设备/追踪集中在 common**：每次实验按统一规范留痕，指标可复现、可对比。

## 指标约定

每个模型同时报告**精度指标**与**延迟/FPS**（标注硬件 = M4 / 16GB / MPS），并给出
ONNX/量化前后的对比。缺陷检测额外报告**漏检率 vs 过杀率**的权衡与阈值选择依据。
