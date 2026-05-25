"""common — 跨模块共享的工程工具：配置、日志、随机种子、设备、实验追踪。

设计原则：这一层只依赖核心依赖（hydra/omegaconf/mlflow/...），不直接 import
重型 ML 框架（torch 等按需在函数内 guarded import），保证基础环境轻量、可测。
"""
