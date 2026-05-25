"""inference — 共享推理层：统一的推理接口与运行时（ONNX/原生）。

三个任务模块训练产物各异，但对外暴露统一的 ``BaseInferencer.predict`` 接口，
让 service 层（FastAPI/Gradio）无需关心底层是哪种模型。
"""
