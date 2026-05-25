"""service — 共享服务层：FastAPI 异步推理服务 + Gradio 演示界面。

依赖（fastapi/uvicorn/gradio）属 service 可选组，故重型 import 放在工厂函数内，
保证未装 service 组时本包仍可被导入（便于纯训练/测试环境）。
"""
