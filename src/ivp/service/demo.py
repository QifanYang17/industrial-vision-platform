"""Gradio 演示界面（阶段一骨架：占位界面，三模块 Tab 待接入）。

启动：python -m ivp.service.demo
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gradio as gr


def build_demo() -> gr.Blocks:
    """构建 Gradio 演示界面（gradio 属 service 组，运行期 import）。"""
    import gradio as gr

    with gr.Blocks(title="工业视觉统一平台") as demo:
        gr.Markdown("# 工业视觉统一平台\n阶段一骨架：缺陷检测 / 检测+计数 / OCR Tab 待接入。")
    return demo


if __name__ == "__main__":
    build_demo().launch()
