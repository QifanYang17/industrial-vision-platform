"""OCR 模块（阶段三）。

路线：两阶段（检测定位铭牌/标签区域 → 文字识别）。
工具：PaddleOCR（注意 Apple Silicon 兼容，需独立 venv 或 Docker），备选 EasyOCR。
难点：旋转、低对比度、金属反光的数据增强。
"""
