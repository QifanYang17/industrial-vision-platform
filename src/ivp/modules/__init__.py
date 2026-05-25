"""modules — 三个可插拔视觉任务模块。

- defect：缺陷检测（无监督异常检测，anomalib / PatchCore）—— 阶段一
- detection：目标检测 + 计数（YOLO26 + ByteTrack）—— 阶段二
- ocr：OCR 管道（两阶段，PaddleOCR / EasyOCR）—— 阶段三

每个模块约定暴露 ``train(cfg)`` / ``infer`` / ``metrics``，由 ``ivp.cli`` 统一分发。
"""
