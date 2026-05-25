"""目标检测 + 计数模块（阶段二）。

工具：Ultralytics YOLO26（端到端 NMS-free、边缘优化），nano/small 起步；
动态计数 = YOLO26 检测 + ByteTrack 跟踪，按目标 ID 去重。
数据集：Roboflow Universe 工业集 / NEU-DET / SKU-110K。
"""
