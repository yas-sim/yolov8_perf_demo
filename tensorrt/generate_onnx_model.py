import os
from ultralytics import YOLO

m = YOLO('yolov8s.pt')

# https://docs.ultralytics.com/ja/usage/cfg/#predict-settings
for batch in (1, 32):
    m.export(format='onnx', batch=batch, imgsz=640)
    os.rename('yolov8s.onnx', f'yolov8s_b{batch}.onnx')
