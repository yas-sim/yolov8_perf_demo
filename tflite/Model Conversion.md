# Converting YOLOv8s model with ultralytics lib

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.export(format='tflite', int8=True)
```

This conversion requires the following library but it is not available for Windows. You need a Linux system to prepare the TFLite model. Ubuntu is recommended.

`"ai-edge-litert>=1.2.0"`
