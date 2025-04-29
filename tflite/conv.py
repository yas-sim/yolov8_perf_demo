from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.export(format='tflite', int8=True)

