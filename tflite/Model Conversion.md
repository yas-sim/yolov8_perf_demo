# Converting YOLOv8s model with ultralytics lib

## 1. Convert the YOLOv8s model.

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.export(format='tflite', int8=True)
```

This conversion requires the following library but it is not available for Windows. You need a Linux system to prepare the TFLite model. Ubuntu is recommended.

`"ai-edge-litert>=1.2.0"`


## 2. Convert the YOLOv8s model with a container.

`Dockerfile` and some script files are placed in the `Docker` folder.

```sh
cd docker
docker build -t yolo8_conv .
cd ..
```

```sh
docker run --rm -v <full_path_to_work_dir>:/work/share yolo8_conv 
```
The models will be generated in the `yolov8s_saved_model` directory.
