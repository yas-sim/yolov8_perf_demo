import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8s.pt').to('cuda')

cap = None
stime = time.perf_counter()
num_infer = 0
fps = 0
while True:
    if cap is None:
        cap = cv2.VideoCapture('../data/people.mp4')
    sts, image = cap.read()
    if sts == False or image is None:
        cap.release()
        cap = None
        continue
    image = cv2.resize(image, (640, 640))

    results = model.predict(source=image, imgsz=640, device='cuda:0', stream=True, verbose=False)
    #results = model.predict(source=image, imgsz=640, device='cpu', stream=True, verbose=False)
    num_infer += 1

    if num_infer % 10 == 0:
        etime = time.perf_counter()
        fps = 1.0 / ((etime-stime)/10)
        stime = etime

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = box.cls[0]
            confidence = box.conf[0]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(image, f'{fps:.2f} FPS', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 6)
            cv2.putText(image, f'{fps:.2f} FPS', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)

    cv2.imshow('Ultralytics: YOLOv8s', image)
    if cv2.waitKey(1) == 27:        # ESC key
        break

cv2.destroyAllWindows()
