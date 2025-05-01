import time
import random
from functools import partial

import numpy as np
import cv2

import threading
import queue

import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InputVStreamParams,
    InputVStreams,
    OutputVStreamParams,
    OutputVStreams,
    VDevice,
    Device
)

MODEL_PATH = './yolov8s.hef'
MOVIE_PATH = '../data/people.mp4'

YOLO_LABELS = ['person', 'bicycle','car','motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

BBOX_COLORS = [[random.randint(64, 255) for _ in range(3)] for _ in range(len(YOLO_LABELS))] 

def capt():
    global capt_img_q
    cap = None
    while abort_flag == False:
        if cap is None:
            cap = cv2.VideoCapture(MOVIE_PATH)
        sts, img = cap.read()
        if sts == False or img is None:
            cap.release()
            cap = None
            continue
        img = cv2.resize(img, (640, 640))
        tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        capt_img_q.put((img, tensor))
    cap.release()


def callback(bindings, img, completion_info):
    global inf_result_q
    if completion_info.exception:
        print(f'Inference error: {completion_info.exception}')
        return
    for binding in bindings:
        res = binding.output().get_buffer()
        inf_result_q.put((img, res))

def infer():
    global abort_flag
    global capt_img_q
    global MODEL_PATH
    hailo8_devices = Device.scan()
    print(f'{len(hailo8_devices)} Hailo8 devices are found. {hailo8_devices}')
    params = VDevice.create_params()
    params.device_ids = hailo8_devices
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    # The vdevice is used as a context manager ("with" statement) to ensure it's released on time.
    with VDevice(params) as vdevice:

        # Create an infer model from an HEF:
        infer_model = vdevice.create_infer_model(MODEL_PATH)
        model_input = infer_model.input()
        print(model_input.name, model_input.shape)
        
        # Configure the infer model and create bindings for it
        with infer_model.configure() as configured_infer_model:
            bindings = configured_infer_model.create_bindings()
            buffer = np.zeros(infer_model.output().shape).astype(np.float32)
            bindings.output().set_buffer(buffer)

            # Run asynchronous inference
            queue_size = configured_infer_model.get_async_queue_size()
            print(f'Async queue size: {queue_size}')

            img = None
            tensor = None
            while abort_flag == False:
                while capt_img_q.qsize() > 0:
                    img, tensor = capt_img_q.get()
                if tensor is None:
                    continue
                bindings.input().set_buffer(tensor)

                configured_infer_model.wait_for_async_ready()
                job = configured_infer_model.run_async(
                    bindings = [bindings], 
                    callback = partial(callback, [bindings], img)
                )


def put_text_with_fringe(img, text, origin, color=(0, 255, 0), scale=1.0, thickness=1, font=cv2.FONT_HERSHEY_PLAIN):
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, (0,0,0), thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, color, thickness//2)
    return img

def postprocess_and_rendering(img:np.ndarray, res):
    img_h, img_w = img.shape[:2]
    for clsid, detections in enumerate(res):
        if len(detections) == 0: continue
        for det in detections:
            bbox, conf = det[:4], det[4]
            if conf < 0.6: continue
            x0 = int(bbox[1] * img_w)
            y0 = int(bbox[0] * img_h)
            x1 = int(bbox[3] * img_w)
            y1 = int(bbox[2] * img_h)
            color = BBOX_COLORS[clsid]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            text = f'{YOLO_LABELS[clsid]} {conf:.2f}'
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
            lx0, ly0 = x0, y0
            lx1, ly1 = lx0 + t_size[0], ly0 - t_size[1]
            cv2.rectangle(img, (lx0, ly0), (lx1, ly1), color, -1)
            cv2.putText(img, text, (lx0, ly0), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    return img

def postprocess():
    global abort_flag
    global inf_result_q
    global render_img_q
    global MODEL_PATH
    infer_count = 0
    fps = 0
    stime = time.perf_counter()
    last_submission_time = time.perf_counter()
    while abort_flag == False:
        while inf_result_q.qsize() == 0:
            time.sleep(10e-3)
        img, res = inf_result_q.get()

        infer_count += 1
        if infer_count % 100 == 0:
            etime = time.perf_counter()
            fps = 1/((etime-stime)/100)
            stime = time.perf_counter()                
            infer_count = 0

        rendered_img = postprocess_and_rendering(img, res)
        text = f'NPU {fps:.2f} FPS'
        put_text_with_fringe(rendered_img, text, (0,0), (0,255,0), 4, 6)
        put_text_with_fringe(rendered_img, MODEL_PATH, (0,600), (0,255,0), 2, 4)
        curr_time = time.perf_counter()
        if curr_time - last_submission_time > 1/30:
            last_submission_time = curr_time
            render_img_q.put(rendered_img)


def render():
    global abort_flag
    global render_img_q
    while abort_flag == False:
        if render_img_q.qsize() == 0:
            time.sleep(10e-3)
        img = render_img_q.get()

        cv2.imshow('result', img)
        key = cv2.waitKey(1)
        if key in [27, ord('Q'), ord('q'), ord(' ')]:
            abort_flag = True
    cv2.destroyAllWindows()





capt_img_q   = queue.Queue()
inf_result_q = queue.Queue()
render_img_q = queue.Queue()

abort_flag = False

th_capt        = threading.Thread(target=capt,        daemon=True)
th_infer       = threading.Thread(target=infer,       daemon=True)
th_postprocess = threading.Thread(target=postprocess, daemon=True)
th_render      = threading.Thread(target=render,      daemon=True)

th_capt.start()
th_infer.start()
th_postprocess.start()
th_render.start()

while abort_flag == False:
    time.sleep(10e-3)
