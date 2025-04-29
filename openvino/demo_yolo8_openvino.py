import os, sys
import copy
import time
import queue
import threading
import random
from logging import getLogger, DEBUG, INFO, WARNING, ERROR, CRITICAL 

logger = getLogger(__name__)
logger.setLevel(INFO)

# Workaround for very slow OpenCV camera opening by VideoCapture() function issue on Windows
if os.name == 'nt':
    logger.info('OS is a Windows. Applied OpenCV workaround.')
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import openvino as ov
import openvino.properties as props 
import openvino.properties.hint as hints 


YOLO_LABELS = ['person', 'bicycle','car','motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

BBOX_COLORS = [[random.randint(64, 255) for _ in range(3)] for _ in range(len(YOLO_LABELS))] 

MOVIE_PATH = '../data/people.mp4'
MODEL_PATH = './yolov8s_with_preprocess.xml'

g_abort_flag = False
g_target_device = '---'
g_fps = 0
g_model_file_name = '---'


class Ordered_Queue:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.lock.acquire()
        self.data = []
        self.indices = []
        self.next_idx = 0
        self.lock.release()

    def put(self, index:int, data:any):
        self.lock.acquire()
        self.data.append(data)
        self.indices.append(index)
        self.lock.release()
        assert len(self.data) == len(self.indices)

    def get(self, block:bool=True, timeout:int=-1):
        entry_time = time.time()
        while True:
            self.lock.acquire()
            if len(self.indices) > 0:
                min_idx = min(self.indices)
                if min_idx == self.next_idx:
                    data_idx = self.indices.index(min_idx)
                    data = self.data.pop(data_idx)
                    _ = self.indices.pop(data_idx)
                    self.next_idx += 1
                    self.lock.release()                
                    assert len(self.data) == len(self.indices)
                    return data
            self.lock.release()
            if block == False:
                return None
            if timeout != -1 and entry_time - time.time() > timeout:
                return None
            time.sleep(1e-3)




def load_model(model_file_name:str, target_device:str='CPU') -> ov.AsyncInferQueue:
    global g_model_file_name
    global g_target_device

    ov_model = ov.Core().read_model(model_file_name)

    # obtain model input information
    input_shape = ov_model.inputs[0].shape
    input_name = list(ov_model.inputs[0].names)[0]
    input_n, input_c, input_h, input_w = input_shape

    # obtain model output information
    #output_shape = ov_model.outputs[0].shape     # yolov8s=1,84,8400, f32
    #output_name = list(ov_model.outputs[0].names)[0]

    # OpenVINO performance optimize parameters and hints
    config={'CACHE_DIR':'./cache'}
    config.update({hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
    config.update({hints.num_requests:"16"})            # number of request queue
    if target_device in ['CPU']:
        config.update({props.inference_num_threads: "16"})  # number of thread used by OpenVINO runtime
    if target_device in ['CPU', 'GPU', 'GPU.0']:
        config.update({props.num_streams: "8"})             # number of simultaneous inference request execution

    model = ov.compile_model(ov_model, device_name=target_device, config=config)

    # Create async queue for easy handling
    async_infer_queue = ov.AsyncInferQueue(model=model, jobs=0) # jobs=0 means, automatic
    async_infer_queue.set_callback(callback)

    g_model_file_name = model_file_name
    g_target_device = target_device

    return async_infer_queue


def put_text_with_fringe(img, text, origin, color=(0, 255, 0), scale=1.0, thickness=1, font=cv2.FONT_HERSHEY_PLAIN):
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, (0,0,0), thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, color, thickness//2)


def thread_render_result(queue_rendered_result):
    global g_abort_flag
    global g_target_device, g_fps, g_model_file_name
    time_last_render_result_update = time.perf_counter()
    while True:
        img = queue_rendered_result.get()

        # send the rendered result every 1/30 sec only
        if time.perf_counter() - time_last_render_result_update < 1/30:  # check if 1/30 sec elapsed
            continue
        time_last_render_result_update = time.perf_counter()

        text = f'{g_target_device} {g_fps:7.2f} FPS'
        put_text_with_fringe(img, text, (0, 0), (0,255,0), 4, 6)
        text = f'Model: {g_model_file_name}'
        put_text_with_fringe(img, text, (0, 600), (0, 255,0), 2, 4)

        cv2.imshow('image', img)
        key = cv2.waitKey(5)       # a little shorter than 1/30 sec
        if key in (27, ord('q'), ord('Q'), ord(' ')):   # 27 is ESC key
            cv2.destroyAllWindows()
            g_abort_flag = True



# thread to capture the input image for inference
def thread_capture_image(queue_image):
    global g_abort_flag
    cap = None
    while g_abort_flag == False:
        if cap is None:
            cap = cv2.VideoCapture(MOVIE_PATH)
        sts, img = cap.read()
        if sts == False or img is None:
            cap.release()
            cap = None
            continue
        img = cv2.resize(img, (640, 640))
        tensor = preprocess(img)
        while queue_image.qsize() > 10: # wait when queue size grown excessively
            time.sleep(10e-3)
        queue_image.put((img, tensor))


# input image preprocess
def preprocess(img):
    tensor = cv2.resize(img, (640, 640))
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    # Yolov8s input layout is (NHWC). No transpose is required
    #tensor = np.transpose(tensor, (2, 0, 1))
    tensor = tensor[np.newaxis, :, :, :]
    return tensor


def postprocess_yolov8(output, conf_threshold=0.2, iou_threshold=0.8):
    """
    Postprocessing for YOLOv8s model output
    :param output: model output (shape: (1, 84, 8400))
    :param input_shape: image size ([height, width])
    :param conf_threshold: threshold for confidences
    :param iou_threshold: threshould for IoU (for NMS)
    :return: bounding boxes and classes after NMS
    """
    # separate the bounding box coordinates (x, y, w, h) and class scores
    boxes = output[0, :4, :]  # (4, 8400)
    class_scores = output[0, 4:, :]  # (80, 8400)

    # Dequantization for fully int8-nized model    
    #boxes = ((boxes.astype(np.float32) + 128.0) / 256.0) * 640
    #class_scores = (class_scores.astype(np.float32) + 128.0) / 256.0

    # get the maximum class score and its index
    class_ids = np.argmax(class_scores, axis=0)  # class IDs of detections
    confidences = np.max(class_scores, axis=0)  # confidence values of detections

    # screen out low confidence boxes
    valid_indices = np.where(confidences > conf_threshold)[0]
    boxes = boxes[:, valid_indices]
    confidences = confidences[valid_indices]
    class_ids = class_ids[valid_indices]

    # convert the bouding box format (x_center, y_center, w, h) → (x0, y0, x1, y1)
    box_xywh = boxes.T
    box_xyxy = np.zeros_like(box_xywh)
    box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2  # x0
    box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2  # y0
    box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2  # x1
    box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2  # y1

    # apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(box_xyxy.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()

    # obtain NMS result
    final_boxes = box_xyxy[indices]
    final_confidences = confidences[indices]
    final_class_ids = class_ids[indices]

    results = []
    for i in range(len(final_boxes)):
        box = final_boxes[i]
        results.append({
            "box": [box[0], box[1], box[2], box[3]],
            "confidence": final_confidences[i],
            "class_id": final_class_ids[i],
        })
    return results

def draw_bboxes(img, bboxes):
    result_image = img.copy()
    img_h, img_w, _ = img.shape
    for bbox in bboxes:
        x0, y0, x1, y1 = [ int(coord) for coord in bbox['box']]
        conf = bbox['confidence']
        clsid = bbox['class_id']
        class_label = YOLO_LABELS[clsid]
        color = BBOX_COLORS[clsid]

        cv2.rectangle(result_image, (x0, y0), (x1, y1), color, 2)

        text = f'{class_label} {conf:.2f}'
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
        lx0, ly0 = x0, y0
        lx1, ly1 = lx0 + t_size[0], ly0 - t_size[1]
        cv2.rectangle(result_image, (lx0, ly0), (lx1, ly1), color, -1)
        cv2.putText(result_image, text, (lx0, ly0), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    return result_image


def thread_postprocess(queue_inference_result, queue_rendered_result):
    while True:
        inf_id, infer_result, img = queue_inference_result.get()

        result_image = img
        bboxes = postprocess_yolov8(infer_result)
        rendered_result = draw_bboxes(result_image, bboxes)

        queue_rendered_result.put(inf_id, rendered_result)


# callback function to receive the asynchronous inference result
def callback(request, userdata):      # userdata == input image for inferencing
    res = list(request.results.values())[0]
    inf_id, img, queue_inference_result = userdata
    #assert queue_inference_result.qsize() < 10
    queue_inference_result.put((inf_id, res, img))

def main():
    global g_fps

    queue_image = queue.Queue()
    #queue_rendered_result = queue.Queue()
    queue_rendered_result = Ordered_Queue()
    queue_inference_result = queue.Queue()

    th_render = threading.Thread(target=thread_render_result, args=(queue_rendered_result,), daemon=True)
    th_postprocess  = threading.Thread(target=thread_postprocess, args=(queue_inference_result, queue_rendered_result,), daemon=True)
    th_input = threading.Thread(target=thread_capture_image, args=(queue_image,), daemon=True)

    th_render.start()
    th_postprocess.start()
    th_input.start()

    devices = ov.Core().available_devices

    image = None
    tensor = None

    # run inference and measure performance
    num_loop = 10
    while g_abort_flag == False:
        for device in devices:
            print(f'Loading model to {device}...', end='', flush=True)
            async_infer_queue = load_model(MODEL_PATH, device)
            print('Completed.')

            inf_id = 0
            queue_rendered_result.reset()

            session_start_time = time.time()
            fps = 0
            while time.time() - session_start_time < 10: # run 10 sec for each device
                stime = time.perf_counter()
                for _ in range(num_loop):
                    if queue_image.qsize() > 0:
                        image, tensor = queue_image.get()
                    if tensor is None or image is None:
                        continue
                    async_infer_queue.start_async(inputs=tensor, userdata=(inf_id, image, queue_inference_result))
                    inf_id += 1
                    if g_abort_flag:
                        return 
                etime = time.perf_counter()
                g_fps = 1/((etime-stime)/num_loop)
            async_infer_queue.wait_all()

if __name__ == '__main__':
    main()

""" Preprocessing
Input "x":
    User's input tensor: [1,640,640,3], [N,H,W,C], u8
    Model's expected tensor: [1,3,?,?], [N,C,H,W], f32
    Pre-processing steps (3):
      convert type (f32): ([1,640,640,3], [N,H,W,C], u8) -> ([1,640,640,3], [N,H,W,C], f32)
      convert layout [N,C,H,W]: ([1,640,640,3], [N,H,W,C], f32) -> ([1,3,640,640], [N,C,H,W], f32)
      scale (255,255,255): ([1,3,640,640], [N,C,H,W], f32) -> ([1,3,640,640], [N,C,H,W], f32)
"""