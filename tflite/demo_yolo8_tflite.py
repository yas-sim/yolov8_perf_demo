import os
import random
import threading
import queue
import time
import copy

import cv2
import numpy as np

#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite

abort_flag = False
img_id = 0
lock_img_id = threading.Lock()

yolo_labels = ['person', 'bicycle','car','motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

bbox_colors = [[random.randint(64, 255) for _ in range(3)] for _ in range(len(yolo_labels))] 

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
    boxes = ((boxes.astype(np.float32) + 128.0) / 256.0) * 640
    class_scores = (class_scores.astype(np.float32) + 128.0) / 256.0
    #boxes = boxes.astype(np.float32) * 640
    #class_scores = (class_scores.astype(np.float32) + 128.0) / 256.0

    # get the maximum class score and its index
    class_ids = np.argmax(class_scores, axis=0)  # class IDs of detections
    confidences = np.max(class_scores, axis=0)  # confidence values of detections
    #print(confidences)

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

    #print(final_boxes)

    results = []
    for i in range(len(final_boxes)):
        box = final_boxes[i]
        if final_confidences[i] > 0.3:
            results.append({
                "box": [box[0], box[1], box[2], box[3]],
                "confidence": final_confidences[i],
                "class_id": final_class_ids[i],
            })
    return results

def get_img_id():
    global lock_img_id, img_id
    lock_img_id.acquire()
    res = img_id
    img_id += 1
    lock_img_id.release()
    return res

def reset_img_id():
    global lock_img_id, img_id
    lock_img_id.acquire()
    img_id = 0
    lock_img_id.release()


def thread_input_stream(queue_input:queue.Queue):
    reset_img_id()

    cap = None
    while True:
        if cap is None:
            cap = cv2.VideoCapture(MEDIA_PATH)
        sts, img = cap.read()
        if sts == False:
            cap.release()
            cap = None
            continue
        img = cv2.resize(img, (640, 640))
        tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = tensor[np.newaxis, :, :, :]
        tensor = tensor.astype(np.int32) - 128
        tensor = tensor.astype(np.int8)
        # 0.01865844801068306 * (q + 14)

        iid = get_img_id()
        while queue_input.qsize() > 5:
            time.sleep(10e-3)
            continue
        queue_input.put((img_id, img, tensor))


def put_text_with_fringe(img, text, origin, color=(0, 255, 0), scale=1.0, thickness=1, font=cv2.FONT_HERSHEY_PLAIN):
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, (0,0,0), thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, color, thickness//2)
    return img

def render_result(img, bboxes, fps):
    result_image = img.copy()
    img_h, img_w, _ = img.shape
    for bbox in bboxes:
        x0, y0, x1, y1 = [ int(coord) for coord in bbox['box']]
        conf = bbox['confidence']
        clsid = bbox['class_id']
        class_label = yolo_labels[clsid]
        color = bbox_colors[clsid]

        cv2.rectangle(result_image, (x0, y0), (x1, y1), color, 2)

        text = f'{class_label} {conf:.2f}'
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
        lx0, ly0 = x0, y0
        lx1, ly1 = lx0 + t_size[0], ly0 - t_size[1]
        cv2.rectangle(result_image, (lx0, ly0), (lx1, ly1), color, -1)
        cv2.putText(result_image, text, (lx0, ly0), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    text = f'{DEVICE} {fps:7.2f} FPS'
    result_image = put_text_with_fringe(result_image, text, (0, 0), (0, 255, 0), 4, 6)
    _, text = os.path.split(MODEL_PATH)
    result_image = put_text_with_fringe(result_image, text, (0, 600), (0, 255, 0), 2, 4)
    return result_image


def thread_postprocess_and_display(queue_output:queue.Queue):
    global abort_flag

    fps = 0
    measure_iter = 10
    stime = time.perf_counter()
    iter = 0
    last_img_id = -1

    window_name = 'Result'
    cv2.namedWindow(window_name)
    last_reder_time = time.perf_counter()
    while True:
        while queue_output.qsize() == 0:
            time.sleep(1e-3)
        data = queue_output.get()
        iid, infer_result, img = data
        bboxes = postprocess_yolov8(infer_result)
        rendered_result = render_result(img, bboxes, fps)

        # measure fps
        iter += 1
        if iter >= measure_iter:
            etime = time.perf_counter()
            fps = 1/((etime-stime)/measure_iter)
            stime = copy.copy(etime)
            iter = 0

        current_time = time.perf_counter()

        if iid <= last_img_id:
            continue            # ignore past frames

        last_img_id = iid
        if current_time - last_reder_time > 1/30:
            last_reder_time = copy.copy(current_time)
            cv2.imshow(window_name, rendered_result)
            key = cv2.waitKey(1)
            if key in [27, ord('q'), ord('Q')]:
                abort_flag = True
                return
            time.sleep(5e-3)

def thread_infer(model_path, queue_input, queue_output):
    global abort_flag

    match DEVICE:
        case 'NPU':
            delegate_options = { 'use_npu' : True }
            os.environ['USE_GPU_INFERENCE'] = '0'
            npu_delegate = tflite.load_delegate('libvx_delegate.so', options=delegate_options)
            interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[npu_delegate]) 
        case 'GPU':
            delegate_options = { 'use_npu' : False }
            os.environ['USE_GPU_INFERENCE'] = '1'
            npu_delegate = tflite.load_delegate('libvx_delegate.so', options=delegate_options)
            interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[npu_delegate]) 
        case 'CPU':
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
        case _:
            abort_flag = True
            raise ValueError('Device must be one of "CPU", "GPU", or "NPU".')

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #print(input_details, output_details)

    # Prepare the input data
    # Replace `your_input_data` with the actual input data, ensuring it matches the model's expected format and dimensions
    input_shape = input_details[0]['shape']

    image = None
    tensor = None

    while abort_flag == False:
        print('.', end='', flush=True)
        if queue_input.qsize() > 0:
            iid, image, tensor = queue_input.get()
        if image is None:
            time.sleep(5e-3)
            continue

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], tensor)

        # Perform inference
        interpreter.invoke()

        # Retrieve the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])    # Yolov8s = (1, 84, 8400)
        queue_output.put((iid, output_data, image))


MEDIA_PATH = '../data/people.mp4'

MODEL_PATH = "./yolov8s_saved_model/yolov8s_full_integer_quant.tflite"

DEVICE = 'CPU'      # 'CPU', 'GPU', 'NPU'

queue_input = queue.Queue()
queue_output = queue.Queue()

num_threads = 4
th_input = threading.Thread(target=thread_input_stream, args=(queue_input,), daemon=True)
th_output = threading.Thread(target=thread_postprocess_and_display, args=(queue_output,), daemon=True)
th_infer_list = [ threading.Thread(target=thread_infer, args=(MODEL_PATH, queue_input, queue_output), daemon=True) for _ in range(num_threads) ]

th_input.start()
th_output.start()
for th_infer in th_infer_list:
    th_infer.start()

while abort_flag == False:
    time.sleep(10e-3)
