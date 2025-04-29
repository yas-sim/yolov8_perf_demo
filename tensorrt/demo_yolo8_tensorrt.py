import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import time
import cv2
import numpy as np
import random
import os

import threading
import queue

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

fps = 0
abort_flag = False

#ENGINE_PATH = './yolov8s_b32_fp16.engine'
ENGINE_PATH = './yolov8s_b32_int8.engine'

MOVIE_FILE = '../data/people.mp4'

YOLO_LABELS = ['person', 'bicycle','car','motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

BBOX_COLORS = [[random.randint(64, 255) for _ in range(3)] for _ in range(len(YOLO_LABELS))] 


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
    #boxes = output[0, :4, :]  # (4, 8400)
    #class_scores = output[0, 4:, :]  # (80, 8400)
    boxes = output[:4, :]  # (4, 8400)
    class_scores = output[4:, :]  # (80, 8400)

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


def put_text_with_fringe(img, text, origin, color=(0, 255, 0), scale=1.0, thickness=1, font=cv2.FONT_HERSHEY_PLAIN):
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, (0,0,0), thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, color, thickness//2)
    return img


def render_result(img, bboxes):
    global fps
    result_image = img
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

    text = f'GPU {fps:7.2f} FPS'
    result_image = put_text_with_fringe(result_image, text, (0, 0), (0, 255, 0), 4, 6)
    _, text = os.path.split(ENGINE_PATH)
    result_image = put_text_with_fringe(result_image, text, (0, 600), (0, 255, 0), 2, 4)
    return result_image


def capt(capt_img_q):
    global abort_flag
    cap = None
    while abort_flag == False:
        if cap is None:
            cap = cv2.VideoCapture(MOVIE_FILE)
        sts, img = cap.read()
        if sts == False or img is None:
            cap.release()
            cap = None
            continue
        while capt_img_q.qsize() > 5 and abort_flag == False:
            time.sleep(1e-3)
        img = cv2.resize(img, (640, 640))
        tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(tensor, (2, 0, 1)).astype(np.float32)
        tensor /= 255.0

        while capt_img_q.qsize() > 10:
            time.sleep(10e-3)
        assert capt_img_q.qsize() < 10
        capt_img_q.put((img, tensor))



def postprocess(inf_result_q, render_img_q):
    global abort_flag
    while abort_flag == False:
        try:
            res, img = inf_result_q.get(block=True, timeout=10e-3)
        except queue.Empty:
            continue

        bboxes = postprocess_yolov8(res)
        img = render_result(img, bboxes)

        if render_img_q.qsize() > 5:
            continue        # drop the result
        assert render_img_q.qsize() < 10
        render_img_q.put(img)


def render(render_img_q):
    global abort_flag
    last_time_render = time.perf_counter()
    while abort_flag == False:
        try:
            img = render_img_q.get(block=True, timeout=10e-3)
        except queue.Empty:
            continue
        curr_time = time.perf_counter()
        if curr_time - last_time_render >= 1/30:  # limit the refresh rate
            last_time_render = curr_time
            cv2.imshow('Result', img)
            key = cv2.waitKey(1)
            if key == 27:
                abort_flag = True
    cv2.destroyAllWindows()

from functools import reduce

def allocate_buffers(engine, context):
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    input_size = reduce(lambda x, y: x * y, input_shape)
    output_size = reduce(lambda x, y: x * y, input_shape)

    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

    h_input = np.empty(input_shape, dtype=input_dtype)
    h_output = np.empty(output_shape, dtype=output_dtype)
    #h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
    #h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    context.set_input_shape(input_name, input_shape)
    assert context.all_binding_shapes_specified

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    return h_input, h_output, d_input, d_output, input_shape


def infer(capt_img_q, inf_result_q):
    global fps
    global ENGINE_PATH
    global abort_flag

    runtime = trt.Runtime(TRT_LOGGER)
    with open(ENGINE_PATH, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    h_input, h_output, d_input, d_output, input_shape = allocate_buffers(engine, context)

    stream = cuda.Stream()

    tensor = None
    num_infer = 0
    fps = 0

    num_batch = input_shape[0]
    print(f'Model: {ENGINE_PATH}, Batch: {num_batch}')
    cnt_batch = 0
    tensor_buf = np.empty((num_batch, 3, 640, 640), dtype=np.float32)
    img_buf = []

    img = np.empty((640,630,3), dtype=np.uint8)
    pred = np.zeros((84, 8400), dtype=np.float32)

    stime = time.perf_counter()
    while abort_flag == False:
        if capt_img_q.qsize() > 0:
            img, tensor = capt_img_q.get()
        if tensor is None:
            continue

        tensor_buf[cnt_batch,:,:,:] = tensor
        img_buf.append(img)

        cnt_batch += 1
        if cnt_batch < num_batch:
            continue

        cnt_batch = 0
        h_input = np.ascontiguousarray(tensor_buf)

        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        num_infer += 1
        if num_infer == 10:
            etime = time.perf_counter()            
            num_infer = 0
            fps = 1.0 / ((etime-stime)/10)
            fps *= num_batch
            stime = etime

        for b in range(num_batch):
            img = img_buf[b]
            pred = h_output[b]
            assert inf_result_q.qsize() < num_batch*2, inf_result_q.qsize()
            inf_result_q.put((pred, img))
        img_buf = []


def main():
    global abort_flag

    capt_img_q = queue.Queue()
    inf_result_q = queue.Queue()
    render_img_q = queue.Queue()

    capt_th = threading.Thread(target=capt, args=(capt_img_q,), daemon=True)
    postprocess_th = threading.Thread(target=postprocess, args=(inf_result_q, render_img_q), daemon=True)
    render_th = threading.Thread(target=render, args=(render_img_q,), daemon=True)

    capt_th.start()
    postprocess_th.start()
    render_th.start()

    # Inference code can't be thread-tize.
    infer(capt_img_q, inf_result_q)

    abort_flag = True
    capt_th.join()
    postprocess_th.join()
    render_th.join()

if __name__ == '__main__':
    main()

