import time
import threading
import queue

import cv2
import numpy as np

MEDIA_PATH = './data/people.mp4'

g_abort_flag = False



class Ordered_Queue:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.lock.acquire()
        self.data = []
        self.indices = []
        self.next_idx = 0
        self.lock.release()

    def put(self, index:int, data:any) -> int:
        self.lock.acquire()
        self.data.append(data)
        self.indices.append(index)
        len_data = len(self.data)
        len_indices = len(self.indices)
        self.lock.release()
        assert len_data == len_indices
        return len_data

    def get(self, block:bool=True, timeout:float=0.0) -> any:
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
            if timeout > 0.0 and entry_time - time.time() > timeout:
                return None
    
    def qsize(self) -> int:
        return len(self.data)

    def empty(self) -> bool:
        return self.qsize() == 0




def input_gen(q_inf_data):
    global g_abort_flag
    cap = None
    img_id = 0
    while g_abort_flag == False:
        if cap is None:
            cap = cv2.VideoCapture(MEDIA_PATH)
        sts, img = cap.read()
        if sts == False or img is None:
            cap.release()
            cap = None
            continue
        # Preprocess to convert an image data into an input tensor for inference
        img = cv2.resize(img, (640, 480))
        tensor = img.astype(np.float32)

        while q_inf_data.qsize() > 5:
            time.sleep(10e-3)
        q_inf_data.put((img_id, img, tensor))
        img_id += 1


def infer_callback(inf_res, user_data):
    inf_id, img, q_inf_result = user_data
    q_inf_result.put((inf_id, img, inf_res))


def infer(q_inf_data, q_inf_result):
    global g_abort_flag
    img = None
    tensor = None
    inf_id = 0
    while g_abort_flag == False:
        if q_inf_data.qsize() > 0:
            img_id, img, tensor = q_inf_data.get()
        if img is None:
            continue
        
        # ToDo: Inference code here
        time.sleep(1e-3)

        infer_callback(0, (inf_id, img, q_inf_result))  # Dummy callback
        inf_id += 1


def postprocess(q_inf_result, q_output_img):
    global g_abort_flag
    while g_abort_flag == False:
        try:
            inf_id, img, inf_res = q_inf_result.get(block=True, timeout=10e-3)
        except queue.Empty:
            continue
        
        # ToDo: Postprocess & result rendering code here

        result_img = img
        if q_output_img.qsize() > 100:      # Drop the output to avoid excessive queue growth
            continue
        q_output_img.put((inf_id, result_img))


def rendering(q_output_img:queue.Queue):
    global g_abort_flag
    num = 0
    last_rendering_time = time.perf_counter()
    while g_abort_flag == False:
        try:
            inf_id, img = q_output_img.get(block=True, timeout=10e-3)
        except queue.Empty:
            continue
        cv2.putText(img, f'{num}', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        current_time = time.perf_counter()
        if current_time - last_rendering_time > 1/30:
            cv2.imshow('Result', img)
            key = cv2.waitKey(1)
            if key in [ 27, ord('q'), ord(' ') ]:   # 27 == ESC key
                g_abort_flag = True
            last_rendering_time = current_time
        num += 1
    cv2.destroyAllWindows()



def main():
    global g_abort_flag

    q_inf_data = queue.Queue()
    q_inf_result = queue.Queue()
    q_output_img = queue.Queue()

    th_input_gen = threading.Thread(target=input_gen, args=(q_inf_data,), daemon=True)
    th_infer = threading.Thread(target=infer, args=(q_inf_data, q_inf_result), daemon=True)
    th_postprocess = threading.Thread(target=postprocess, args=(q_inf_result, q_output_img), daemon=True)
    th_rendering = threading.Thread(target=rendering, args=(q_output_img,), daemon=True)

    th_input_gen.start()
    th_infer.start()
    th_postprocess.start()
    th_rendering.start()

    while g_abort_flag == False:
        time.sleep(10e-3)
    
    th_input_gen.join()
    th_infer.join()
    th_postprocess.join()
    th_rendering.join()

if __name__ == "__main__":
    main()