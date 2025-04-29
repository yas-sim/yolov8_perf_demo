import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from pathlib import Path
import random

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

class DefaultEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir: str, cache_file: str, batch_size: int):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.image_path_list = list(Path(image_dir).iterdir())
        self.cache_file_path = Path(cache_file)
        self.batch_size = batch_size

        self.calib_loop = 100
        self.loop_index = 0

        n_image_bytes = np.empty((1, 256, 512, 3), np.float32).nbytes
        self.device_input_mem = cuda.mem_alloc(n_image_bytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def load_image(self, image_file_path: Path):
        image = cv2.imread(str(image_file_path))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(640, 640))[640 // 2 :]
        image = image / 255.0
        return image

    def get_batch(self, _):
        if self.loop_index == self.calib_loop:
            return None

        path_list = random.sample(
        self.image_path_list, self.batch_size)
        current_batch = np.array([self.load_image(x) for x in path_list], np.float32)
        cuda.memcpy_htod(self.device_input_mem, current_batch)
        self.loop_index += 1
        print(f"Calib loop [{self.loop_index}]")
        return [self.device_input_mem]

    def read_calibration_cache(self):
        if self.cache_file_path.exists():
            with self.cache_file_path.open("rb") as f:
                return f.read()

    def write_calibration_cache(self, cache_bytes: bytes):
        with self.cache_file_path.open("wb") as f:
            f.write(cache_bytes)


def quantize(input_model_path, output_model_path, batch_size=1, images_dir='../data/imagenet', calib_cache_file='./calibration.cache'):
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    EXPLICIT_BATCH_FLAG = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH_FLAG)

    parser = trt.OnnxParser(network, logger)
    is_success = parser.parse_from_file(input_model_path)
    print("ONNX model parsing:", is_success)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = DefaultEntropyCalibrator(image_dir=images_dir, cache_file=calib_cache_file, batch_size=batch_size)

    trt_engine_bytes = builder.build_serialized_network(network, config)

    with open(output_model_path, "wb") as f:
        f.write(trt_engine_bytes)

for batch in (1, 32):
    input_model_path = f'yolov8s_b{batch}.onnx'
    output_model_path = f'yolov8s_b{batch}_int8.engine'
    quantize(input_model_path, output_model_path, batch_size=batch)
