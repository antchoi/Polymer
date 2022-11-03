from typing import List, Tuple

import numpy as np
import onnxruntime
from loguru import logger

from server.plugin.yolov7 import CLASS_NAMES, MODEL_PATH, letterbox
from server.util.types import DeviceType, EngineDevice


class DetectionModel:
    __device: EngineDevice
    __session: onnxruntime.InferenceSession

    __input_names: List[str]
    __output_names: List[str]

    def __init__(self, device: EngineDevice):
        self.__device = device

        providers = ["CPUExecutionProvider"]
        if self.__device.mode is DeviceType.GPU:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": str(self.__device.id),
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
            ]

        self.__session = onnxruntime.InferenceSession(MODEL_PATH, providers=providers)
        self.__input_names = [i.name for i in self.__session.get_inputs()]
        self.__output_names = [i.name for i in self.__session.get_outputs()]

    def __call__(
        self, frames: List[np.ndarray]
    ) -> Tuple[List[int], List[List[float]], List[int], List[float], List[str]]:
        input_frames: List[np.ndarray] = []
        ratios = []
        dwdhs = []

        if len(frames) <= 0:
            logger.info(f"[{__class__.__name__}] Empty input")
            return []

        for frame in frames:
            image, ratio, dwdh = letterbox(frame.copy(), auto=False)
            input_frames.append(image)
            ratios.append(ratio)
            dwdhs.append(dwdh)

        frame_batch = np.stack(input_frames).transpose((0, 3, 1, 2))
        frame_batch = np.ascontiguousarray(frame_batch)
        frame_batch = frame_batch.astype(np.float32)
        frame_batch /= 255

        input = {self.__input_names[0]: frame_batch}
        output = self.__session.run(self.__output_names, input)[0]

        indices: List[int] = []
        boxes: List[List[float]] = []  # XYXY
        class_ids: List[int] = []
        scores: List[float] = []
        class_names: List[str] = []

        for batch_id, x0, y0, x1, y1, cls_id, score in output:
            index = int(batch_id)
            indices.append(index)

            dwdh = dwdhs[index]
            ratio = ratios[index]

            box: np.ndarray = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = CLASS_NAMES[cls_id]

            boxes.append(box)
            class_ids.append(cls_id)
            scores.append(score)
            class_names.append(name)

        return indices, boxes, class_ids, scores, class_names
