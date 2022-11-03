"""Worker impl module."""
from io import BytesIO
from typing import List

import cv2
import numpy as np
from imutils.video import FileVideoStream
from loguru import logger
from PIL import Image

from server.base.Message import BaseFailed
from server.base.Worker import BaseWorker
from server.engine.detection.Message import (
    Box,
    DetectionInputType,
    DetectionResult,
    DetectionTask,
    FrameData,
    PatchData,
)
from server.engine.detection.Model import DetectionModel
from server.util.types import EngineDevice
from server.util.util import create_timestamp


class DetectionWorker(BaseWorker):
    __model: DetectionModel

    __batch_size: int

    def __init__(
        self, device: EngineDevice = None, batch_size: int = 32, queue_size: int = 100
    ):
        self.__batch_size = batch_size
        if device is None:
            device = EngineDevice()

        super().__init__(queue_size=queue_size, device=device)

    def init(self):
        self.__model = DetectionModel(device=self.device)
        logger.info(f"[{self.__class__.__name__}] worker ready")

    def __transform(self, frame):
        if frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __process_video(self, video_path: str) -> List[FrameData]:

        frame_indices: List[int] = []
        frames: List[np.ndarray] = []

        stream = FileVideoStream(path=video_path, transform=self.__transform).start()

        index = 0
        while stream.running():
            frame = stream.read()  # HWC
            if frame is None:
                continue
            frames.append(frame)
            frame_indices.append(index)
            index += 1

        results: List[FrameData] = [
            FrameData(frame_index=i, patches=[]) for i in range(index)
        ]

        target_length = len(frame_indices)
        for start in range(0, target_length, self.__batch_size):
            end = min(start + self.__batch_size, target_length)
            batch = frames[start:end]

            ts_start = create_timestamp()
            indices, boxes, class_ids, scores, class_names = self.__model(frames=batch)
            ts_end = create_timestamp()

            logger.info(
                f"[{__class__.__name__}] Batch size {len(batch)}: {ts_end - ts_start} ms"
            )

            for index, box, class_id, score, class_name in zip(
                indices, boxes, class_ids, scores, class_names
            ):
                patch = PatchData(
                    box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    class_id=class_id,
                    score=score,
                    class_name=class_name,
                )
                results[start + index].patches.append(patch)

        return results

    def __process_image(self, bimages: List[bytes]) -> List[FrameData]:
        batch = []
        for bimage in bimages:
            image = np.asarray(Image.open(BytesIO(bimage)).convert("RGB"))  # HWC
            batch.append(image)

        num_images = len(batch)

        ts_start = create_timestamp()
        indices, boxes, class_ids, scores, class_names = self.__model(frames=batch)
        ts_end = create_timestamp()

        logger.info(
            f"[{__class__.__name__}] Batch size {len(batch)}: {ts_end - ts_start} ms"
        )

        results: List[FrameData] = [
            FrameData(frame_index=i, patches=[]) for i in range(num_images)
        ]

        for index, box, class_id, score, class_name in zip(
            indices, boxes, class_ids, scores, class_names
        ):
            patch = PatchData(
                box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                class_id=class_id,
                score=score,
                class_name=class_name,
            )
            results[index].patches.append(patch)
        return results

    def process(self, task: DetectionTask):
        if task.input_type == DetectionInputType.IMAGE:
            ret = self.__process_image(task.image_data)
            self.sink.put(DetectionResult(data=ret, task_id=task.id))
        elif task.input_type == DetectionInputType.VIDEO:
            ret = self.__process_video(task.video_path)
            self.sink.put(DetectionResult(data=ret, task_id=task.id))
        else:
            raise RuntimeError(f"Invalid input type: {task.input_type}")

    def handle(self, msg: DetectionTask) -> DetectionResult:
        try:
            self.process(msg)
        except Exception as e:
            self.sink.put(BaseFailed(task_id=msg.id))
            raise RuntimeError(f"Failed to process task `{msg.id}`: {e}")
