import math
import random
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
from imutils.video import FileVideoStream
from loguru import logger

from server.engine.detection.Model import DetectionModel
from server.plugin.yolov7 import CLASS_NAMES
from server.unittest.mock import (
    prepare_detection_images,
    prepare_detection_videos,
    prepare_output_dir,
)
from server.util.types import EngineDevice
from server.util.util import create_timestamp

CLASS_COLORS = {
    name: [random.randint(0, 255) for _ in range(3)]
    for i, name in enumerate(CLASS_NAMES)
}


def transform(frame):
    if frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


class TestAlgorithm:
    __output_dir: str

    __model: DetectionModel

    def setup_class(self):
        self.__model = DetectionModel(device=EngineDevice(0))
        self.__output_dir = prepare_output_dir("algorithm")

    def detect_video(self, video_path: str, batch_size: int = 32, task_index: int = 0):
        output_dir = f"{self.__output_dir}/video-{task_index}"

        shutil.rmtree(output_dir, ignore_errors=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        frame_indices: List[int] = []
        frames: List[np.ndarray] = []

        stream = FileVideoStream(path=video_path, transform=transform).start()

        index = 0
        while stream.running():
            frame = stream.read()  # HWC
            if frame is None:
                continue
            frames.append(frame)
            frame_indices.append(index)
            index += 1

        logger.info(f"Total {index} frames")

        target_length = len(frame_indices)
        math.ceil(target_length / batch_size)

        images = []

        for start in range(0, target_length, batch_size):
            end = min(start + batch_size, target_length)
            batch = frames[start:end]

            ts_start = create_timestamp()
            indices, boxes, class_ids, scores, class_names = self.__model(frames=batch)
            ts_end = create_timestamp()

            logger.info(f"Batch size {len(batch)}: {ts_end - ts_start} ms")

            for index, box, class_id, score, class_name in zip(
                indices, boxes, class_ids, scores, class_names
            ):
                logger.info(
                    f"Frame {start + index}: class {class_name}, class_id: {class_id}, score: {score}"
                )
                image = batch[index]
                # image = batch[index]
                color = CLASS_COLORS[class_name]
                class_name += " " + str(score)
                cv2.rectangle(image, box[:2], box[2:], color, 2)
                cv2.putText(
                    image,
                    class_name,
                    (box[0], box[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    [225, 255, 255],
                    thickness=2,
                )
            images.extend(batch)

        for index, image in enumerate(images):
            img_encoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{output_dir}/{index}.jpg", img_encoded)

    def detect_image(self, image_path: str, task_index: int = 0):
        output_dir = f"{self.__output_dir}/image-{task_index}"

        shutil.rmtree(output_dir, ignore_errors=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        batch = [image]

        ts_start = create_timestamp()
        indices, boxes, class_ids, scores, class_names = self.__model(frames=batch)
        ts_end = create_timestamp()

        logger.info(f"Image shape `{image.shape}`: {ts_end - ts_start} ms")

        output_image = image.copy()
        for index, box, class_id, score, class_name in zip(
            indices, boxes, class_ids, scores, class_names
        ):
            logger.info(
                f"class {class_name}, class_id: {class_id}, score: {score}, box: {box}"
            )
            color = CLASS_COLORS[class_name]
            class_name += " " + str(score)
            cv2.rectangle(output_image, box[:2], box[2:], color, 2)
            cv2.putText(
                output_image,
                class_name,
                (box[0], box[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                [225, 255, 255],
                thickness=2,
            )

        img_encoded = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_dir}/output.jpg", img_encoded)

    def test_detect_images(self):
        samples = prepare_detection_images()
        for i, sample in enumerate(samples):
            self.detect_image(sample, i)

    def test_detect_videos(self):
        samples = prepare_detection_videos()
        for i, sample in enumerate(samples):
            self.detect_video(sample, 32, i)
