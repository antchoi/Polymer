import json
import random
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger
from PIL import Image
from pydantic import parse_obj_as
from requests import Response

from server.application import Application
from server.config.Settings import Settings
from server.controller import router
from server.payload.actuator.Response import Health
from server.payload.detection.Response import DetectionResponse
from server.plugin.yolov7 import CLASS_NAMES
from server.util.types import ServerStatus
from server.util.util import create_timestamp

from .mock import TaskType, prepare_output_dir, prepare_requests

CLASS_COLORS = {
    name: [random.randint(0, 255) for _ in range(3)]
    for i, name in enumerate(CLASS_NAMES)
}


class TestApi:
    __setting: Settings
    __client: TestClient
    __application: Application
    __output_dir: str

    def setup_class(self):
        self.__setting = Settings()
        self.__output_dir = prepare_output_dir("api")

        self.__application = Application(setting=self.__setting)
        wiring_modules = self.__application.wiring_config.modules
        wiring_modules.append(__name__)
        self.__application.wire(modules=wiring_modules)

        server = FastAPI()
        server.include_router(router)

        self.__client = TestClient(server)

    def teardown_class(self):
        self.__application.engine_container().sr_engine().stop()
        self.__application.engine_container().detection_engine().stop()

    def test_detect_image(self):
        output_dir = f"{self.__output_dir}/detection/image"

        shutil.rmtree(output_dir, ignore_errors=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        engine = self.__application.engine_container().detection_engine()

        while not engine.isReady():
            time.sleep(1.0)

        response = self.__client.get("/actuator/health")
        assert response.status_code == 200

        health = Health.parse_obj(response.json())

        assert health.status == ServerStatus.UP.value
        assert health.detection == ServerStatus.UP.value

        requests = prepare_requests(type=TaskType.DetectionImage)

        ts_start = create_timestamp()

        responses: List[Response] = []
        for request in requests:
            response = self.__client.post(url="/api/detection/image", files=request)
            responses.append(response)

        ts_end = create_timestamp()

        logger.info(f"{len(requests)} requests: {ts_end - ts_start} ms")

        requests = prepare_requests(type=TaskType.DetectionImage)
        for request, response in zip(requests, responses):
            assert response.status_code == 200

            image_bytes = request["file"].read()
            image = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            result = parse_obj_as(DetectionResponse, json.loads(response.content))

            for patch in result.patches:
                logger.info(
                    f"class {patch.class_name}, class_id: {patch.class_id}, score: {patch.score}, box: {patch.box}"
                )

                color = CLASS_COLORS[patch.class_name]
                patch.class_name += " " + str(patch.score)

                x1, y1, x2, y2 = (
                    int(patch.box.x1),
                    int(patch.box.y1),
                    int(patch.box.x2),
                    int(patch.box.y2),
                )
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )
                cv2.putText(
                    image,
                    patch.class_name,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    [225, 255, 255],
                    thickness=2,
                )

            cv2.imwrite(f"{output_dir}/output.jpg", image)

    def test_detect_images(self):
        output_dir = f"{self.__output_dir}/detection/images"

        shutil.rmtree(output_dir, ignore_errors=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        engine = self.__application.engine_container().detection_engine()

        while not engine.isReady():
            time.sleep(1.0)

        response = self.__client.get("/actuator/health")
        assert response.status_code == 200

        health = Health.parse_obj(response.json())

        assert health.status == ServerStatus.UP.value
        assert health.detection == ServerStatus.UP.value

        request = prepare_requests(type=TaskType.DetectionImages)

        ts_start = create_timestamp()

        response = self.__client.post(url="/api/detection/images", files=request)

        ts_end = create_timestamp()

        logger.info(f"{len(request)} images: {ts_end - ts_start} ms")

        assert response.status_code == 200

        request = prepare_requests(type=TaskType.DetectionImages)
        image_bytes = [f.read() for _, f in request]
        images = [
            cv2.cvtColor(np.asarray(Image.open(BytesIO(im))), cv2.COLOR_RGB2BGR)
            for im in image_bytes
        ]
        results = parse_obj_as(List[DetectionResponse], json.loads(response.content))

        for result in results:
            index = result.frame_index
            image = images[index]
            for patch in result.patches:
                logger.info(
                    f"[Index {index}] class {patch.class_name}, class_id: {patch.class_id}, score: {patch.score}, box: {patch.box}"
                )

                color = CLASS_COLORS[patch.class_name]
                patch.class_name += " " + str(patch.score)

                x1, y1, x2, y2 = (
                    int(patch.box.x1),
                    int(patch.box.y1),
                    int(patch.box.x2),
                    int(patch.box.y2),
                )
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )
                cv2.putText(
                    image,
                    patch.class_name,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    [225, 255, 255],
                    thickness=2,
                )

        for i, image in enumerate(images):
            cv2.imwrite(f"{output_dir}/{i}.jpg", image)

    def test_detect_video(self):
        output_dir = f"{self.__output_dir}/detection/video"

        shutil.rmtree(output_dir, ignore_errors=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        engine = self.__application.engine_container().detection_engine()

        while not engine.isReady():
            time.sleep(1.0)

        response = self.__client.get("/actuator/health")
        assert response.status_code == 200

        health = Health.parse_obj(response.json())

        assert health.status == ServerStatus.UP.value
        assert health.detection == ServerStatus.UP.value

        requests = prepare_requests(type=TaskType.DetectionVideo)

        ts_start = create_timestamp()

        responses: List[Response] = []
        for request in requests:
            response = self.__client.post(url="/api/detection/video", files=request)
            responses.append(response)

        ts_end = create_timestamp()

        logger.info(f"{len(requests)} requests: {ts_end - ts_start} ms")

        for request, response in zip(requests, responses):
            assert response.status_code == 200
