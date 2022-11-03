import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger
from PIL import Image
from pydantic.json import pydantic_encoder
from requests import Response

from server.application import Application
from server.config.Settings import Settings
from server.controller import router
from server.payload.actuator.Response import Health
from server.payload.SR.Request import SRRequest
from server.util.types import ServerStatus
from server.util.util import create_timestamp

from .mock import TaskType, prepare_output_dir, prepare_requests, save_image


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

    def test_sr_api(self):
        engine = self.__application.engine_container().sr_engine()

        while not engine.isReady():
            time.sleep(1.0)

        response = self.__client.get("/actuator/health")
        assert response.status_code == 200

        health = Health.parse_obj(response.json())

        assert health.status == ServerStatus.UP.value
        assert health.super_resolution == ServerStatus.UP.value

        requests = prepare_requests(type=TaskType.SuperResolution)

        ts_start = create_timestamp()

        responses: List[Response] = []
        for request in requests:
            payload = json.loads(json.dumps(request, default=pydantic_encoder))
            response = self.__client.post(url="/api/superresolution", json=payload)
            assert response.status_code == 200
            assert response.content is not None
            responses.append(response)

        ts_end = create_timestamp()

        logger.info(f"{len(requests)} requests: {ts_end - ts_start} ms")

        for index, response in enumerate(responses):
            # Save image for test
            image = Image.open(io.BytesIO(response.content))
            save_image(
                data=np.array(image),
                output_dir=self.__output_dir,
                filename=f"{index}.png",
            )

    def test_sr_api_parallel(self):
        engine = self.__application.engine_container().sr_engine()

        while not engine.isReady():
            time.sleep(1.0)

        response = self.__client.get("/actuator/health")
        assert response.status_code == 200

        health = Health.parse_obj(response.json())

        assert health.status == ServerStatus.UP.value
        assert health.super_resolution == ServerStatus.UP.value

        requests = prepare_requests(type=TaskType.SuperResolution)

        def __send(req: SRRequest):
            payload = json.loads(json.dumps(req, default=pydantic_encoder))
            return self.__client.post(url="/api/superresolution", json=payload)

        ts_start = create_timestamp()

        responses: List[Response] = None
        with ThreadPoolExecutor() as pool:
            responses = pool.map(__send, requests)

        ts_end = create_timestamp()

        logger.info(f"{len(requests)} requests: {ts_end - ts_start} ms")

        num_responses = 0
        for resp in responses:
            assert resp.status_code == 200
            num_responses += 1

        assert num_responses == len(requests)

        for index, response in enumerate(responses):
            # Save image for test
            image = Image.open(io.BytesIO(response.content))
            save_image(
                data=np.array(image),
                output_dir=self.__output_dir,
                filename=f"{index}.png",
            )
