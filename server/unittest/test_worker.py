import os
import time
from typing import List, Tuple

import numpy as np
import pytest
import torch
from loguru import logger

from server.base.Worker import BaseWorker
from server.config.Settings import Settings
from server.engine.impl.MessageImpl import MessageType, Result, Task
from server.engine.impl.WorkerImpl import WorkerImpl
from server.engine.Message import BaseMessage
from server.payload.SR.Request import TaskRequest
from server.util.types import DeviceType, EngineDevice

from .mock import prepare_requests, prepare_setting, save_image


@pytest.fixture
def target_devices() -> Tuple[EngineDevice]:
    devices: List[EngineDevice] = []

    devices.append(EngineDevice())

    setting = Settings()

    if len(setting.server.devices) > 0 and setting.server.devices[0] != "cpu":
        devices.append(EngineDevice(setting.server.devices[0]))

    return tuple(devices)


def run_worker(
    device: EngineDevice,
    requests: List[TaskRequest],
    output_dir: str,
    output_basedir: str,
) -> List[BaseMessage]:
    worker: BaseWorker = WorkerImpl(device=device)
    worker.start()

    cnt = 0
    while not worker.isReady():
        time.sleep(1)
        cnt += 1
        logger.info(f"Waiting for worker... {cnt}")

    logger.info("Worker ready")

    results: List[BaseMessage] = []
    for request in requests:
        task = Task(
            output_dir=output_dir, output_basedir=output_basedir, image=request.data
        )

        worker.push(task)
        results.append(worker.sink.get())

    worker.stop()

    return results


def test_worker(target_devices: Tuple[EngineDevice]):
    setting = prepare_setting("worker")
    requests, filenames = prepare_requests()

    output_dir = setting.server.output_dir
    output_basedir = setting.server.output_basedir
    output_fulldir = f"{output_dir}/{output_basedir}"

    for device in target_devices:
        os.environ["NVIDIA_VISIBLE_DEVICES"] = (
            str(device) if device.mode == DeviceType.GPU else ""
        )
        invariant_env = os.environ["NVIDIA_VISIBLE_DEVICES"]

        results = run_worker(
            device=device,
            requests=requests,
            output_dir=output_dir,
            output_basedir=output_basedir,
        )

        assert os.environ["NVIDIA_VISIBLE_DEVICES"] == invariant_env

        for result, filename in zip(results, filenames):
            assert result.type == MessageType.RETURN_RESULT
            result: Result = result

            img_ndarray: np.ndarray = result.data

            # Save image for test
            save_image(data=img_ndarray, output_dir=output_fulldir, filename=filename)
