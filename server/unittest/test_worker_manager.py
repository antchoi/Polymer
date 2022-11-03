import time
from typing import List, Tuple

import numpy as np
import pytest
from loguru import logger

from server.base.Worker import BaseWorker
from server.base.WorkerManager import BaseWorkerManager
from server.config.Settings import Settings
from server.engine.impl.MessageImpl import MessageType, Result, Task
from server.engine.impl.WorkerManagerImpl import WorkerManagerImpl
from server.engine.Message import BaseMessage
from server.payload.SR.Request import TaskRequest
from server.util.types import EngineDevice

from .mock import prepare_requests, prepare_setting, save_image


@pytest.fixture
def target_devices() -> Tuple[Tuple[EngineDevice]]:
    devices: List[EngineDevice] = []

    devices.append(tuple())

    setting = Settings()
    devices.append(tuple(EngineDevice(dev) for dev in setting.server.devices))

    return tuple(devices)


def run_worker_manager(
    worker_num: int,
    requests: List[TaskRequest],
    output_dir: str,
    output_basedir: str,
    devices: Tuple[EngineDevice] = None,
) -> List[BaseMessage]:
    worker_manager: BaseWorkerManager = WorkerManagerImpl(
        worker_num=worker_num, devices=devices
    )

    cnt = 0
    while not worker_manager.isReady():
        time.sleep(1)
        cnt += 1
        logger.info(f"Waiting for workers... {cnt}")

    logger.info("All workers ready")

    results: List[BaseMessage] = []
    for request in requests:
        task = Task(
            output_dir=output_dir, output_basedir=output_basedir, image=request.data
        )

        worker: BaseWorker = worker_manager.next()
        worker.push(task)
        results.append(worker.sink.get())

    worker_manager.stop()

    return results


def test_worker_manager(target_devices: Tuple[Tuple[EngineDevice]]):
    setting = prepare_setting("worker_manager")
    requests, filenames = prepare_requests()
    worker_num = setting.server.worker_num

    output_dir = setting.server.output_dir
    output_basedir = setting.server.output_basedir
    output_fulldir = f"{output_dir}/{output_basedir}"

    for devices in target_devices:
        results = run_worker_manager(
            worker_num=worker_num,
            requests=requests,
            devices=devices,
            output_dir=output_dir,
            output_basedir=output_basedir,
        )

        for result, filename in zip(results, filenames):
            assert result.type == MessageType.RETURN_RESULT

            result: Result = result

            img_ndarray: np.ndarray = result.data

            # Save image for test
            save_image(data=img_ndarray, output_dir=output_fulldir, filename=filename)
