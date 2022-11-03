import time
from typing import List, Tuple

import numpy as np
import pytest
from loguru import logger

from server.base.Engine import BaseEngine
from server.base.Message import BaseResult
from server.config.Settings import Settings
from server.engine.SR.Engine import SREngine
from server.engine.SR.Message import SRResult
from server.payload.SR.Request import SRRequest, TaskRequest
from server.util.types import DeviceType

from .mock import prepare_requests, prepare_setting, save_image


@pytest.fixture
def target_settings() -> Tuple[Settings]:
    result: List[Settings] = []

    s1 = prepare_setting(tag="engine_with_cpu")
    s1.server.devices = [DeviceType.CPU.name.lower()]
    s1.server.worker_num = 4

    result.append(s1)

    s2 = prepare_setting(tag="engine_with_gpu")
    s2.server.worker_num = 4

    result.append(s2)

    return tuple(result)


def run_engine(setting: Settings, requests: List[SRRequest]) -> List[BaseResult]:
    engine: BaseEngine = SREngine(setting=setting)

    cnt = 0
    while not engine.isReady():
        time.sleep(1)
        cnt += 1
        logger.info(f"Waiting for engine... {cnt}")

    logger.info(f"Engine with {engine.worker_manager.worker_num} workers ready")

    results: List[BaseResult] = []

    for request in requests:
        results.append(engine.process(request))

    engine.stop()

    return results


def test_engine(target_settings: Tuple[Settings]):
    requests, filenames = prepare_requests()

    for setting in target_settings:
        output_dir = setting.server.output_dir
        output_basedir = setting.server.output_basedir
        output_fulldir = f"{output_dir}/{output_basedir}"

        results = run_engine(setting=setting, requests=requests)

        for result, filename in zip(results, filenames):
            assert result.type == MessageType.RETURN_RESULT

            result: Result = result
            img_ndarray: np.ndarray = result.data

            # Save image for test
            save_image(data=img_ndarray, output_dir=output_fulldir, filename=filename)
