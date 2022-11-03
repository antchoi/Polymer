"""BaseWorkerManager module."""
import threading
from typing import Tuple

from loguru import logger

from server.base.Worker import BaseWorker
from server.util.types import EngineDevice


class BaseWorkerManager:
    devices: Tuple[EngineDevice]
    workers: Tuple[BaseWorker]

    worker_num: int
    worker_index: int
    worker_queue_size: int

    lock: threading.Lock

    def __init__(
        self,
        worker_num: int = 1,
        devices: Tuple[EngineDevice] = None,
        queue_size: int = -1,
    ):
        self.worker_index = 0
        self.lock = threading.Lock()

        self.worker_num = worker_num
        self.worker_queue_size = queue_size

        self.devices = devices

        if self.devices is None or len(self.devices) <= 0:
            self.devices = [EngineDevice()]

        self.init()

    def init(self):
        raise NotImplementedError()

    def next(self) -> BaseWorker:
        with self.lock:
            worker_index = self.worker_index
            self.worker_index = (self.worker_index + 1) % self.worker_num
        return self.workers[worker_index]

    def stop(self):
        for worker in self.workers:
            worker.stop()

        for worker in self.workers:
            worker.join()

        self.workers = None

    def isReady(self) -> bool:
        for worker in self.workers:
            if not worker.isReady():
                return False
        return True
