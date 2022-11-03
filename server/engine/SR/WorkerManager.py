"""WorkerManager impl module."""
from typing import List

from server.base.Worker import BaseWorker
from server.base.WorkerManager import BaseWorkerManager
from server.config.Settings import Settings
from server.engine.SR.Worker import SRWorker
from server.util.types import EngineDevice


class SRWorkerManager(BaseWorkerManager):
    def __init__(
        self,
        setting: Settings,
    ):
        worker_num = setting.server.worker_num
        worker_queue_size = setting.server.worker_queue_size
        devices = [EngineDevice(dev) for dev in setting.server.devices]
        super().__init__(
            worker_num=worker_num, devices=devices, queue_size=worker_queue_size
        )

    def init(self):
        device_num = len(self.devices)
        workers: List[BaseWorker] = []
        for i in range(self.worker_num):
            worker = SRWorker(
                device=self.devices[i % device_num], queue_size=self.worker_queue_size
            )
            worker.start()
            workers.append(worker)

        self.workers = tuple(workers)
