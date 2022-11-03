"""Engine impl module."""
from typing import Union

from server.base.Engine import BaseEngine
from server.base.Message import BaseFailed
from server.base.WorkerManager import BaseWorkerManager
from server.config.Settings import Settings
from server.engine.SR.Message import SRResult, SRTask


class SREngine(BaseEngine):

    __worker_manager: BaseWorkerManager

    def __init__(self, setting: Settings, worker_manager: BaseWorkerManager):
        self.__worker_manager = worker_manager

    def process(self, task: SRTask) -> Union[SRResult, BaseFailed]:
        worker = self.__worker_manager.next()
        worker.push(task)
        return worker.sink.get()

    def stop(self):
        self.__worker_manager.stop()

    def isReady(self) -> bool:
        return self.__worker_manager.isReady()
