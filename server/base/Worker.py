"""Worker module."""
from queue import Queue

from server.base.Thread import BaseThread
from server.util.types import EngineDevice


class BaseWorker(BaseThread):
    device: EngineDevice
    sink: Queue
    id: str

    def __init__(self, queue_size: int, device: EngineDevice):
        super().__init__(queue_size)
        self.device = device
        self.sink = Queue()
