"""Loader module."""
import threading
import traceback
from abc import abstractmethod
from queue import Queue
from uuid import uuid4

from loguru import logger

from server.base.Message import BaseModel


class BaseThread(threading.Thread):
    id: str
    queue: Queue
    is_stopped: bool

    _is_ready: bool
    lock: threading.Lock

    def __init__(self, maxsize=-1):
        threading.Thread.__init__(self)

        self.lock = threading.Lock()
        self.is_stopped = False
        self.id = str(uuid4())
        self.queue = Queue(maxsize)
        self.is_ready = False

    def push(self, msg: BaseModel):
        self.queue.put(msg)

    def init(self):
        pass

    @abstractmethod
    def handle(self, msg: BaseModel):
        pass

    def run(self):
        try:
            self.init()
        except Exception as err:
            self.is_stopped = True
            logger.warning(err)
            tb = traceback.format_tb(err.__traceback__)
            for t in tb:
                logger.warning(t)
        self.is_ready = True

        while not self.is_stopped:
            msg = None
            try:
                msg: BaseModel = self.queue.get(timeout=1)
            except:
                pass

            if not msg:
                continue

            try:
                self.handle(msg)
            except Exception as err:
                logger.warning(err)
                tb = traceback.format_tb(err.__traceback__)
                for t in tb:
                    logger.warning(t)

    def stop(self):
        self.is_stopped = True

    def isReady(self):
        return self.is_ready

    @property
    def is_ready(self):
        with self.lock:
            return self._is_ready

    @is_ready.setter
    def is_ready(self, value):
        with self.lock:
            self._is_ready = value
