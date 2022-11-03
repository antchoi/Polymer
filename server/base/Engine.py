"""Base Engine module."""
from abc import ABCMeta, abstractmethod
from typing import Union

from server.base.Message import BaseFailed, BaseResult, BaseTask


class BaseEngine(metaclass=ABCMeta):
    @abstractmethod
    def process(self, task: BaseTask) -> Union[BaseResult, BaseFailed]:
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def isReady(self) -> bool:
        pass
