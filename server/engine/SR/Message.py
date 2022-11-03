"""Parser impl module."""
import numpy as np

from server.base.Message import BaseResult, BaseTask


class SRTask(BaseTask):
    image: str


class SRResult(BaseResult):
    data: np.ndarray
