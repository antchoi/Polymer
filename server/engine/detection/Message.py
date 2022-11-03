"""Parser impl module."""
from enum import IntEnum, auto
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from server.base.Message import BaseResult, BaseTask


class DetectionInputType(IntEnum):
    IMAGE = auto()
    VIDEO = auto()


class DetectionTask(BaseTask):
    input_type: DetectionInputType
    image_data: Optional[List[bytes]]
    video_path: Optional[str]


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PatchData(BaseModel):
    box: Box
    class_id: int
    score: float
    class_name: Optional[str]


class FrameData(BaseModel):
    frame_index: Optional[int]
    patches: List[PatchData]


class DetectionResult(BaseResult):
    data: List[FrameData]
