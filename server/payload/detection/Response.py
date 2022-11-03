from typing import List, Optional

import cv2
from fastapi import Response
from pydantic import BaseModel

from server.engine.detection.Message import PatchData
from server.engine.SR.Message import SRResult


class SRResponse(Response):
    def __init__(self, result: SRResult):
        _, data_png = cv2.imencode(".png", cv2.cvtColor(result.data, cv2.COLOR_RGB2BGR))
        super().__init__(content=data_png.tobytes(), media_type="image/png")


class DetectionResponse(BaseModel):
    frame_index: Optional[int]
    patches: List[PatchData]
