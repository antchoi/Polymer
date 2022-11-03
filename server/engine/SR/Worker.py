"""Worker impl module."""
import base64
from io import BytesIO

import numpy as np
import torch
from loguru import logger
from PIL import Image

from server.base.Message import BaseFailed
from server.base.Worker import BaseWorker
from server.engine.SR.Message import SRResult, SRTask
from server.engine.SR.Model import SRModel
from server.util.types import EngineDevice
from server.util.util import create_timestamp


class SRWorker(BaseWorker):
    model: SRModel

    def __init__(
        self,
        device: EngineDevice = None,
        queue_size: int = 100,
    ):
        if device is None:
            device = EngineDevice()

        super().__init__(queue_size=queue_size, device=device)

    def init(self):
        self.model = SRModel(device=self.device)

    def process(self, msg: SRTask):
        ts_start = create_timestamp()

        task_id = msg.id
        image_base64 = msg.image
        del msg

        logger.info(f"Task `{str(task_id)}` started")

        imgdata = base64.b64decode(image_base64)
        img_ndarray = np.asarray(Image.open(BytesIO(imgdata)).convert("RGB"))  # HWC

        img_tensor = img_ndarray / 255.0
        img_tensor = torch.from_numpy(img_tensor).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # 1CHW

        output = self.model(img_tensor)

        img_output: np.ndarray = (
            output.detach().cpu().squeeze().permute(1, 2, 0).numpy() * 255.0
        )

        img_output = np.array(img_output)

        result = SRResult(task_id=task_id, data=img_output)

        ts_end = create_timestamp()

        self.sink.put(result)

        elapsed = (ts_end - ts_start) / 1000.0

        logger.info(
            f"[{self.device.device_model}] Task `{str(task_id)}` with image shape `{img_ndarray.shape}` ended: ({elapsed:.3f} seconds elapsed)"
        )

    def handle(self, msg: SRTask) -> SRResult:
        try:
            self.process(msg)
        except Exception as e:
            self.sink.put(BaseFailed(task_id=msg.id))
            raise RuntimeError(f"Failed to process task `{msg.id}`: {e}")
