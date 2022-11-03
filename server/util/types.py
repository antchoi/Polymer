import platform
from enum import Enum

import torch
from attrs import define
from loguru import logger

from server.util.cpuinfo import get_cpu_info


class ServerStatus(Enum):
    UP = "UP"
    DOWN = "DOWN"


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "cuda"


@define
class EngineDevice:
    device_str: str = DeviceType.CPU.name.lower()
    mode: DeviceType = DeviceType.CPU
    id: int = 0
    device_model: str = None

    def __attrs_post_init__(self):
        dev = str(self.device_str).strip().upper()
        if dev == DeviceType.CPU.name:
            self.mode = DeviceType.CPU
        else:
            id = 0
            try:
                id = int(dev)
                self.mode = DeviceType.GPU
                self.id = id
            except:
                logger.warning(f"Invalid device id: {dev}")

        if self.mode == DeviceType.GPU:
            self.device_model = torch.cuda.get_device_name(self.torch())
        else:
            self.device_model = platform.processor()

            cpuinfo = get_cpu_info()
            self.device_model = (
                cpuinfo["brand_raw"] if "brand_raw" in cpuinfo else "CPU"
            )

    def torch(self):
        if self.mode is DeviceType.CPU:
            return torch.device(device=self.mode.value)
        return torch.device(type=self.mode.value, index=self.id)

    def __str__(self) -> str:
        if self.mode is DeviceType.CPU:
            return DeviceType.CPU.name.lower()
        else:
            return f"cuda:{self.id}"


@define
class XYHW:
    x: int
    y: int
    h: int
    w: int


@define
class XYXY:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@define
class FrameShape:
    width: int
    height: int


@define(kw_only=True)
class PatchRegion:
    shape: FrameShape
    xyhw: XYHW = None
    xyxy: XYXY = None
    patch_id: int = None
    area: int = None

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        xyhw: XYHW = None,
        xyxy: XYXY = None,
        patch_id: int = None,
    ):
        self.shape = FrameShape(width=frame_width, height=frame_height)
        self.xyhw = xyhw
        self.xyxy = xyxy
        self.patch_id = patch_id

        if self.xyhw is not None and self.xyxy is None:
            self.xyxy = XYXY(
                x_min=max(int(self.xyhw.x - self.xyhw.w / 2), 0),
                x_max=min(int(self.xyhw.x + self.xyhw.w / 2), self.shape.width - 1),
                y_min=max(int(self.xyhw.y - self.xyhw.h / 2), 0),
                y_max=min(int(self.xyhw.y + self.xyhw.h / 2), self.shape.height - 1),
            )
        elif self.xyhw is None and self.xyxy is not None:
            self.xyhw = XYHW(
                x=int((self.xyxy.x_min + self.xyxy.x_max) / 2),
                y=int((self.xyxy.y_min + self.xyxy.y_max) / 2),
                h=int(self.xyxy.y_max - self.xyxy.y_min),
                w=int(self.xyxy.x_max - self.xyxy.x_min),
            )
        else:
            raise RuntimeError("Invalid patch region object")

        self.area = int(self.xyhw.h * self.xyhw.w)


class ImageFormat(Enum):
    JPG = "jpg"
    PNG = "png"
    GIF = "gif"


class ImageExt(Enum):
    JPG = ".jpg"
    PNG = ".png"
    GIF = ".gif"
