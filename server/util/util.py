import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cv2
import torch
from attr import define


@define
class Region:
    frame: torch.Tensor
    x: int
    y: int
    h: int
    w: int
    path: str


def save_regions(regions: Tuple[Region], chunk_size=10):
    chunk_num = math.ceil(len(regions) / chunk_size)
    chunks = []
    for i in range(0, chunk_num):
        chunks.append(regions[i * chunk_size : (i + 1) * chunk_size])

    def _save(rs: Tuple[Region]):
        for r in rs:
            x, y, h, w = r.x, r.y, r.h, r.w
            cropped = r.frame[y : y + h, x : x + w]
            cv2.imwrite(r.path, cropped.cpu().numpy())

    with ThreadPoolExecutor() as pool:
        pool.map(_save, chunks)


def create_timestamp() -> int:
    return int(time.time_ns() / 1000000)


@define
class RectRegion:
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def xywh2xyxy(x, y, w, h, frame_width, frame_height):
    return RectRegion(
        x_min=max(int(x - w / 2), 0),
        x_max=min(int(x + w / 2), frame_width - 1),
        y_min=max(int(y - h / 2), 0),
        y_max=min(int(y + h / 2), frame_height - 1),
    )


def calculate_ewma(prev_avg: int, measured: int, ewma_alpha: float = 0.2):
    if prev_avg is None:
        return measured

    return int(float(measured) * ewma_alpha + float(prev_avg) * (1 - ewma_alpha))


def flatten(xss):
    return [x for xs in xss for x in xs]
