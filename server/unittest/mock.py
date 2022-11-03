import base64
import glob
import os
from enum import Enum, auto
from pathlib import Path
from typing import List
from uuid import uuid4

import cv2
import numpy as np

from server.payload.SR.Request import SRRequest


class TaskType(Enum):
    SuperResolution = auto()
    DetectionVideo = auto()
    DetectionImage = auto()
    DetectionImages = auto()


def prepare_output_dir(tag: str = "test") -> str:
    output_dir = f"{os.getcwd()}/output/{tag}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_sr_requests() -> List[SRRequest]:
    valid_exts = ("png", "jpg", "gif")

    samples: List[str] = []
    for ext in valid_exts:
        samples.extend(glob.glob(f"server/unittest/sample/SR/*.{ext}"))

    samples = sorted(samples)

    requests: List[SRRequest] = []
    for sample in samples:
        with open(sample, "rb") as infile:
            base64_image = base64.b64encode(infile.read())
            requests.append(SRRequest(data=base64_image.decode("ascii")))

    return requests


def prepare_detection_videos() -> List[str]:
    valid_exts = ["mpg"]

    samples: List[str] = []
    for ext in valid_exts:
        samples.extend(glob.glob(f"server/unittest/sample/detection/*.{ext}"))

    return sorted(samples)


def prepare_detection_images() -> List[str]:
    valid_exts = ("png", "jpg", "gif")

    samples: List[str] = []
    for ext in valid_exts:
        samples.extend(glob.glob(f"server/unittest/sample/detection/*.{ext}"))

    return sorted(samples)


def prepare_detection_video_requests() -> List:
    valid_exts = ["mpg"]

    samples: List[str] = []
    for ext in valid_exts:
        samples.extend(glob.glob(f"server/unittest/sample/detection/*.{ext}"))

    samples = sorted(samples)

    requests = []
    for sample in samples:
        requests.append({"file": open(sample, "rb")})

    return requests


def prepare_detection_image_requests() -> List:
    valid_exts = ("png", "jpg", "gif")

    samples: List[str] = []
    for ext in valid_exts:
        samples.extend(glob.glob(f"server/unittest/sample/detection/*.{ext}"))

    samples = sorted(samples)

    requests = []
    for sample in samples:
        requests.append({"file": open(sample, "rb")})

    return requests


def prepare_detection_images_request():
    valid_exts = ("png", "jpg", "gif")

    samples: List[str] = []
    for ext in valid_exts:
        samples.extend(glob.glob(f"server/unittest/sample/detection/*.{ext}"))

    samples = sorted(samples)
    return [("file", open(f, "rb")) for f in samples]


def prepare_requests(type: TaskType) -> List:
    if type == TaskType.SuperResolution:
        return prepare_sr_requests()

    if type == TaskType.DetectionVideo:
        return prepare_detection_video_requests()

    if type == TaskType.DetectionImage:
        return prepare_detection_image_requests()

    if type == TaskType.DetectionImages:
        return prepare_detection_images_request()

    else:
        return []


def save_image(data: np.ndarray, output_dir: str, filename: str = None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = str(uuid4()) if filename is None else filename
    output_path = f"{output_dir}/{filename}.png"
    img_encoded = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_encoded)
