import tempfile
from typing import List

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from loguru import logger

from server.application import Application
from server.base.Engine import BaseEngine
from server.base.Message import BaseMessage, BaseMessageType, BaseResult
from server.engine.Container import EngineContainer
from server.engine.detection.Message import (
    DetectionInputType,
    DetectionResult,
    DetectionTask,
    FrameData,
)
from server.engine.SR.Message import SRTask
from server.payload.actuator.Response import Health
from server.payload.detection.Response import DetectionResponse
from server.payload.SR.Request import SRRequest
from server.payload.SR.Response import SRResponse
from server.util.types import ServerStatus

actuator = APIRouter(prefix="/actuator")


@actuator.get("/health")
@inject
async def get_actuator_health(
    engines: EngineContainer = Depends(Provide[Application.engine_container]),
):
    sr_status = ServerStatus.UP if engines.sr_engine().isReady() else ServerStatus.DOWN
    det_status = (
        ServerStatus.UP if engines.detection_engine().isReady() else ServerStatus.DOWN
    )
    return Health(
        status=ServerStatus.UP, super_resolution=sr_status, detection=det_status
    )


api = APIRouter(prefix="/api")


@api.post(
    "/superresolution",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
@inject
async def submit_sr(
    request: SRRequest,
    engine: BaseEngine = Depends(Provide[Application.engine_container().sr_engine]),
):
    if not engine.isReady():
        raise HTTPException(status_code=503, detail="Engine not ready")
    task = SRTask(image=request.data)
    result: BaseMessage = engine.process(task)
    if result.type == BaseMessageType.FAILED:
        raise HTTPException(status_code=500, detail=f"Failed to process task")

    return SRResponse(result=result)


@api.post("/detection/image")
@inject
async def submit_detection_image(
    file: UploadFile,
    engine: BaseEngine = Depends(
        Provide[Application.engine_container().detection_engine]
    ),
):
    if not engine.isReady():
        raise HTTPException(status_code=503, detail="Engine not ready")

    image = await file.read()
    task = DetectionTask(input_type=DetectionInputType.IMAGE, image_data=[image])

    result: DetectionResult = engine.process(task)
    if result.type == BaseMessageType.FAILED or len(result.data) <= 0:
        raise HTTPException(status_code=500, detail=f"Failed to process task")

    data: FrameData = result.data[0]
    return DetectionResponse(patches=data.patches)


@api.post("/detection/images")
@inject
async def submit_detection_images(
    file: List[UploadFile],
    engine: BaseEngine = Depends(
        Provide[Application.engine_container().detection_engine]
    ),
):
    if not engine.isReady():
        raise HTTPException(status_code=503, detail="Engine not ready")

    images = [await f.read() for f in file]
    task = DetectionTask(input_type=DetectionInputType.IMAGE, image_data=images)

    result: DetectionResult = engine.process(task)
    if result.type == BaseMessageType.FAILED or len(result.data) <= 0:
        raise HTTPException(status_code=500, detail=f"Failed to process task")

    return [
        DetectionResponse(frame_index=d.frame_index, patches=d.patches)
        for d in result.data
    ]


@api.post("/detection/video")
@inject
async def submit_detection_video(
    file: UploadFile,
    engine: BaseEngine = Depends(
        Provide[Application.engine_container().detection_engine]
    ),
) -> List[DetectionResponse]:
    if not engine.isReady():
        raise HTTPException(status_code=503, detail="Engine not ready")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/{file.filename}"
        with open(video_path, mode="wb") as f:
            f.write(await file.read())

        task = DetectionTask(input_type=DetectionInputType.VIDEO, video_path=video_path)
        result: DetectionResult = engine.process(task)
        if result.type == BaseMessageType.FAILED or len(result.data) <= 0:
            raise HTTPException(status_code=500, detail=f"Failed to process task")

    response: List[DetectionResponse] = []
    for ret in result.data:
        response.append(
            DetectionResponse(frame_index=ret.frame_index, patches=ret.patches)
        )
    return response


router = APIRouter()
router.include_router(actuator)
router.include_router(api)
