import cv2
from pydantic import BaseModel, Field

from server.util.types import ServerStatus


class Health(BaseModel):
    status: ServerStatus = Field(default=ServerStatus.DOWN)
    super_resolution: ServerStatus = Field(default=ServerStatus.DOWN)
    detection: ServerStatus = Field(default=ServerStatus.DOWN)

    class Config:
        use_enum_values = True
