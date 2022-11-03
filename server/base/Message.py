"""Message module."""
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from server.util.util import create_timestamp

MessageType = UUID


class BaseMessageType(object):
    REGISTER_TASK = uuid4()
    RETURN_RESULT = uuid4()
    FAILED = uuid4()


class BaseMessage(BaseModel):
    type: MessageType = Field(default=None, exclude=True)
    created_at: int = Field(default_factory=create_timestamp, const=True, exclude=True)

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


class BaseTask(BaseMessage):
    type: MessageType = Field(
        default=BaseMessageType.REGISTER_TASK, const=True, exclude=True
    )
    id: UUID = Field(default_factory=uuid4, const=True)


class BaseResult(BaseMessage):
    type: MessageType = Field(
        default=BaseMessageType.RETURN_RESULT, const=True, exclude=True
    )
    task_id: UUID


class BaseFailed(BaseMessage):
    type: MessageType = Field(default=BaseMessageType.FAILED, const=True, exclude=True)
    task_id: UUID
