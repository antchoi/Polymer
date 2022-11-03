from pydantic import BaseModel


class SRRequest(BaseModel):
    data: str  # Base64 encoded
