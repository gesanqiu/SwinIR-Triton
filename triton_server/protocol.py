from pydantic import BaseModel, Field
import uuid


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class UpScaleRequest(BaseModel):
    bitmap: str
    telephoto: int
    request_id: str = Field(default_factory=lambda: f"request-{random_uuid()}")


class UpScaleResponse(BaseModel):
    bitmap: str
    upScale: int
    message: str
    request_id: str
    receive_time: int
    response_time: int


class ErrorResponse(BaseModel):
    message: str
    type: str
