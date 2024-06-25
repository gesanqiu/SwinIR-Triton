from pydantic import BaseModel


class UpScaleRequest(BaseModel):
    bitmap: str
    telephoto: int
    
class UpScaleResponse(BaseModel):
    bitmap: str
    upScale: int
    message: str

class ErrorResponse(BaseModel):
    message: str
    type: str
