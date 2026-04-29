from pydantic import BaseModel


class PlayerMessage(BaseModel):
    content: str


class AudioUploadResponse(BaseModel):
    file_path: str
    transcript: str
