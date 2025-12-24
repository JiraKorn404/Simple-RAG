from pydantic import BaseModel
from typing import Optional

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    # sources: List[str] = []

class FileUploadResponse(BaseModel):
    status: str
    file_id: str
    filename: str
    uploaded_at: str
    message: str

class CheckFiles(BaseModel):
    filename: str | None
    file_id: str
    size_kb: float
    uploaded_at: str

class DeleteFileResponse(BaseModel):
    message: str
    file_id: str
    mongodb_deleted: bool
    qdrant_deleted: bool