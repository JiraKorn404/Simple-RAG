from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    # sources: List[str] = []

class IngestResponse(BaseModel):
    status: str
    message: str

class FileUploadResponse(BaseModel):
    file_id: str
    uploaded_at: str
    # authentication later (userID)