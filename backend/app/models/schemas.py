from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"
    TEXT = "text"

class DocumentStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"

class DocumentBase(BaseModel):
    title: str
    content: Optional[str] = None
    document_type: DocumentType
    metadata: Optional[Dict[str, Any]] = {}

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = 0
    file_path: Optional[str] = None
    file_size: Optional[int] = None

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question")
    conversation_id: Optional[str] = None
    include_sources: bool = True
    max_sources: int = Field(default=3, ge=1, le=10)

class SourceReference(BaseModel):
    document_id: str
    document_title: str
    chunk_text: str
    similarity_score: float
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[SourceReference] = []
    processing_time: float
    model_used: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size: int
    status: DocumentStatus
    message: str

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime