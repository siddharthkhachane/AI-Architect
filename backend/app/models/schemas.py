from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"
    TEXT = "text"
    REPOSITORY = "repository"

class DocumentStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
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
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

class DocumentStats(BaseModel):
    total_documents: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    total_chunks: int
    total_size_mb: float

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    conversation_id: Optional[str] = None
    include_sources: bool = True
    max_sources: int = Field(default=3, ge=1, le=10)
    document_ids: Optional[List[str]] = Field(default=None, description="Specific documents to search in")
    temperature: Optional[float] = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4000)

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
    tokens_used: Optional[int] = None
    context_length: Optional[int] = None

class ConversationSummary(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: Optional[str] = None
    title: Optional[str] = None

class ConversationDetail(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessage]
    metadata: Optional[Dict[str, Any]] = {}

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size: int
    status: DocumentStatus
    message: str
    estimated_processing_time: Optional[float] = None

class BatchUploadResponse(BaseModel):
    successful_uploads: List[UploadResponse]
    failed_uploads: List[Dict[str, str]]
    total_files: int
    success_count: int
    failure_count: int

class RepositoryRequest(BaseModel):
    repo_url: str = Field(..., description="Git repository URL")
    branch: Optional[str] = Field(default="main", description="Branch to clone")
    include_patterns: Optional[List[str]] = Field(default=None, description="File patterns to include")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="File patterns to exclude")
    max_files: Optional[int] = Field(default=1000, ge=1, le=5000)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    document_ids: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class SearchResponse(BaseModel):
    query: str
    results: List[SourceReference]
    total_results: int
    processing_time: float

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]
    uptime: Optional[float] = None
    system_info: Optional[Dict[str, Any]] = {}

class SystemMetrics(BaseModel):
    total_documents: int
    total_conversations: int
    total_api_calls: int
    average_response_time: float
    error_rate: float
    storage_used_mb: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    error_code: Optional[str] = None
    suggestion: Optional[str] = None

class ValidationError(BaseModel):
    field: str
    message: str
    value: Any

class BulkOperationResponse(BaseModel):
    operation: str
    total_items: int
    successful_items: int
    failed_items: int
    errors: List[str] = []
    processing_time: float