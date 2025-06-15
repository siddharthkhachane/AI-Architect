from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG Knowledge Assistant"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "LangChain + RAG-powered AI knowledge assistant"
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://localhost:3000",
        "https://localhost:3001",
    ]
    
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".md", ".txt", ".py", ".js", ".tsx", ".jsx", ".java", ".cpp", ".go", ".rs"}
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_TIMEOUT: int = 120
    
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8000
    CHROMADB_PERSIST_DIRECTORY: str = "./chroma_db"
    COLLECTION_NAME: str = "knowledge_base"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    MAX_CONTEXT_LENGTH: int = 4000
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60
    
    ENABLE_METRICS: bool = True
    METRICS_RETENTION_DAYS: int = 30
    
    MAX_CONVERSATIONS_PER_USER: int = 50
    MAX_CONVERSATION_LENGTH: int = 100
    
    ENABLE_DOCUMENT_VALIDATION: bool = True
    MAX_DOCUMENTS_PER_USER: int = 100
    
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMADB_PERSIST_DIRECTORY, exist_ok=True)