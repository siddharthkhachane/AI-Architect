from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.api.v1.endpoints import documents, chat
from app.models.schemas import HealthCheck, ErrorResponse
from app.services.llm import OllamaService
from app.services.document_manager import DocumentManager
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(
    documents.router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["documents"]
)

app.include_router(
    chat.router,
    prefix=f"{settings.API_V1_STR}/chat",
    tags=["chat"]
)

@app.get("/health", response_model=HealthCheck)
async def health_check():
    ollama_service = OllamaService()
    document_manager = DocumentManager()
    
    ollama_status = "connected" if await ollama_service.check_health() else "disconnected"
    
    try:
        doc_count = await document_manager.vector_store.get_document_count()
        chromadb_status = "connected"
    except:
        doc_count = 0
        chromadb_status = "disconnected"
    
    services = {
        "api": "running",
        "ollama": ollama_status,
        "chromadb": chromadb_status,
        "document_count": str(doc_count)
    }
    
    overall_status = "healthy" if all(
        status in ["running", "connected"] for status in [services["api"], services["ollama"], services["chromadb"]]
    ) else "degraded"
    
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.VERSION,
        services=services
    )

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Document upload and processing (PDF, Markdown, Code)",
            "Repository processing via Git clone",
            "RAG-powered Q&A with source citations",
            "Conversation management",
            "Document summarization",
            "Real-time streaming responses"
        ]
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else "An unexpected error occurred",
            timestamp=datetime.utcnow()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )