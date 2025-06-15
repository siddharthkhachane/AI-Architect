from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.logging import RequestLoggingMiddleware
from app.api.v1.endpoints import documents, chat
from app.api.v1.endpoints.admin import router as admin_router
from app.models.schemas import HealthCheck, ErrorResponse
from app.services.llm import OllamaService
from app.services.document_manager import DocumentManager
from app.services.metrics import metrics_service
from datetime import datetime
import logging
import time

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_time = time.time()

if settings.ENABLE_RATE_LIMITING:
    app.add_middleware(
        RateLimitMiddleware,
        calls=settings.RATE_LIMIT_CALLS,
        period=settings.RATE_LIMIT_PERIOD
    )

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time_req = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time_req
    
    if settings.ENABLE_METRICS:
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        
        await metrics_service.record_api_call(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time=process_time,
            client_ip=client_ip
        )
    
    return response

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

app.include_router(
    admin_router,
    prefix=f"{settings.API_V1_STR}/admin",
    tags=["admin"]
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
    
    uptime = time.time() - start_time
    
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
        services=services,
        uptime=uptime,
        system_info={
            "python_version": "3.11+",
            "environment": "development" if settings.DEBUG else "production"
        }
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
            "Conversation management with history",
            "Document summarization",
            "Real-time streaming responses",
            "Rate limiting and monitoring",
            "Comprehensive metrics and analytics",
            "Batch operations support"
        ],
        "endpoints": {
            "documents": f"{settings.API_V1_STR}/documents",
            "chat": f"{settings.API_V1_STR}/chat",
            "admin": f"{settings.API_V1_STR}/admin"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else "An unexpected error occurred",
            timestamp=datetime.utcnow(),
            error_code="INTERNAL_ERROR",
            suggestion="Please try again later or contact support if the issue persists."
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )