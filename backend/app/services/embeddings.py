from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        logger.info(f"EmbeddingService initialized with model: {self.model_name}")
    
    async def embed_text(self, text: str):
        pass
    
    async def embed_documents(self, documents: list[str]):
        pass