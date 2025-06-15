from sentence_transformers import SentenceTransformer
from app.core.config import settings
from typing import List, Dict, Any
import numpy as np
import logging
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None
        logger.info(f"EmbeddingService initialized with model: {self.model_name}")
    
    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model
    
    async def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            self._encode_text, 
            text.strip()
        )
        return embedding.tolist()
    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        if not documents:
            return []
        
        valid_docs = [doc.strip() for doc in documents if doc and doc.strip()]
        if not valid_docs:
            return [[0.0] * self.dimension] * len(documents)
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self._encode_batch,
            valid_docs
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def _encode_text(self, text: str) -> np.ndarray:
        try:
            return self.model.encode(text, convert_to_tensor=False)
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return np.zeros(self.dimension)
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        try:
            return self.model.encode(texts, convert_to_tensor=False, batch_size=32)
        except Exception as e:
            logger.error(f"Error encoding batch: {str(e)}")
            return np.zeros((len(texts), self.dimension))
    
    async def get_similarity(self, text1: str, text2: str) -> float:
        embeddings = await self.embed_documents([text1, text2])
        
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)