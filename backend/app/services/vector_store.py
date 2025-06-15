import chromadb
from chromadb.config import Settings
from app.core.config import settings
from app.services.embeddings import EmbeddingService
from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.collection_name = settings.COLLECTION_NAME
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        if self._client is None:
            logger.info("Initializing ChromaDB client")
            self._client = chromadb.PersistentClient(
                path=settings.CHROMADB_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB client initialized")
        return self._client
    
    @property
    def collection(self):
        if self._collection is None:
            logger.info(f"Getting or creating collection: {self.collection_name}")
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection ready")
        return self._collection
    
    async def add_document(self, document_id: str, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        try:
            if not chunks:
                logger.warning(f"No chunks to add for document {document_id}")
                return
            
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.embedding_service.embed_documents(chunk_texts)
            
            ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_length": chunk.get("length", len(chunk["text"])),
                    **metadata
                }
                
                ids.append(chunk_id)
                metadatas.append(chunk_metadata)
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error adding document {document_id} to vector store: {str(e)}")
            raise
    
    async def search(self, query: str, n_results: int = None, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            if n_results is None:
                n_results = settings.TOP_K_RESULTS
            
            query_embedding = await self.embedding_service.embed_text(query)
            
            where_filter = None
            if document_ids:
                where_filter = {"document_id": {"$in": document_ids}}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    similarity_score = 1 - distance
                    
                    if similarity_score >= settings.SIMILARITY_THRESHOLD:
                        search_results.append({
                            "document_id": metadata["document_id"],
                            "chunk_text": doc,
                            "similarity_score": similarity_score,
                            "metadata": metadata
                        })
            
            logger.info(f"Found {len(search_results)} relevant chunks for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str):
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            else:
                logger.info(f"No chunks found for document {document_id}")
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise
    
    async def get_document_count(self) -> int:
        try:
            result = self.collection.count()
            return result
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    async def list_documents(self) -> List[str]:
        try:
            results = self.collection.get(include=["metadatas"])
            document_ids = set()
            
            for metadata in results["metadatas"]:
                if "document_id" in metadata:
                    document_ids.add(metadata["document_id"])
            
            return list(document_ids)
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []