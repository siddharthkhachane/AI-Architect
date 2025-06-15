import httpx
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from app.core.config import settings
from app.models.schemas import SourceReference
import logging

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        self.max_context_length = settings.MAX_CONTEXT_LENGTH
    
    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return False
    
    async def ensure_model_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json()
                    available_models = [model["name"] for model in models.get("models", [])]
                    
                    if self.model in available_models:
                        return True
                    
                    logger.info(f"Model {self.model} not found. Attempting to pull...")
                    pull_response = await client.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model},
                        timeout=300
                    )
                    return pull_response.status_code == 200
                    
        except Exception as e:
            logger.error(f"Error ensuring model availability: {str(e)}")
            return False
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    raise Exception(f"LLM generation failed: {response.status_code}")
                    
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            raise Exception("LLM request timed out")
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def generate_streaming_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                                    if data.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        raise Exception(f"Streaming failed: {response.status_code}")
                        
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise

class RAGService:
    def __init__(self, vector_store, document_manager):
        self.llm = OllamaService()
        self.vector_store = vector_store
        self.document_manager = document_manager
        self.max_context_length = settings.MAX_CONTEXT_LENGTH
    
    async def generate_answer(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        document_ids: Optional[List[str]] = None,
        include_sources: bool = True,
        max_sources: int = 5
    ) -> Dict[str, Any]:
        start_time = asyncio.get_event_loop().time()
        
        try:
            relevant_chunks = await self.vector_store.search(
                query=query,
                n_results=max_sources,
                document_ids=document_ids
            )
            
            context = self._build_context(relevant_chunks)
            
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, context, conversation_history)
            
            response = await self.llm.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            sources = []
            if include_sources and relevant_chunks:
                sources = await self._build_source_references(relevant_chunks)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "response": response,
                "sources": sources,
                "processing_time": processing_time,
                "model_used": self.llm.model,
                "context_used": len(context) > 0
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG answer: {str(e)}")
            raise
    
    async def generate_streaming_answer(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        document_ids: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            relevant_chunks = await self.vector_store.search(
                query=query,
                n_results=settings.TOP_K_RESULTS,
                document_ids=document_ids
            )
            
            context = self._build_context(relevant_chunks)
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, context, conversation_history)
            
            sources = await self._build_source_references(relevant_chunks) if relevant_chunks else []
            
            yield {
                "type": "sources",
                "data": sources
            }
            
            async for chunk in self.llm.generate_streaming_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            ):
                yield {
                    "type": "content",
                    "data": chunk
                }
                
        except Exception as e:
            logger.error(f"Error in streaming RAG answer: {str(e)}")
            yield {
                "type": "error",
                "data": f"Error generating response: {str(e)}"
            }
    
    def _build_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        if not relevant_chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(relevant_chunks, 1):
            chunk_text = chunk["chunk_text"]
            doc_title = chunk["metadata"].get("filename", "Unknown Document")
            
            context_part = f"Source {i} (from {doc_title}):\n{chunk_text}\n"
            
            if current_length + len(context_part) > self.max_context_length:
                break
            
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _build_system_prompt(self) -> str:
        return """You are an AI knowledge assistant. Your role is to answer questions based on the provided context from documents.

Guidelines:
1. Answer questions directly and accurately based on the provided context
2. If the context doesn't contain enough information, clearly state what you can answer and what limitations exist
3. Use specific information from the sources when possible
4. Be concise but comprehensive
5. If asked about something not in the context, politely explain that you need the relevant documents to answer
6. Maintain a helpful and professional tone
7. When referencing information, you can mention which source it comes from (e.g., "According to Source 1...")

Remember: You can only answer based on the provided context. Do not make up information."""
    
    def _build_user_prompt(
        self, 
        query: str, 
        context: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for msg in conversation_history[-3:]:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")
        
        if context:
            prompt_parts.append("Context from documents:")
            prompt_parts.append(context)
            prompt_parts.append("")
        
        prompt_parts.append(f"Question: {query}")
        
        if context:
            prompt_parts.append("\nPlease answer the question based on the provided context.")
        else:
            prompt_parts.append("\nNo relevant context was found in the documents. Please let the user know that you need relevant documents to answer their question.")
        
        return "\n".join(prompt_parts)
    
    async def _build_source_references(self, relevant_chunks: List[Dict[str, Any]]) -> List[SourceReference]:
        sources = []
        
        for chunk in relevant_chunks:
            metadata = chunk["metadata"]
            
            source = SourceReference(
                document_id=chunk["document_id"],
                document_title=metadata.get("filename", "Unknown Document"),
                chunk_text=chunk["chunk_text"][:500] + "..." if len(chunk["chunk_text"]) > 500 else chunk["chunk_text"],
                similarity_score=round(chunk["similarity_score"], 3),
                metadata={
                    "chunk_index": metadata.get("chunk_index"),
                    "document_type": metadata.get("document_type"),
                    "file_size": metadata.get("file_size")
                }
            )
            sources.append(source)
        
        return sources
    
    async def summarize_document(self, document_id: str) -> str:
        try:
            chunks = await self.vector_store.search(
                query="summary overview main points key information",
                n_results=10,
                document_ids=[document_id]
            )
            
            if not chunks:
                return "No content found for summarization."
            
            context = self._build_context(chunks)
            
            system_prompt = "You are an AI assistant that creates concise, informative summaries of documents."
            
            user_prompt = f"""Please provide a comprehensive summary of the following document content:

{context}

Create a summary that includes:
1. Main topics and themes
2. Key points and important information
3. Overall structure and organization

Keep the summary concise but comprehensive."""
            
            summary = await self.llm.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing document {document_id}: {str(e)}")
            return f"Error generating summary: {str(e)}"