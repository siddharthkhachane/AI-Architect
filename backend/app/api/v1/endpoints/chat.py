from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.core.deps import get_settings
from app.services.document_manager import DocumentManager
from app.services.llm import RAGService
from app.services.conversation_manager import ConversationManager
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

router = APIRouter()

class ConversationResponse(BaseModel):
    id: str
    created_at: str
    updated_at: str
    message_count: int

class DocumentSummaryRequest(BaseModel):
    document_id: str

document_manager = DocumentManager()
conversation_manager = ConversationManager()

rag_service = RAGService(
    vector_store=document_manager.vector_store,
    document_manager=document_manager
)

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    settings = Depends(get_settings)
):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        if not await rag_service.llm.check_health():
            raise HTTPException(
                status_code=503, 
                detail="LLM service is not available. Please ensure Ollama is running."
            )
        
        conversation_history = await conversation_manager.get_conversation_history(
            conversation_id, limit=6
        )
        
        result = await rag_service.generate_answer(
            query=request.message,
            conversation_history=conversation_history,
            include_sources=request.include_sources,
            max_sources=request.max_sources
        )
        
        background_tasks.add_task(
            save_conversation_messages,
            conversation_id,
            request.message,
            result["response"]
        )
        
        return ChatResponse(
            response=result["response"],
            conversation_id=conversation_id,
            sources=result["sources"],
            processing_time=result["processing_time"],
            model_used=result["model_used"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        if not await rag_service.llm.check_health():
            raise HTTPException(
                status_code=503,
                detail="LLM service is not available. Please ensure Ollama is running."
            )
        
        conversation_history = await conversation_manager.get_conversation_history(
            conversation_id, limit=6
        )
        
        async def generate_stream():
            try:
                collected_response = ""
                
                async for chunk in rag_service.generate_streaming_answer(
                    query=request.message,
                    conversation_history=conversation_history
                ):
                    if chunk["type"] == "content":
                        collected_response += chunk["data"]
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                await save_conversation_messages(
                    conversation_id,
                    request.message,
                    collected_response
                )
                
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
                
            except Exception as e:
                error_chunk = {
                    "type": "error",
                    "data": f"Streaming error: {str(e)}"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming chat failed: {str(e)}")

@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations():
    conversations = await conversation_manager.list_conversations()
    return [ConversationResponse(**conv) for conv in conversations]

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = await conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    success = await conversation_manager.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted successfully"}

@router.post("/summarize")
async def summarize_document(request: DocumentSummaryRequest):
    try:
        if not await rag_service.llm.check_health():
            raise HTTPException(
                status_code=503,
                detail="LLM service is not available. Please ensure Ollama is running."
            )
        
        document = await document_manager.get_document(request.document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        summary = await rag_service.summarize_document(request.document_id)
        
        return {
            "document_id": request.document_id,
            "document_title": document.title,
            "summary": summary,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@router.get("/health")
async def chat_health():
    ollama_status = await rag_service.llm.check_health()
    
    return {
        "status": "healthy" if ollama_status else "degraded",
        "ollama": "connected" if ollama_status else "disconnected",
        "model": rag_service.llm.model,
        "timestamp": datetime.utcnow().isoformat()
    }

async def save_conversation_messages(conversation_id: str, user_message: str, assistant_response: str):
    try:
        await conversation_manager.add_message(conversation_id, "user", user_message)
        await conversation_manager.add_message(conversation_id, "assistant", assistant_response)
    except Exception as e:
        print(f"Error saving conversation: {str(e)}")