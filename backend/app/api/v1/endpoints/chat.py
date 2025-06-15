from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    ChatRequest, ChatResponse, ConversationSummary, ConversationDetail
)
from app.core.deps import get_settings
from app.services.document_manager import DocumentManager
from app.services.llm import RAGService
from app.services.conversation_manager import ConversationManager
from app.services.metrics import metrics_service
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

router = APIRouter()

class DocumentSummaryRequest(BaseModel):
    document_id: str
    summary_type: str = "comprehensive"

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
    start_time = datetime.utcnow()
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
            document_ids=request.document_ids,
            include_sources=request.include_sources,
            max_sources=request.max_sources
        )
        
        background_tasks.add_task(
            save_conversation_with_metrics,
            conversation_id,
            request.message,
            result["response"],
            len(request.message),
            len(result["response"]),
            result["processing_time"],
            len(result["sources"])
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ChatResponse(
            response=result["response"],
            conversation_id=conversation_id,
            sources=result["sources"],
            processing_time=processing_time,
            model_used=result["model_used"],
            context_length=len(str(result.get("context", "")))
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
                sources_count = 0
                
                async for chunk in rag_service.generate_streaming_answer(
                    query=request.message,
                    conversation_history=conversation_history,
                    document_ids=request.document_ids
                ):
                    if chunk["type"] == "content":
                        collected_response += chunk["data"]
                    elif chunk["type"] == "sources":
                        sources_count = len(chunk["data"])
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                await save_conversation_with_metrics(
                    conversation_id,
                    request.message,
                    collected_response,
                    len(request.message),
                    len(collected_response),
                    0.0,
                    sources_count
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

@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = 50,
    offset: int = 0
):
    conversations = await conversation_manager.list_conversations()
    
    summaries = []
    for conv in conversations[offset:offset + limit]:
        conversation_detail = await conversation_manager.get_conversation(conv["id"])
        
        last_message = None
        title = None
        
        if conversation_detail and conversation_detail["messages"]:
            last_user_message = next(
                (msg for msg in reversed(conversation_detail["messages"]) if msg["role"] == "user"),
                None
            )
            if last_user_message:
                last_message = last_user_message["content"][:100] + "..." if len(last_user_message["content"]) > 100 else last_user_message["content"]
                title = last_user_message["content"][:50] + "..." if len(last_user_message["content"]) > 50 else last_user_message["content"]
        
        summaries.append(ConversationSummary(
            id=conv["id"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
            message_count=conv["message_count"],
            last_message_preview=last_message,
            title=title
        ))
    
    return summaries

@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    conversation = await conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationDetail(**conversation)

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
            "summary_type": request.summary_type,
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

async def save_conversation_with_metrics(
    conversation_id: str, 
    user_message: str, 
    assistant_response: str, 
    query_length: int,
    response_length: int,
    processing_time: float,
    sources_found: int
):
    try:
        await conversation_manager.add_message(conversation_id, "user", user_message)
        await conversation_manager.add_message(conversation_id, "assistant", assistant_response)
        
        await metrics_service.record_chat_interaction(
            query_length=query_length,
            response_length=response_length,
            processing_time=processing_time,
            sources_found=sources_found
        )
    except Exception as e:
        print(f"Error saving conversation: {str(e)}")