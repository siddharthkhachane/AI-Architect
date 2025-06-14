from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.core.deps import get_settings
import uuid
from datetime import datetime

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings = Depends(get_settings)
):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    return ChatResponse(
        response="Hello! I'm your AI knowledge assistant. I'll be able to answer questions based on your uploaded documents once the RAG pipeline is implemented.",
        conversation_id=conversation_id,
        sources=[],
        processing_time=0.1,
        model_used=settings.OLLAMA_MODEL
    )

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    return {"conversation_id": conversation_id, "messages": []}