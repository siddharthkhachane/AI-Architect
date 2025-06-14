from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.models.schemas import DocumentResponse, UploadResponse
from app.core.deps import validate_file_upload, get_settings
from typing import List
import uuid
from datetime import datetime

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    validated_file: UploadFile = Depends(validate_file_upload),
    settings = Depends(get_settings)
):
    document_id = str(uuid.uuid4())
    
    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        file_size=file.size or 0,
        status="uploading",
        message="File uploaded successfully. Processing will begin shortly."
    )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents():
    return []

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    raise HTTPException(status_code=404, detail="Document not found")

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    return {"message": "Document deleted successfully"}