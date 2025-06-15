from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from app.models.schemas import DocumentResponse, UploadResponse, DocumentStatus
from app.core.deps import validate_file_upload, get_settings, save_upload_file
from app.services.document_manager import DocumentManager
from typing import List
import uuid
import os
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()

class RepositoryRequest(BaseModel):
    repo_url: str

document_manager = DocumentManager()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    validated_file: UploadFile = Depends(validate_file_upload),
    settings = Depends(get_settings)
):
    document_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}_{file.filename}")
    
    try:
        await save_upload_file(file, file_path)
        
        background_tasks.add_task(
            process_document_background,
            file_path,
            file.filename,
            document_id
        )
        
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=os.path.getsize(file_path),
            status=DocumentStatus.UPLOADING,
            message="File uploaded successfully. Processing started."
        )
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/repository", response_model=UploadResponse)
async def process_repository(
    background_tasks: BackgroundTasks,
    request: RepositoryRequest
):
    document_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        process_repository_background,
        request.repo_url,
        document_id
    )
    
    return UploadResponse(
        document_id=document_id,
        filename=f"Repository: {request.repo_url}",
        file_size=0,
        status=DocumentStatus.UPLOADING,
        message="Repository processing started."
    )

async def process_document_background(file_path: str, filename: str, document_id: str):
    try:
        await document_manager.process_uploaded_file(file_path, filename)
    except Exception as e:
        print(f"Background processing failed for {document_id}: {str(e)}")

async def process_repository_background(repo_url: str, document_id: str):
    try:
        await document_manager.process_repository(repo_url)
    except Exception as e:
        print(f"Background repository processing failed for {document_id}: {str(e)}")

@router.get("/", response_model=List[DocumentResponse])
async def list_documents():
    return await document_manager.list_documents()

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    document = await document_manager.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    success = await document_manager.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}

@router.get("/{document_id}/search")
async def search_in_document(document_id: str, query: str):
    results = await document_manager.search_documents(query, document_ids=[document_id])
    return {"query": query, "results": results}