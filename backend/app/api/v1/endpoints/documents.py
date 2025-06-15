from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from app.models.schemas import (
    DocumentResponse, UploadResponse, DocumentStats, SearchRequest, SearchResponse,
    BatchUploadResponse, RepositoryRequest, BulkOperationResponse
)
from app.core.deps import validate_file_upload, get_settings, save_upload_file
from app.services.document_manager import DocumentManager
from app.services.metrics import metrics_service
from typing import List, Optional
import uuid
import os
import asyncio
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()

document_manager = DocumentManager()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    validated_file: UploadFile = Depends(validate_file_upload),
    settings = Depends(get_settings)
):
    start_time = datetime.utcnow()
    document_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}_{file.filename}")
    
    try:
        await save_upload_file(file, file_path)
        file_size = os.path.getsize(file_path)
        
        estimated_time = file_size / (1024 * 1024) * 2
        
        background_tasks.add_task(
            process_document_with_metrics,
            file_path,
            file.filename,
            document_id,
            file_size
        )
        
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            status="uploading",
            message="File uploaded successfully. Processing started.",
            estimated_processing_time=estimated_time
        )
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        
        await metrics_service.record_document_upload(
            file_size=0,
            processing_time=0,
            success=False
        )
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    settings = Depends(get_settings)
):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    successful_uploads = []
    failed_uploads = []
    
    for file in files:
        try:
            if file.filename:
                validated_file = await validate_file_upload(file)
                document_id = str(uuid.uuid4())
                file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}_{file.filename}")
                
                await save_upload_file(file, file_path)
                file_size = os.path.getsize(file_path)
                
                background_tasks.add_task(
                    process_document_with_metrics,
                    file_path,
                    file.filename,
                    document_id,
                    file_size
                )
                
                successful_uploads.append(UploadResponse(
                    document_id=document_id,
                    filename=file.filename,
                    file_size=file_size,
                    status="uploading",
                    message="File uploaded successfully"
                ))
                
        except Exception as e:
            failed_uploads.append({
                "filename": file.filename or "unknown",
                "error": str(e)
            })
    
    return BatchUploadResponse(
        successful_uploads=successful_uploads,
        failed_uploads=failed_uploads,
        total_files=len(files),
        success_count=len(successful_uploads),
        failure_count=len(failed_uploads)
    )

@router.post("/repository", response_model=UploadResponse)
async def process_repository(
    background_tasks: BackgroundTasks,
    request: RepositoryRequest
):
    document_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        process_repository_with_metrics,
        request.repo_url,
        document_id,
        request
    )
    
    return UploadResponse(
        document_id=document_id,
        filename=f"Repository: {request.repo_url}",
        file_size=0,
        status="uploading",
        message="Repository processing started."
    )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    document_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    documents = await document_manager.list_documents()
    
    if status:
        documents = [doc for doc in documents if doc.status == status]
    
    if document_type:
        documents = [doc for doc in documents if doc.document_type == document_type]
    
    return documents[offset:offset + limit]

@router.get("/stats", response_model=DocumentStats)
async def get_document_stats():
    documents = await document_manager.list_documents()
    
    total_size = sum(doc.file_size or 0 for doc in documents)
    total_chunks = sum(doc.chunk_count or 0 for doc in documents)
    
    by_type = {}
    by_status = {}
    
    for doc in documents:
        by_type[doc.document_type] = by_type.get(doc.document_type, 0) + 1
        by_status[doc.status] = by_status.get(doc.status, 0) + 1
    
    return DocumentStats(
        total_documents=len(documents),
        by_type=by_type,
        by_status=by_status,
        total_chunks=total_chunks,
        total_size_mb=round(total_size / (1024 * 1024), 2)
    )

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    start_time = datetime.utcnow()
    
    results = await document_manager.search_documents(
        query=request.query,
        document_ids=request.document_ids
    )
    
    if request.similarity_threshold:
        results = [r for r in results if r["similarity_score"] >= request.similarity_threshold]
    
    results = results[:request.limit]
    
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        processing_time=processing_time
    )

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

@router.delete("/", response_model=BulkOperationResponse)
async def bulk_delete_documents(document_ids: List[str]):
    start_time = datetime.utcnow()
    successful_deletions = 0
    errors = []
    
    for doc_id in document_ids:
        try:
            success = await document_manager.delete_document(doc_id)
            if success:
                successful_deletions += 1
            else:
                errors.append(f"Document {doc_id} not found")
        except Exception as e:
            errors.append(f"Error deleting {doc_id}: {str(e)}")
    
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    return BulkOperationResponse(
        operation="bulk_delete",
        total_items=len(document_ids),
        successful_items=successful_deletions,
        failed_items=len(document_ids) - successful_deletions,
        errors=errors,
        processing_time=processing_time
    )

async def process_document_with_metrics(file_path: str, filename: str, document_id: str, file_size: int):
    start_time = datetime.utcnow()
    success = False
    
    try:
        await document_manager.process_uploaded_file(file_path, filename)
        success = True
    except Exception as e:
        print(f"Background processing failed for {document_id}: {str(e)}")
    finally:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await metrics_service.record_document_upload(
            file_size=file_size,
            processing_time=processing_time,
            success=success
        )

async def process_repository_with_metrics(repo_url: str, document_id: str, request: RepositoryRequest):
    start_time = datetime.utcnow()
    success = False
    
    try:
        await document_manager.process_repository(repo_url)
        success = True
    except Exception as e:
        print(f"Background repository processing failed for {document_id}: {str(e)}")
    finally:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await metrics_service.record_document_upload(
            file_size=0,
            processing_time=processing_time,
            success=success
        )