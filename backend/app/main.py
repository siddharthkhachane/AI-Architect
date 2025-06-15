import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
import chromadb
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="AI-Architect", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DB_PATH = "./chromadb"
USER = "siddharthkhachane"

class ChatRequest(BaseModel):
    message: str
    include_sources: bool = True

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[str] = []
    processing_time: float = 0.0
    model_used: str = "llama3.1:8b"
    context_length: int = 0

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class RepositoryRequest(BaseModel):
    repo_url: str
    branch: str = "main"

def get_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return client.get_collection("documents")
    except:
        return None

def create_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection("documents")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def find_text_files(directory: Path) -> List[Path]:
    text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yml', '.yaml', '.xml', '.sql', '.sh', '.bat', '.ps1', '.cs', '.java', '.cpp', '.c', '.h', '.php', '.rb', '.go', '.rs', '.kt', '.swift', '.ts', '.jsx', '.tsx', '.vue', '.svelte'}
    text_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in text_extensions:
            try:
                if file_path.stat().st_size < 10 * 1024 * 1024:
                    text_files.append(file_path)
            except:
                continue
    return text_files

def read_file(file_path: Path) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except:
            return ""

def clone_repository(repo_url: str, branch: str = "main") -> Path:
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    clone_path = Path(f"./temp_repos/{repo_name}")
    
    if clone_path.exists():
        shutil.rmtree(clone_path)
    
    clone_path.parent.mkdir(exist_ok=True)
    
    subprocess.run(['git', 'clone', '-b', branch, repo_url, str(clone_path)], 
                  check=True, capture_output=True)
    return clone_path

@app.get("/health")
async def health_check():
    return {"status": "healthy", "user": USER, "gpu_enabled": True}

@app.get("/api/v1/documents/collections")
async def get_collections():
    try:
        collection = get_collection()
        count = collection.count() if collection else 0
        return {
            "collections": [{
                "name": "documents",
                "count": count,
                "user": USER
            }]
        }
    except:
        return {"collections": []}

@app.post("/api/v1/documents/search")
async def search_documents(request: SearchRequest):
    try:
        collection = get_collection()
        if not collection or collection.count() == 0:
            return {"results": [], "total_results": 0}
        
        results = collection.query(
            query_texts=[request.query],
            n_results=request.limit
        )
        
        formatted_results = []
        if results and 'documents' in results and results['documents'][0]:
            for doc in results['documents'][0]:
                formatted_results.append({"content": doc})
        
        return {
            "query": request.query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
    except Exception as e:
        return {"results": [], "total_results": 0, "error": str(e)}

@app.post("/api/v1/documents/upload")
async def upload_repository(request: RepositoryRequest):
    try:
        repo_path = clone_repository(request.repo_url, request.branch)
        
        documents = []
        for file_path in find_text_files(repo_path):
            content = read_file(file_path)
            if content.strip():
                chunks = chunk_text(content)
                for chunk in chunks:
                    documents.append({
                        "id": str(uuid.uuid4()),
                        "content": chunk,
                        "metadata": {
                            "file": str(file_path.relative_to(repo_path)),
                            "repo": request.repo_url,
                            "user": USER
                        }
                    })
        
        if documents:
            collection = create_collection()
            collection.add(
                documents=[doc["content"] for doc in documents],
                ids=[doc["id"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents]
            )
        
        shutil.rmtree(repo_path)
        
        return {
            "message": "Repository processed successfully",
            "documents_added": len(documents),
            "repo_url": request.repo_url,
            "user": USER
        }
    except Exception as e:
        return {"error": str(e), "repo_url": request.repo_url}

@app.post("/api/v1/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    import time
    start_time = time.time()
    
    try:
        collection = get_collection()
        search_results = []
        context_length = 0
        
        if collection and collection.count() > 0:
            results = collection.query(
                query_texts=[request.message],
                n_results=3
            )
            
            if results and 'documents' in results and results['documents'][0]:
                search_results = results['documents'][0]
                context_length = len(' '.join(search_results))
        
        context = "\n".join(search_results) if search_results else ""
        
        if context:
            prompt = f"Context from {USER}'s documents:\n{context}\n\nUser: {request.message}\nAssistant:"
        else:
            prompt = f"User {USER}: {request.message}\nAssistant:"
        
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'num_gpu': 1,
                'temperature': 0.7,
                'num_ctx': 4096
            }
        )
        ai_response = response['message']['content']
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=ai_response,
            conversation_id=str(uuid.uuid4()),
            sources=search_results if search_results else [],
            processing_time=processing_time,
            model_used="llama3.1:8b",
            context_length=context_length
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return ChatResponse(
            response=f"Error: {str(e)}",
            conversation_id=str(uuid.uuid4()),
            processing_time=processing_time
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
