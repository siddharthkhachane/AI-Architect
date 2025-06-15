from fastapi import HTTPException
from app.core.config import settings
from app.models.schemas import DocumentType, DocumentStatus
from typing import Dict, List, Optional, Any
import PyPDF2
import pdfplumber
import markdown
import os
import uuid
import json
import aiofiles
import asyncio
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
    
    async def process_file(self, file_path: str, document_id: str, filename: str) -> Dict[str, Any]:
        try:
            file_ext = Path(filename).suffix.lower()
            document_type = self._get_document_type(file_ext)
            
            if document_type == DocumentType.PDF:
                content = await self._process_pdf(file_path)
            elif document_type == DocumentType.MARKDOWN:
                content = await self._process_markdown(file_path)
            elif document_type == DocumentType.TEXT:
                content = await self._process_text(file_path)
            elif document_type == DocumentType.CODE:
                content = await self._process_code(file_path, file_ext)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            chunks = self._chunk_text(content)
            
            metadata = {
                "filename": filename,
                "file_path": file_path,
                "document_type": document_type.value,
                "chunk_count": len(chunks),
                "processed_at": datetime.utcnow().isoformat(),
                "file_size": os.path.getsize(file_path)
            }
            
            return {
                "document_id": document_id,
                "title": filename,
                "content": content,
                "chunks": chunks,
                "metadata": metadata,
                "status": DocumentStatus.PROCESSING
            }
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
    def _get_document_type(self, file_ext: str) -> DocumentType:
        type_mapping = {
            ".pdf": DocumentType.PDF,
            ".md": DocumentType.MARKDOWN,
            ".txt": DocumentType.TEXT,
            ".py": DocumentType.CODE,
            ".js": DocumentType.CODE,
            ".tsx": DocumentType.CODE,
            ".jsx": DocumentType.CODE,
            ".ts": DocumentType.CODE,
            ".java": DocumentType.CODE,
            ".cpp": DocumentType.CODE,
            ".c": DocumentType.CODE,
            ".go": DocumentType.CODE,
            ".rs": DocumentType.CODE,
        }
        return type_mapping.get(file_ext, DocumentType.TEXT)
    
    async def _process_pdf(self, file_path: str) -> str:
        content = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        content += f"\n--- Page {page_num} ---\n{page_text}\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {str(e)}")
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            content += f"\n--- Page {page_num} ---\n{page_text}\n"
            except Exception as e2:
                raise ValueError(f"Failed to process PDF with both libraries: {str(e2)}")
        
        return content.strip()
    
    async def _process_markdown(self, file_path: str) -> str:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            md_content = await f.read()
        
        html = markdown.markdown(md_content)
        
        import re
        text = re.sub('<[^<]+?>', '', html)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    async def _process_text(self, file_path: str) -> str:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        return content.strip()
    
    async def _process_code(self, file_path: str, file_ext: str) -> str:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            code_content = await f.read()
        
        language = self._get_language_from_ext(file_ext)
        
        formatted_content = f"Language: {language}\n"
        formatted_content += f"File: {Path(file_path).name}\n"
        formatted_content += "-" * 50 + "\n"
        formatted_content += code_content
        
        return formatted_content
    
    def _get_language_from_ext(self, file_ext: str) -> str:
        lang_mapping = {
            ".py": "Python",
            ".js": "JavaScript",
            ".tsx": "TypeScript React",
            ".jsx": "JavaScript React",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".go": "Go",
            ".rs": "Rust",
        }
        return lang_mapping.get(file_ext, "Unknown")
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "index": len(chunks),
                    "length": current_length
                })
                
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
        
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "index": len(chunks),
                "length": len(current_chunk)
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    async def process_repository(self, repo_url: str, document_id: str) -> Dict[str, Any]:
        import git
        import tempfile
        import shutil
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = Path(temp_dir) / "repo"
                
                git.Repo.clone_from(repo_url, repo_path)
                
                all_content = ""
                file_count = 0
                
                for file_path in repo_path.rglob("*"):
                    if file_path.is_file() and self._should_process_file(file_path):
                        try:
                            relative_path = file_path.relative_to(repo_path)
                            file_content = await self._process_repo_file(file_path)
                            
                            all_content += f"\n\n=== {relative_path} ===\n"
                            all_content += file_content
                            file_count += 1
                            
                        except Exception as e:
                            logger.warning(f"Skipping file {file_path}: {str(e)}")
                            continue
                
                chunks = self._chunk_text(all_content)
                
                metadata = {
                    "repo_url": repo_url,
                    "document_type": "repository",
                    "file_count": file_count,
                    "chunk_count": len(chunks),
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                return {
                    "document_id": document_id,
                    "title": f"Repository: {repo_url.split('/')[-1]}",
                    "content": all_content,
                    "chunks": chunks,
                    "metadata": metadata,
                    "status": DocumentStatus.PROCESSING
                }
                
        except Exception as e:
            logger.error(f"Error processing repository {repo_url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process repository: {str(e)}")
    
    def _should_process_file(self, file_path: Path) -> bool:
        if file_path.name.startswith('.'):
            return False
        
        ignore_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}
        if any(part in ignore_dirs for part in file_path.parts):
            return False
        
        allowed_extensions = {'.py', '.js', '.tsx', '.jsx', '.ts', '.md', '.txt', '.json', '.yml', '.yaml'}
        return file_path.suffix.lower() in allowed_extensions
    
    async def _process_repo_file(self, file_path: Path) -> str:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            return content
        except UnicodeDecodeError:
            return f"[Binary file: {file_path.name}]"