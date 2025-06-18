import os
import zipfile
import shutil
import git
import stat
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_ollama.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryInput(BaseModel):
    query: str

class RepoInput(BaseModel):
    url: str

DB_PATH = "./vector_store"
WORK_DIR = "./repos"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

vector_store = None
qa_chain = None
llm = ChatOllama(model="llama3.1:8b", temperature=0)

def load_docs(path):
    allowed = [".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".md", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".swift", ".kt"]
    docs = []
    for root, _, files in os.walk(path):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in allowed:
                try:
                    loader = TextLoader(os.path.join(root, file))
                    docs.extend(loader.load())
                except:
                    pass
    print(f"✅ Loaded {len(docs)} documents from {path}")
    return docs

def rebuild_vector_store():
    all_docs = []
    for root, dirs, _ in os.walk(WORK_DIR):
        for d in dirs:
            docs = load_docs(os.path.join(root, d))
            all_docs.extend(docs)
    if not all_docs:
        print("❌ No documents found.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        print("❌ No chunks created.")
        return None
    print(f"✅ Total Chunks: {len(chunks)}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(DB_PATH)
        return vector_store
    except Exception as e:
        print(f"❌ Vector store build failed: {e}")
        return None

@app.post("/ask")
async def ask_question(input: QueryInput):
    if not vector_store or not qa_chain:
        return {"result": "❌ Upload a repo or zip file first."}
    result = qa_chain.invoke({"query": input.query})
    return {"result": result["result"]}

@app.post("/upload_zip")
async def upload_zip(file: UploadFile = File(...)):
    global vector_store, qa_chain
    contents = await file.read()
    zip_path = os.path.join(WORK_DIR, file.filename)
    with open(zip_path, "wb") as f:
        f.write(contents)
    extract_path = os.path.join(WORK_DIR, file.filename.replace(".zip", ""))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(zip_path)
    vector_store = rebuild_vector_store()
    if vector_store:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 5}), return_source_documents=True)
    return {"message": "ZIP uploaded and processed."}

@app.post("/clone_repo")
async def clone_repo(repo: RepoInput):
    global vector_store, qa_chain
    folder_name = repo.url.rstrip("/").split("/")[-1]
    dest = os.path.join(WORK_DIR, folder_name)
    if os.path.exists(dest):
        def on_rm_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(dest, onerror=on_rm_error)
    git.Repo.clone_from(repo.url, dest)
    vector_store = rebuild_vector_store()
    if vector_store:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 5}), return_source_documents=True)
    return {"message": "Repository cloned and processed."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.middleware("http")
async def catch_unknown_routes(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 404:
        return JSONResponse(content={"detail": f"Path {request.url.path} not found."}, status_code=404)
    return response
