"""
main.py
"""

import json
import faiss  
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat import router
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.utils.logger import get_logger
from app.vectorstore.faiss_store import FAISSStore

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

app = FastAPI(
    title="GenAI RAG Assistant",
    description="Company support chatbot powered by RAG + Gemini",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-gemini-assistant-1.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Dependency injection helper
def get_rag_service():
    return app.state.rag_service

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=== Application startup (Lazy Loading Enabled) ===")
    
    root_dir = Path(__file__).resolve().parent.parent.parent
    index_path = root_dir / "vector_store.index"
    chunks_path = root_dir / "chunks.json"
    
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"Required initialization files missing at {root_dir}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Initialize services without calling .warmup()
    embedding_service = EmbeddingService()
    
    faiss_store = FAISSStore(dimension=index.d) 
    faiss_store.index = index 
    faiss_store.chunks = chunks
    
    retrieval_service = RetrievalService(faiss_store, embedding_service)
    llm_service = LLMService()
    memory_service = MemoryService()
    
    # Bundle services
    rag_service = RAGService(retrieval_service, llm_service, memory_service)
    app.state.rag_service = rag_service

    logger.info("=== All services configured. Ready to serve requests. ===")

# Chat endpoint using dependency injection
@app.post("/chat")
async def chat(session_id: str, question: str, rag: RAGService = Depends(get_rag_service)):
    return rag.answer(session_id, question)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)