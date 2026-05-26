"""
main.py
-------
FastAPI application entry point.
Initialises all services at startup and mounts the chat router.
"""

import json
import os
import faiss  
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.routes.chat import router
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.llm_service import LLMService
from backend.app.services.memory_service import MemoryService
from backend.app.services.rag_service import RAGService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.utils.logger import get_logger
from backend.app.vectorstore.faiss_store import FAISSStore

# Load environment variables from .env
load_dotenv()

logger = get_logger(__name__)

# ── Application factory ──────────────────────────────────────────────────────

app = FastAPI(
    title="GenAI RAG Assistant",
    description="Company support chatbot powered by RAG + Gemini",
    version="1.0.0",
)

# CORS – allow the frontend to reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-gemini-assistant-1.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# ── Health check endpoint ────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=== Application startup (Static Loading) ===")
    
    root_dir = Path(__file__).resolve().parent.parent.parent
    index_path = root_dir / "vector_store.index"
    chunks_path = root_dir / "chunks.json"
    
    logger.info(f"Looking for index at: {index_path}")
    logger.info(f"Looking for chunks at: {chunks_path}")
    
    if not index_path.exists() or not chunks_path.exists():
        logger.error(f"Files not found at {root_dir}")
        raise FileNotFoundError(f"Required initialization files missing at {root_dir}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Initialise Embedding Service and warm up the model
    embedding_service = EmbeddingService()
    try:
        embedding_service.model.warmup()
        logger.info("Embedding model warmed up successfully.")
    except Exception as e:
        logger.warning(f"Could not warm up embedding model (this may be normal depending on the model): {e}")
    
    faiss_store = FAISSStore(dimension=index.d) 
    faiss_store.index = index 
    faiss_store.chunks = chunks
    
    retrieval_service = RetrievalService(faiss_store, embedding_service)
    llm_service = LLMService()
    memory_service = MemoryService()
    rag_service = RAGService(retrieval_service, llm_service, memory_service)
    app.state.rag_service = rag_service

    logger.info("=== All services initialised from file. Ready to serve requests. ===")


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)