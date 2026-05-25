"""
main.py
-------
FastAPI application entry point.
Initialises all services at startup and mounts the chat router.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.chat import router
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.utils.chunking import chunk_documents
from app.utils.logger import get_logger
from app.vectorstore.faiss_store import FAISSStore

# Load environment variables from .env
load_dotenv()

logger = get_logger(__name__)

# ── Application factory ──────────────────────────────────────────────────────

app = FastAPI(
    title="GenAI RAG Assistant",
    description="Company support chatbot powered by RAG + Gemini",
    version="1.0.0",
)

# CORS – allow the frontend (Netlify / local dev) to reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=== Application startup ===")

    # 1. Load documents
    docs_path = Path(__file__).parent.parent / "docs.json"
    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents)} document(s) from docs.json")

    # 2. Chunk documents
    chunks = chunk_documents(
        documents,
        chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
    )

    # 3. Generate embeddings
    embedding_service = EmbeddingService()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_service.embed(texts)

    # 4. Build FAISS index
    faiss_store = FAISSStore(dimension=embedding_service.dimension)
    faiss_store.add(embeddings, chunks)

    # 5. Wire up services
    retrieval_service = RetrievalService(faiss_store, embedding_service)
    llm_service = LLMService()
    memory_service = MemoryService()
    rag_service = RAGService(retrieval_service, llm_service, memory_service)

    # Store on app.state so routes can access them
    app.state.rag_service = rag_service

    logger.info("=== All services initialised. Ready to serve requests. ===")


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
