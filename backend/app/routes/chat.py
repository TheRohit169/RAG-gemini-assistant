"""
chat.py
-------
FastAPI router for /api/chat and /health endpoints.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health-check endpoint."""
    return {"status": "healthy"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """
    Accepts a user message and session ID, runs the RAG pipeline,
    and returns a grounded AI reply.
    """
    rag_service = request.app.state.rag_service

    session_id = body.sessionId.strip()
    message = body.message.strip()

    if not session_id:
        raise HTTPException(status_code=422, detail="sessionId must not be blank")
    if not message:
        raise HTTPException(status_code=422, detail="message must not be blank")

    logger.info(f"Chat request | session='{session_id}' | message='{message[:80]}'")

    try:
        result = rag_service.answer(session_id=session_id, question=message)
    except RuntimeError as exc:
        logger.error(f"RAG pipeline error: {exc}")
        return JSONResponse(status_code=502, content={"error": str(exc)})
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        return JSONResponse(
            status_code=500, content={"error": "An internal server error occurred."}
        )

    return ChatResponse(
        reply=result["reply"],
        tokensUsed=result.get("tokensUsed"),
        retrievedChunks=result["retrievedChunks"],
    )
