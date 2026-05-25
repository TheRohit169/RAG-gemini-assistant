"""
schemas.py
----------
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User's message")


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: Optional[int] = None
    retrievedChunks: int


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
