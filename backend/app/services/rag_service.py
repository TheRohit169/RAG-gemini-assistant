"""
rag_service.py
--------------
Orchestrates the full RAG pipeline:
  1. Retrieve relevant chunks
  2. Build grounded prompt
  3. Call LLM
  4. Update memory
"""

from typing import Dict, Any

from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.llm_service import LLMService
from backend.app.services.memory_service import MemoryService
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_MESSAGE = (
    "I could not find enough information in the knowledge base to answer this question."
)

PROMPT_TEMPLATE = """You are a helpful AI assistant for a company's internal support system.

Use ONLY the provided context to answer the user's question.
If the answer is not present in the context, say you do not have enough information.
Be concise, professional, and accurate.

Context:
{retrieved_context}

Conversation History:
{history}

User Question:
{question}

Answer:"""


class RAGService:
    """End-to-end RAG pipeline service."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        memory_service: MemoryService,
    ):
        self.retriever = retrieval_service
        self.llm = llm_service
        self.memory = memory_service

    def answer(self, session_id: str, question: str) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline for a user question.

        Args:
            session_id: Unique session identifier
            question: User's natural-language question

        Returns:
            Dict with keys: reply, tokensUsed, retrievedChunks
        """
        # Step 1: Retrieve relevant chunks (MUST happen before LLM)
        chunks, scores = self.retriever.retrieve(question)

        if not chunks:
            logger.info("No chunks above threshold – returning fallback response")
            self.memory.add_exchange(session_id, question, FALLBACK_MESSAGE)
            return {
                "reply": FALLBACK_MESSAGE,
                "tokensUsed": None,
                "retrievedChunks": 0,
            }

        # Step 2: Build context from retrieved chunks
        context_parts = []
        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            context_parts.append(
                f"[Source: {chunk['source_title']} | Relevance: {score:.2f}]\n{chunk['text']}"
            )
        retrieved_context = "\n\n".join(context_parts)

        # Step 3: Get conversation history
        history = self.memory.format_history(session_id)

        # Step 4: Build prompt
        prompt = PROMPT_TEMPLATE.format(
            retrieved_context=retrieved_context,
            history=history,
            question=question,
        )
        logger.debug(f"Prompt length: {len(prompt)} chars")

        # Step 5: Call LLM
        reply, tokens_used = self.llm.generate(prompt)

        # Step 6: Store in memory
        self.memory.add_exchange(session_id, question, reply)

        return {
            "reply": reply,
            "tokensUsed": tokens_used,
            "retrievedChunks": len(chunks),
        }
