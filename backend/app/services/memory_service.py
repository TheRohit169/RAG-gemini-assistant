"""
memory_service.py
-----------------
In-memory session-based conversation history (last 5 pairs per session).
"""

from typing import List, Dict
from collections import defaultdict

from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_HISTORY_PAIRS = 5


class MemoryService:
    """Stores and manages per-session conversation history."""

    def __init__(self):
        # session_id → list of {"role": "user"|"assistant", "content": str}
        self._store: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Returns the conversation history for a session."""
        return list(self._store[session_id])

    def add_exchange(
        self, session_id: str, user_message: str, assistant_reply: str
    ) -> None:
        """
        Appends a user-assistant exchange and trims to the last MAX_HISTORY_PAIRS pairs.
        """
        history = self._store[session_id]
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})

        # Keep only the last N pairs (2 * MAX_HISTORY_PAIRS messages)
        max_messages = MAX_HISTORY_PAIRS * 2
        if len(history) > max_messages:
            self._store[session_id] = history[-max_messages:]

        logger.debug(
            f"Session '{session_id}' history size: {len(self._store[session_id])} messages"
        )

    def format_history(self, session_id: str) -> str:
        """Returns history formatted as a readable string for prompt injection."""
        history = self.get_history(session_id)
        if not history:
            return "No prior conversation."

        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        """Clears history for a session."""
        self._store.pop(session_id, None)
        logger.info(f"Cleared history for session '{session_id}'")
