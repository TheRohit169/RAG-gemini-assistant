"""
llm_service.py
--------------
Interfaces with Google Gemini for text generation with retry logic.
"""

import os
import time
from typing import Tuple, Optional

import google.generativeai as genai
from google.api_core.exceptions import (
    InvalidArgument,
    ResourceExhausted,
    DeadlineExceeded,
    GoogleAPIError,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.2


class LLMService:
    """Sends prompts to Gemini and returns (reply_text, tokens_used)."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=genai.GenerationConfig(temperature=TEMPERATURE),
        )
        logger.info(f"LLM service initialised with model: {MODEL_NAME}")

    def generate(self, prompt: str) -> Tuple[str, Optional[int]]:
        """
        Sends the prompt to Gemini and returns the response with retry logic.

        Args:
            prompt: Fully constructed RAG prompt

        Returns:
            Tuple of (reply_text, tokens_used or None)
        """
        retries = 3
        for attempt in range(retries):
            try:
                response = self._model.generate_content(prompt)
                reply = response.text.strip()

                # Token usage (if available in the response metadata)
                tokens_used: Optional[int] = None
                try:
                    usage = response.usage_metadata
                    tokens_used = (
                        usage.prompt_token_count + usage.candidates_token_count
                    )
                    logger.info(
                        f"Token usage – prompt: {usage.prompt_token_count}, "
                        f"response: {usage.candidates_token_count}, "
                        f"total: {tokens_used}"
                    )
                except Exception:
                    logger.debug("Token usage metadata not available")

                return reply, tokens_used

            except ResourceExhausted as exc:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2  # Wait 2s, 4s, 6s...
                    logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {retries} attempts: {exc}")
                    raise RuntimeError("Rate limit exceeded. Please wait a minute.")

            except InvalidArgument as exc:
                logger.error(f"Invalid API key or argument: {exc}")
                raise RuntimeError("Invalid API key. Please check your GEMINI_API_KEY.") from exc

            except DeadlineExceeded as exc:
                logger.error(f"Request timed out: {exc}")
                raise RuntimeError("The request to Gemini timed out. Please retry.") from exc

            except GoogleAPIError as exc:
                logger.error(f"Gemini API error: {exc}")
                raise RuntimeError(f"Gemini API error: {str(exc)}") from exc

            except Exception as exc:
                logger.error(f"Unexpected LLM error: {exc}")
                raise RuntimeError(f"Unexpected error communicating with LLM: {str(exc)}") from exc