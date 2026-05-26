"""
llm_service.py
--------------
Interfaces with Google Gemini for text generation with lazy initialization.
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
        pass

    def _get_model(self):
        """Internal helper to initialize the model on demand."""
        if LLMService._model is None:
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")
            genai.configure(api_key=api_key)
            LLMService._model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                generation_config=genai.GenerationConfig(temperature=TEMPERATURE),
            )
            logger.info(f"LLM model lazily initialized: {MODEL_NAME}")
        return LLMService._model

    def generate(self, prompt: str) -> Tuple[str, Optional[int]]:
        """Sends prompt to Gemini with retry logic."""
        model = self._get_model()
        retries = 3
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                reply = response.text.strip()
                
                tokens_used: Optional[int] = None
                try:
                    usage = response.usage_metadata
                    tokens_used = usage.prompt_token_count + usage.candidates_token_count
                except Exception:
                    logger.debug("Token usage metadata not available")
                return reply, tokens_used

            except ResourceExhausted as exc:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError("Rate limit exceeded.")
            except (InvalidArgument, DeadlineExceeded, GoogleAPIError) as exc:
                logger.error(f"Gemini API error: {exc}")
                raise RuntimeError(f"Gemini API error: {str(exc)}")
            except Exception as exc:
                logger.error(f"Unexpected LLM error: {exc}")
                raise RuntimeError(f"Unexpected error: {str(exc)}")