"""
Async LLM client — thin wrapper around the vLLM OpenAI-compatible API.

All settings are read from environment variables (see config.py):
    LLM_BASE_URL   e.g. http://192.168.1.100:8000/v1
    LLM_MODEL      e.g. Qwen/Qwen3-8B-AWQ
    LLM_API_KEY    any non-empty string (vLLM ignores the value)

Usage:
    client = LLMClient()
    reply = await client.chat("What is the capital of France?")
"""

import logging
from typing import AsyncIterator, List

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

import config

_LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful home voice assistant. "
    "Give short, spoken-language answers — one or two sentences unless more detail is clearly needed. "
    "Do not use markdown, bullet points, or any formatting that doesn't read naturally aloud."
)


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )
        self._model = config.LLM_MODEL
        _LOGGER.info("LLM client ready  model=%s  base_url=%s", self._model, config.LLM_BASE_URL)

    async def chat(self, user_text: str) -> str:
        """Single-turn chat. Returns the full assistant reply as a string."""
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]
        _LOGGER.debug("→ LLM  %r", user_text)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )
        reply = response.choices[0].message.content or ""
        _LOGGER.debug("← LLM  %r", reply)
        return reply.strip()

    async def stream_chat(self, user_text: str) -> AsyncIterator[str]:
        """Single-turn streaming chat. Yields text chunks as they arrive.

        Used in Step 7 to start TTS sentence-by-sentence before the full
        response is complete.
        """
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]
        _LOGGER.debug("→ LLM (stream)  %r", user_text)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
