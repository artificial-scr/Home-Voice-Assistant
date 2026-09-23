"""
Integration tests for brain/pipeline.py event handling.

Uses in-process fakes for AsyncTcpClient and LLMClient — no real TCP or LLM
calls.  Tests that the pipeline routes Wyoming events correctly and sends
Synthesize events in the right order/count.
"""

import asyncio
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Patch wyoming types used by pipeline so we can introspect calls
from pipeline import _MIN_SENTENCE_CHARS, handle_transcript


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLLM:
    """Yields pre-canned token chunks from stream_chat."""

    def __init__(self, chunks: List[str]):
        self._chunks = chunks

    async def stream_chat(self, text: str) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class RecordingClient:
    """Records every event passed to write_event."""

    def __init__(self):
        self.written: List[object] = []

    async def write_event(self, event):
        self.written.append(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Because wyoming is fully mocked, Synthesize is a MagicMock.  We intercept
# calls to it by patching pipeline.Synthesize with a real wrapper that records
# the text argument.

class _CapturingSynthesize:
    """Drop-in for wyoming.tts.Synthesize that records text args."""
    captured: List[str] = []

    def __init__(self, text: str):
        _CapturingSynthesize.captured.append(text)
        self._text = text

    def event(self):
        return MagicMock()

    @classmethod
    def reset(cls):
        cls.captured = []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_capture():
    _CapturingSynthesize.reset()


@pytest.mark.asyncio
async def test_single_sentence_one_synthesize_event():
    llm = FakeLLM(["Hello there. "])
    client = RecordingClient()

    with patch("pipeline.Synthesize", _CapturingSynthesize):
        await handle_transcript("hi", llm, client)

    assert len(_CapturingSynthesize.captured) == 1
    assert _CapturingSynthesize.captured[0] == "Hello there."


@pytest.mark.asyncio
async def test_three_sentences_three_synthesize_events():
    llm = FakeLLM([
        "First sentence. ",
        "Second sentence. ",
        "Third sentence.",
    ])
    client = RecordingClient()

    with patch("pipeline.Synthesize", _CapturingSynthesize):
        await handle_transcript("question", llm, client)

    assert len(_CapturingSynthesize.captured) == 3
    assert _CapturingSynthesize.captured[0] == "First sentence."
    assert _CapturingSynthesize.captured[1] == "Second sentence."
    assert _CapturingSynthesize.captured[2] == "Third sentence."


@pytest.mark.asyncio
async def test_tail_below_min_chars_not_synthesized():
    # Tail "OK." is 3 chars — below _MIN_SENTENCE_CHARS, should not be sent
    llm = FakeLLM(["OK."])
    client = RecordingClient()

    with patch("pipeline.Synthesize", _CapturingSynthesize):
        await handle_transcript("hi", llm, client)

    assert len(_CapturingSynthesize.captured) == 0


@pytest.mark.asyncio
async def test_tail_at_min_chars_is_synthesized():
    tail = "A" * _MIN_SENTENCE_CHARS
    llm = FakeLLM([tail])
    client = RecordingClient()

    with patch("pipeline.Synthesize", _CapturingSynthesize):
        await handle_transcript("hi", llm, client)

    assert len(_CapturingSynthesize.captured) == 1
    assert _CapturingSynthesize.captured[0] == tail


@pytest.mark.asyncio
async def test_chunks_accumulate_before_split():
    # Sentence split only happens when a boundary appears; partial chunks
    # must not trigger early synthesis.
    llm = FakeLLM(["Hel", "lo wor", "ld. ", "Next bit"])
    client = RecordingClient()

    with patch("pipeline.Synthesize", _CapturingSynthesize):
        await handle_transcript("hi", llm, client)

    # One complete sentence flushed mid-stream, remainder in tail (< 8 chars)
    assert _CapturingSynthesize.captured[0] == "Hello world."


@pytest.mark.asyncio
async def test_empty_llm_response_no_synthesize():
    llm = FakeLLM([])
    client = RecordingClient()

    with patch("pipeline.Synthesize", _CapturingSynthesize):
        await handle_transcript("hi", llm, client)

    assert len(_CapturingSynthesize.captured) == 0


@pytest.mark.asyncio
async def test_sentences_flushed_in_order():
    # Order must be preserved even when chunks straddle sentence boundaries.
    # The final chunk "Last sentence here." is 19 chars >= _MIN_SENTENCE_CHARS.
    llm = FakeLLM(["One. Two. ", "Last sentence here."])
    client = RecordingClient()

    with patch("pipeline.Synthesize", _CapturingSynthesize):
        await handle_transcript("hi", llm, client)

    assert _CapturingSynthesize.captured == ["One.", "Two.", "Last sentence here."]


@pytest.mark.asyncio
async def test_whitespace_only_transcript_no_llm_call():
    # handle_transcript is only called when text.strip() is non-empty
    # (the guard lives in run_pipeline), but verify the function itself
    # handles an already-stripped empty string gracefully.
    call_count = 0

    class CountingLLM:
        async def stream_chat(self, text):
            nonlocal call_count
            call_count += 1
            return
            yield  # make it an async generator

    with patch("pipeline.Synthesize", _CapturingSynthesize):
        await handle_transcript("", CountingLLM(), RecordingClient())

    # LLM was called (handle_transcript doesn't guard empty text itself),
    # but nothing was synthesized.
    assert call_count == 1
    assert len(_CapturingSynthesize.captured) == 0
