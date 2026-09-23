"""
Integration tests for AsrWhisperHandler event sequencing in brain/asr_whisper.py.

Instantiates the handler directly (no real TCP server), injects Wyoming events,
and asserts the correct Transcript events come out.  WhisperModel is stubbed.
"""

import asyncio
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asr_whisper import (
    AsrWhisperHandler,
    _EXPECTED_CHANNELS,
    _EXPECTED_RATE,
    _EXPECTED_WIDTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcm(n_samples: int = 1600, rate: int = _EXPECTED_RATE) -> bytes:
    """Build n_samples of silent 16-bit mono PCM."""
    return b"\x00\x00" * n_samples


def _audio_start_event(rate=_EXPECTED_RATE, width=_EXPECTED_WIDTH, channels=_EXPECTED_CHANNELS):
    from wyoming.audio import AudioStart
    evt = MagicMock()
    AudioStart.is_type.return_value = False
    AudioStart.from_event.return_value = MagicMock(
        rate=rate, width=width, channels=channels
    )
    # Make AudioStart.is_type return True only for this event
    evt._type = "audio-start"
    return evt


class FakeAudioStart:
    def __init__(self, rate=_EXPECTED_RATE, width=_EXPECTED_WIDTH, channels=_EXPECTED_CHANNELS):
        self.rate = rate
        self.width = width
        self.channels = channels


class FakeAudioChunk:
    def __init__(self, audio: bytes):
        self.audio = audio


class RecordingHandler:
    """Minimal base to capture write_event calls without real TCP."""
    def __init__(self):
        self.written: List[object] = []

    async def write_event(self, event):
        self.written.append(event)


class InstrumentedHandler(AsrWhisperHandler, RecordingHandler):
    """AsrWhisperHandler with write_event replaced by recorder."""
    def __init__(self, model, executor, info):
        RecordingHandler.__init__(self)
        # Skip AsyncEventHandler.__init__ which requires a real reader/writer
        self._model = model
        self._executor = executor
        self._info = info
        self._audio_buf = []
        self._rate = _EXPECTED_RATE
        self._width = _EXPECTED_WIDTH
        self._channels = _EXPECTED_CHANNELS
        self._format_ok = True

    async def write_event(self, event):
        # Override the real AsyncEventHandler.write_event (which needs self.writer)
        self.written.append(event)


def _make_handler(transcribe_return: str = "hello world") -> InstrumentedHandler:
    stub_model = MagicMock()
    # _transcribe unpacks model.transcribe() as (segments, info); configure it.
    stub_model.transcribe.return_value = ([], MagicMock())
    executor = ThreadPoolExecutor(max_workers=1)
    info = MagicMock()

    handler = InstrumentedHandler(stub_model, executor, info)

    # Patch _transcribe so it returns our canned text without loading Whisper
    async def fake_run_in_executor(exc, fn, *args):
        return transcribe_return

    handler._loop_run_in_executor = fake_run_in_executor
    return handler


# ---------------------------------------------------------------------------
# Wyoming event type routing helpers
# ---------------------------------------------------------------------------
# Because wyoming is fully mocked, AudioStart.is_type etc. are MagicMocks.
# We use patch to make them return True/False at the right moments.

def _patch_event_type(event_type_cls_path: str, target_type: str):
    """Return a patcher that makes <EventClass>.is_type(t) True iff t == target_type."""
    # We'll use a simpler approach: directly call handle_event with a fake event
    # whose type string matches what the handler checks.
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audio_start_clears_buffer():
    from wyoming.audio import AudioStart, AudioChunk, AudioStop
    from wyoming.info import Describe

    AudioStart.is_type = lambda t: t == "audio-start"
    AudioStart.from_event = lambda e: FakeAudioStart()
    AudioChunk.is_type = lambda t: t == "audio-chunk"
    AudioChunk.from_event = lambda e: FakeAudioChunk(b"\x00\x00" * 10)
    AudioStop.is_type = lambda t: t == "audio-stop"
    Describe.is_type = lambda t: t == "describe"

    handler = _make_handler("test transcript")
    handler._audio_buf = [b"stale data"]

    start_event = MagicMock()
    start_event.type = "audio-start"

    await handler.handle_event(start_event)
    assert handler._audio_buf == []


@pytest.mark.asyncio
async def test_wrong_format_sets_format_ok_false(caplog):
    import logging
    from wyoming.audio import AudioStart, AudioChunk, AudioStop
    from wyoming.info import Describe

    AudioStart.is_type = lambda t: t == "audio-start"
    AudioStart.from_event = lambda e: FakeAudioStart(rate=8000, width=1, channels=2)
    AudioChunk.is_type = lambda t: t == "audio-chunk"
    AudioStop.is_type = lambda t: t == "audio-stop"
    Describe.is_type = lambda t: t == "describe"

    handler = _make_handler()

    start_event = MagicMock()
    start_event.type = "audio-start"

    with caplog.at_level(logging.ERROR, logger="asr-whisper"):
        await handler.handle_event(start_event)

    assert handler._format_ok is False
    assert "Unsupported audio format" in caplog.text


@pytest.mark.asyncio
async def test_audio_stop_with_bad_format_no_transcript():
    from wyoming.audio import AudioStart, AudioChunk, AudioStop
    from wyoming.info import Describe

    AudioStart.is_type = lambda t: t == "audio-start"
    AudioStart.from_event = lambda e: FakeAudioStart(rate=8000, width=1, channels=2)
    AudioChunk.is_type = lambda t: t == "audio-chunk"
    AudioStop.is_type = lambda t: t == "audio-stop"
    Describe.is_type = lambda t: t == "describe"

    handler = _make_handler()

    start_evt = MagicMock(); start_evt.type = "audio-start"
    stop_evt = MagicMock(); stop_evt.type = "audio-stop"

    await handler.handle_event(start_evt)
    await handler.handle_event(stop_evt)

    assert len(handler.written) == 0  # no Transcript sent


@pytest.mark.asyncio
async def test_audio_stop_with_empty_buffer_no_transcript():
    from wyoming.audio import AudioStart, AudioChunk, AudioStop
    from wyoming.info import Describe

    AudioStart.is_type = lambda t: t == "audio-start"
    AudioStart.from_event = lambda e: FakeAudioStart()
    AudioChunk.is_type = lambda t: t == "audio-chunk"
    AudioStop.is_type = lambda t: t == "audio-stop"
    Describe.is_type = lambda t: t == "describe"

    handler = _make_handler()

    start_evt = MagicMock(); start_evt.type = "audio-start"
    stop_evt = MagicMock(); stop_evt.type = "audio-stop"

    await handler.handle_event(start_evt)
    # No audio chunks added
    await handler.handle_event(stop_evt)

    assert len(handler.written) == 0


@pytest.mark.asyncio
async def test_odd_length_buffer_does_not_raise():
    from wyoming.audio import AudioStart, AudioChunk, AudioStop
    from wyoming.info import Describe

    AudioStart.is_type = lambda t: t == "audio-start"
    AudioStart.from_event = lambda e: FakeAudioStart()
    AudioChunk.is_type = lambda t: t == "audio-chunk"
    AudioChunk.from_event = lambda e: FakeAudioChunk(b"\x00" * 5)  # odd length
    AudioStop.is_type = lambda t: t == "audio-stop"
    Describe.is_type = lambda t: t == "describe"

    handler = _make_handler("ok")

    # Patch run_in_executor to call _transcribe synchronously
    loop = asyncio.get_running_loop()
    original_run = loop.run_in_executor

    async def patched_run(exc, fn, *args):
        # Call the function directly — if it raises on odd buffer, we catch it
        return fn(*args)

    with patch.object(loop, "run_in_executor", patched_run):
        start_evt = MagicMock(); start_evt.type = "audio-start"
        chunk_evt = MagicMock(); chunk_evt.type = "audio-chunk"
        stop_evt = MagicMock(); stop_evt.type = "audio-stop"

        await handler.handle_event(start_evt)
        await handler.handle_event(chunk_evt)
        # Should not raise struct.error
        await handler.handle_event(stop_evt)
