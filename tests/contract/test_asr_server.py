"""
Contract tests for the ASR Wyoming server (brain/asr_whisper.py).

Spins up a real AsyncServer on a random loopback port, connects with a real
AsyncTcpClient, sends Wyoming events, and asserts the correct events come back.
WhisperModel is stubbed — no ML model loaded.
"""

import asyncio
import struct
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcript
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncServer

from asr_whisper import AsrWhisperHandler, _EXPECTED_CHANNELS, _EXPECTED_RATE, _EXPECTED_WIDTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_whisper_model(return_text: str = "hello world") -> MagicMock:
    model = MagicMock()
    fake_segment = MagicMock()
    fake_segment.text = return_text
    model.transcribe.return_value = ([fake_segment], MagicMock())
    return model


async def _start_server(model, port: int) -> asyncio.Task:
    executor = ThreadPoolExecutor(max_workers=1)
    info = MagicMock(spec=Info)
    info.event.return_value = Info(asr=[]).event()

    def factory(*args, **kwargs):
        return AsrWhisperHandler(*args, model=model, executor=executor, info=info, **kwargs)

    server = AsyncServer.from_uri(f"tcp://127.0.0.1:{port}")
    task = asyncio.create_task(server.run(factory))
    # Give the server a moment to bind
    await asyncio.sleep(0.05)
    return task, executor


def _silent_pcm(seconds: float = 0.1) -> bytes:
    n = int(_EXPECTED_RATE * seconds)
    return b"\x00\x00" * n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_describe_returns_info_event():
    port = 19801
    model = _stub_whisper_model()
    task, executor = await _start_server(model, port)

    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Describe().event())
            event = await asyncio.wait_for(client.read_event(), timeout=3.0)
            assert event is not None
            assert Info.is_type(event.type)
    finally:
        task.cancel()
        executor.shutdown(wait=False)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_audio_produces_transcript():
    port = 19802
    model = _stub_whisper_model("turn on the lights")
    task, executor = await _start_server(model, port)

    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            pcm = _silent_pcm(0.1)
            await client.write_event(
                AudioStart(
                    rate=_EXPECTED_RATE, width=_EXPECTED_WIDTH, channels=_EXPECTED_CHANNELS
                ).event()
            )
            chunk_size = _EXPECTED_RATE * _EXPECTED_WIDTH  # 1 s chunks
            for offset in range(0, len(pcm), chunk_size):
                await client.write_event(
                    AudioChunk(
                        rate=_EXPECTED_RATE,
                        width=_EXPECTED_WIDTH,
                        channels=_EXPECTED_CHANNELS,
                        audio=pcm[offset : offset + chunk_size],
                    ).event()
                )
            await client.write_event(AudioStop().event())

            event = await asyncio.wait_for(client.read_event(), timeout=5.0)
            assert event is not None
            assert Transcript.is_type(event.type)
            transcript = Transcript.from_event(event)
            assert transcript.text == "turn on the lights"
    finally:
        task.cancel()
        executor.shutdown(wait=False)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_wrong_audio_format_no_transcript():
    port = 19803
    model = _stub_whisper_model()
    task, executor = await _start_server(model, port)

    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            # Send 8 kHz / 8-bit mono — wrong format
            await client.write_event(
                AudioStart(rate=8000, width=1, channels=1).event()
            )
            pcm = b"\x00" * 800
            await client.write_event(
                AudioChunk(rate=8000, width=1, channels=1, audio=pcm).event()
            )
            await client.write_event(AudioStop().event())

            # No Transcript should arrive; any event that comes must NOT be a Transcript
            try:
                event = await asyncio.wait_for(client.read_event(), timeout=0.5)
                if event is not None:
                    assert not Transcript.is_type(event.type)
            except asyncio.TimeoutError:
                pass  # timeout is expected — no transcript sent
    finally:
        task.cancel()
        executor.shutdown(wait=False)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_audio_no_transcript():
    port = 19804
    model = _stub_whisper_model()
    task, executor = await _start_server(model, port)

    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(
                AudioStart(
                    rate=_EXPECTED_RATE, width=_EXPECTED_WIDTH, channels=_EXPECTED_CHANNELS
                ).event()
            )
            # No AudioChunk events
            await client.write_event(AudioStop().event())

            try:
                event = await asyncio.wait_for(client.read_event(), timeout=0.5)
                if event is not None:
                    assert not Transcript.is_type(event.type)
            except asyncio.TimeoutError:
                pass
    finally:
        task.cancel()
        executor.shutdown(wait=False)
        await asyncio.gather(task, return_exceptions=True)
