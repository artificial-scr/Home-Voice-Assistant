"""
Contract tests for the TTS Wyoming server (brain/tts_piper.py).

Spins up a real AsyncServer on a random loopback port, connects with a real
AsyncTcpClient, and asserts that Synthesize events produce valid
AudioStart → AudioChunk(s) → AudioStop sequences.
PiperVoice is stubbed — no ONNX model loaded.
"""

import asyncio
import io
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize

from tts_piper import TtsPiperHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RATE = 22050
_WIDTH = 2
_CHANNELS = 1


def _make_wav(n_samples: int = 2205) -> bytes:
    """Build a minimal WAV buffer with silent PCM."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_WIDTH)
        wf.setframerate(_RATE)
        wf.writeframes(b"\x00\x00" * n_samples)
    buf.seek(0)
    return buf.read()


def _stub_piper_voice() -> MagicMock:
    """Returns a PiperVoice stub whose synthesize() writes a valid WAV."""
    voice = MagicMock()

    def fake_synthesize(text, wav_file):
        wav_bytes = _make_wav()
        with io.BytesIO(wav_bytes) as src:
            with wave.open(src, "rb") as src_wav:
                wav_file.setnchannels(src_wav.getnchannels())
                wav_file.setsampwidth(src_wav.getsampwidth())
                wav_file.setframerate(src_wav.getframerate())
                wav_file.writeframes(src_wav.readframes(src_wav.getnframes()))

    voice.synthesize.side_effect = fake_synthesize
    return voice


async def _start_tts_server(voice, port: int):
    from wyoming.info import Attribution, TtsProgram, TtsVoice, TtsVoiceSpeaker

    info = Info(
        tts=[
            TtsProgram(
                name="piper",
                description="stub",
                version="0",
                attribution=Attribution(name="test", url=""),
                installed=True,
                voices=[
                    TtsVoice(
                        name="test-voice",
                        description="test",
                        version="0",
                        attribution=Attribution(name="test", url=""),
                        installed=True,
                        languages=["en"],
                        speakers=[TtsVoiceSpeaker(name="default")],
                    )
                ],
            )
        ]
    )
    executor = ThreadPoolExecutor(max_workers=1)

    def factory(*args, **kwargs):
        return TtsPiperHandler(*args, voice=voice, executor=executor, info=info, **kwargs)

    server = AsyncServer.from_uri(f"tcp://127.0.0.1:{port}")
    task = asyncio.create_task(server.run(factory))
    await asyncio.sleep(0.05)
    return task, executor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_describe_returns_info_event():
    port = 19901
    voice = _stub_piper_voice()
    task, executor = await _start_tts_server(voice, port)

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
async def test_synthesize_produces_audio_sequence():
    port = 19902
    voice = _stub_piper_voice()
    task, executor = await _start_tts_server(voice, port)

    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="hello world").event())

            # Must receive AudioStart first
            event = await asyncio.wait_for(client.read_event(), timeout=5.0)
            assert event is not None
            assert AudioStart.is_type(event.type)
            audio_start = AudioStart.from_event(event)
            assert audio_start.rate > 0
            assert audio_start.width > 0
            assert audio_start.channels > 0

            # Must receive one or more AudioChunk events
            chunks_received = 0
            while True:
                event = await asyncio.wait_for(client.read_event(), timeout=3.0)
                assert event is not None
                if AudioStop.is_type(event.type):
                    break
                assert AudioChunk.is_type(event.type), f"Unexpected event type: {event.type}"
                chunk = AudioChunk.from_event(event)
                assert len(chunk.audio) > 0
                chunks_received += 1

            assert chunks_received >= 1

    finally:
        task.cancel()
        executor.shutdown(wait=False)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_text_no_audio():
    port = 19903
    voice = _stub_piper_voice()
    task, executor = await _start_tts_server(voice, port)

    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="   ").event())

            # No audio should arrive; timeout is expected
            try:
                event = await asyncio.wait_for(client.read_event(), timeout=0.5)
                if event is not None:
                    assert not AudioStart.is_type(event.type)
            except asyncio.TimeoutError:
                pass  # expected
    finally:
        task.cancel()
        executor.shutdown(wait=False)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_audio_format_matches_wav_header():
    """AudioStart rate/width/channels must match the WAV Piper synthesised."""
    port = 19904
    voice = _stub_piper_voice()
    task, executor = await _start_tts_server(voice, port)

    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="test audio format").event())

            event = await asyncio.wait_for(client.read_event(), timeout=5.0)
            assert AudioStart.is_type(event.type)
            audio_start = AudioStart.from_event(event)

            assert audio_start.rate == _RATE
            assert audio_start.width == _WIDTH
            assert audio_start.channels == _CHANNELS
    finally:
        task.cancel()
        executor.shutdown(wait=False)
        await asyncio.gather(task, return_exceptions=True)
