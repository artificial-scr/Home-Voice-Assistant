"""
Wyoming ASR server backed by faster-whisper — Step 4.

Replaces asr_stub.py. Receives streamed audio from the satellite,
transcribes with faster-whisper on CPU, returns a Transcript event.

Run:
    python brain/asr_whisper.py
    python brain/asr_whisper.py --model medium --device cpu

Model is loaded once at startup and shared across all connections.
Transcription is offloaded to a thread pool so the asyncio loop stays free.
"""

import argparse
import asyncio
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel
from wyoming.asr import AsrModel, AsrProgram, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [asr-whisper] %(message)s")
_LOGGER = logging.getLogger(__name__)

_EXPECTED_RATE = 16000
_EXPECTED_WIDTH = 2   # bytes (16-bit signed PCM)
_EXPECTED_CHANNELS = 1


def _build_info(model_name: str) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="faster-whisper",
                description="faster-whisper speech recognition",
                version="1.0.0",
                attribution=Attribution(name="faster-whisper", url=""),
                installed=True,
                models=[
                    AsrModel(
                        name=model_name,
                        description=f"Whisper {model_name}",
                        attribution=Attribution(name="OpenAI", url=""),
                        installed=True,
                        languages=["en"],
                    )
                ],
            )
        ]
    )


def _pcm_to_float(raw: bytes) -> np.ndarray:
    """Convert raw 16-bit signed little-endian PCM to float32 [-1, 1]."""
    n_samples = len(raw) // _EXPECTED_WIDTH
    samples = struct.unpack(f"<{n_samples}h", raw)
    return np.array(samples, dtype=np.float32) / 32768.0


def _transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    """Run transcription synchronously (called in a thread pool)."""
    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,           # strip silence at edges
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


class AsrWhisperHandler(AsyncEventHandler):
    def __init__(
        self,
        *args,
        model: WhisperModel,
        executor: ThreadPoolExecutor,
        info: Info,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._executor = executor
        self._info = info
        self._audio_buf: List[bytes] = []
        self._rate: int = _EXPECTED_RATE
        self._width: int = _EXPECTED_WIDTH
        self._channels: int = _EXPECTED_CHANNELS

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())

        elif AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self._audio_buf = []
            self._rate = audio_start.rate
            self._width = audio_start.width
            self._channels = audio_start.channels
            _LOGGER.debug("Audio stream started (%d Hz, %d-bit, %d ch)",
                          self._rate, self._width * 8, self._channels)

        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_buf.append(chunk.audio)

        elif AudioStop.is_type(event.type):
            if not self._audio_buf:
                _LOGGER.warning("AudioStop received with no audio — skipping.")
                return True

            raw = b"".join(self._audio_buf)
            total_s = len(raw) / (_EXPECTED_RATE * _EXPECTED_WIDTH * _EXPECTED_CHANNELS)
            _LOGGER.info("Transcribing %.1f s of audio...", total_s)

            audio_f32 = _pcm_to_float(raw)

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self._executor,
                _transcribe,
                self._model,
                audio_f32,
            )

            _LOGGER.info("Transcript: %r", text)
            await self.write_event(Transcript(text=text).event())
            self._audio_buf = []

        return True


async def main(model_name: str, device: str) -> None:
    compute = "int8" if device == "cpu" else "float16"
    _LOGGER.info("Loading faster-whisper model '%s' on %s (compute: %s)...",
                 model_name, device, compute)
    model = WhisperModel(model_name, device=device, compute_type=compute)
    _LOGGER.info("Model loaded.")

    info = _build_info(model_name)
    executor = ThreadPoolExecutor(max_workers=1)

    def handler_factory(*args, **kwargs):
        return AsrWhisperHandler(*args, model=model, executor=executor, info=info, **kwargs)

    server = AsyncServer.from_uri(f"tcp://{config.BIND_HOST}:{config.ASR_PORT}")
    _LOGGER.info("ASR server listening on %s:%s", config.BIND_HOST, config.ASR_PORT)
    await server.run(handler_factory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wyoming faster-whisper ASR server")
    parser.add_argument("--model", default=config.WHISPER_MODEL,
                        help=f"Whisper model name (default: {config.WHISPER_MODEL})")
    parser.add_argument("--device", default=config.WHISPER_DEVICE,
                        choices=["cpu", "cuda"],
                        help=f"Compute device (default: {config.WHISPER_DEVICE})")
    args = parser.parse_args()

    asyncio.run(main(args.model, args.device))
