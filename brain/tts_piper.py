"""
Wyoming TTS server backed by Piper — Step 6.

Replaces tts_stub.py. Receives a Synthesize event (text), synthesises it
with a local Piper ONNX model, then streams back 16-bit mono PCM as
AudioStart / AudioChunk / AudioStop events.

Run:
    python brain/tts_piper.py
    PIPER_MODEL_PATH=~/brain/models/piper/en_US-lessac-medium.onnx \
        python brain/tts_piper.py

The Piper model is loaded once at startup. Synthesis runs in a thread pool
so the asyncio loop stays free.
"""

import argparse
import asyncio
import io
import logging
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from piper import PiperVoice
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [tts-piper] %(message)s")
_LOGGER = logging.getLogger(__name__)

# Stream in 20 ms chunks (filled in after model load when rate is known)
_CHUNK_MS = 20


def _load_voice(model_path: str) -> PiperVoice:
    config_path = model_path + ".json"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Piper model not found: {model_path}\n"
            "Run install_brain.sh or set PIPER_MODEL_PATH."
        )
    _LOGGER.info("Loading Piper model: %s", model_path)
    return PiperVoice.load(model_path, config_path=config_path)


def _synthesize(voice: PiperVoice, text: str) -> tuple[bytes, int, int, int]:
    """Synthesise text → (raw_pcm, rate, width_bytes, channels). Runs in thread pool."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_out:
        voice.synthesize(text, wav_out)
    buf.seek(0)
    with wave.open(buf, "rb") as wav_in:
        rate = wav_in.getframerate()
        width = wav_in.getsampwidth()
        channels = wav_in.getnchannels()
        raw = wav_in.readframes(wav_in.getnframes())
    return raw, rate, width, channels


def _build_info(voice_name: str) -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="piper",
                description="Piper neural TTS",
                version="1.0.0",
                attribution=Attribution(name="rhasspy", url=""),
                installed=True,
                voices=[
                    TtsVoice(
                        name=voice_name,
                        description=voice_name,
                        attribution=Attribution(name="rhasspy", url=""),
                        installed=True,
                        languages=["en"],
                        speakers=[TtsVoiceSpeaker(name="default")],
                    )
                ],
            )
        ]
    )


class TtsPiperHandler(AsyncEventHandler):
    def __init__(
        self,
        *args,
        voice: PiperVoice,
        executor: ThreadPoolExecutor,
        info: Info,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._voice = voice
        self._executor = executor
        self._info = info

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())

        elif Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            text = synth.text.strip()
            if not text:
                return True

            _LOGGER.info("Synthesizing: %r", text)
            loop = asyncio.get_running_loop()
            raw, rate, width, channels = await loop.run_in_executor(
                self._executor, _synthesize, self._voice, text
            )
            duration_s = len(raw) / (rate * width * channels)
            _LOGGER.info("Synthesized %.2f s of audio — streaming...", duration_s)

            await self.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )
            chunk_size = int(rate * (_CHUNK_MS / 1000)) * width * channels
            for offset in range(0, len(raw), chunk_size):
                await self.write_event(
                    AudioChunk(
                        rate=rate,
                        width=width,
                        channels=channels,
                        audio=raw[offset : offset + chunk_size],
                    ).event()
                )
            await self.write_event(AudioStop().event())
            _LOGGER.debug("Done streaming.")

        return True


async def main(model_path: str) -> None:
    voice = _load_voice(model_path)
    voice_name = Path(model_path).stem  # e.g. "en_US-lessac-medium"
    info = _build_info(voice_name)
    executor = ThreadPoolExecutor(max_workers=1)

    def handler_factory(*args, **kwargs):
        return TtsPiperHandler(*args, voice=voice, executor=executor, info=info, **kwargs)

    server = AsyncServer.from_uri(f"tcp://{config.BIND_HOST}:{config.TTS_PORT}")
    _LOGGER.info("TTS server listening on %s:%s  model=%s", config.BIND_HOST, config.TTS_PORT, voice_name)
    await server.run(handler_factory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wyoming Piper TTS server")
    parser.add_argument(
        "--model",
        default=config.PIPER_MODEL_PATH,
        help=f"Path to Piper .onnx model file (default: {config.PIPER_MODEL_PATH})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model))
