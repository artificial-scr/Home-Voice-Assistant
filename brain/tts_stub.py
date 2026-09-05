"""
Wyoming TTS stub server — Step 2 (audio pipe validation).

Accepts a Synthesize event (text), logs it, then streams back 0.5 s of
silence as 16-bit mono PCM at 22050 Hz so the satellite has audio to play.

Run:
    python brain/tts_stub.py
"""

import asyncio
import logging
import struct

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

import config

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [tts-stub] %(message)s")
_LOGGER = logging.getLogger(__name__)

_RATE = 22050
_CHANNELS = 1
_WIDTH = 2  # bytes (16-bit)

_INFO = Info(
    tts=[
        TtsProgram(
            name="stub-tts",
            description="Stub TTS — returns 0.5 s of silence",
            version="0.1.0",
            attribution=Attribution(name="stub", url=""),
            installed=True,
            voices=[
                TtsVoice(
                    name="stub",
                    description="Stub voice",
                    attribution=Attribution(name="stub", url=""),
                    installed=True,
                    languages=["en"],
                    speakers=[TtsVoiceSpeaker(name="default")],
                )
            ],
        )
    ]
)


def _silence(seconds: float) -> bytes:
    """Generate silence as raw 16-bit signed PCM."""
    n_samples = int(_RATE * seconds)
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


class TtsStubHandler(AsyncEventHandler):
    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(_INFO.event())

        elif Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            _LOGGER.info("Synthesize: %r", synth.text)

            audio = _silence(0.5)
            await self.write_event(
                AudioStart(rate=_RATE, width=_WIDTH, channels=_CHANNELS).event()
            )
            # Send in 20 ms chunks (matching typical Wyoming framing)
            chunk_size = int(_RATE * 0.02) * _WIDTH * _CHANNELS
            for offset in range(0, len(audio), chunk_size):
                await self.write_event(
                    AudioChunk(
                        rate=_RATE,
                        width=_WIDTH,
                        channels=_CHANNELS,
                        audio=audio[offset : offset + chunk_size],
                    ).event()
                )
            await self.write_event(AudioStop().event())
            _LOGGER.debug("Silence sent.")

        return True


async def main() -> None:
    server = AsyncServer.from_uri(f"tcp://{config.BIND_HOST}:{config.TTS_PORT}")
    _LOGGER.info("TTS stub listening on %s:%s", config.BIND_HOST, config.TTS_PORT)
    await server.run(TtsStubHandler)


if __name__ == "__main__":
    asyncio.run(main())
