"""
Wyoming ASR stub server — Step 2 (audio pipe validation).

Accepts audio from the satellite, waits for AudioStop, then returns a
hardcoded transcript so you can verify the full audio path without any ML.

Run:
    python brain/asr_stub.py
"""

import asyncio
import logging

from wyoming.asr import AsrModel, AsrProgram, Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

import config

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [asr-stub] %(message)s")
_LOGGER = logging.getLogger(__name__)

_INFO = Info(
    asr=[
        AsrProgram(
            name="stub-asr",
            description="Stub ASR — returns a hardcoded transcript",
            version="0.1.0",
            attribution=Attribution(name="stub", url=""),
            installed=True,
            models=[
                AsrModel(
                    name="stub",
                    description="Stub model",
                    attribution=Attribution(name="stub", url=""),
                    installed=True,
                    languages=["en"],
                )
            ],
        )
    ]
)


class AsrStubHandler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunks: int = 0

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(_INFO.event())

        elif AudioStart.is_type(event.type):
            self._chunks = 0
            _LOGGER.debug("Audio stream started")

        elif AudioChunk.is_type(event.type):
            self._chunks += 1

        elif AudioStop.is_type(event.type):
            _LOGGER.info("Audio stream ended (%d chunks). Returning stub transcript.", self._chunks)
            await self.write_event(Transcript(text="stub transcript: audio pipe is working").event())

        return True


async def main() -> None:
    server = AsyncServer.from_uri(f"tcp://{config.BIND_HOST}:{config.ASR_PORT}")
    _LOGGER.info("ASR stub listening on %s:%s", config.BIND_HOST, config.ASR_PORT)
    await server.run(AsrStubHandler)


if __name__ == "__main__":
    asyncio.run(main())
