"""
Brain pipeline controller — Step 5+.

Connects to the wyoming-satellite as a Wyoming client and drives the
full voice assistant loop:

  Step 5: Detection → Transcript → LLM → print reply
  Step 6: + TTS → stream audio back to satellite

The satellite handles its own ASR (via --asr-uri → asr_whisper.py) and
speaker playback. This controller handles the Transcript → LLM → TTS
orchestration.

Run:
    python brain/pipeline.py
    SATELLITE_HOST=192.168.1.50 python brain/pipeline.py
"""

import asyncio
import logging
import sys

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.satellite import RunSatellite
from wyoming.wake import Detection

import config
from llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [pipeline] %(levelname)s %(message)s",
    stream=sys.stdout,
)
_LOGGER = logging.getLogger(__name__)

# Reconnect delay if the satellite connection drops
_RECONNECT_DELAY = 5.0


async def handle_transcript(text: str, llm: LLMClient) -> str:
    """Call the LLM and return its reply. Step 6 will also trigger TTS here."""
    _LOGGER.info("Transcript: %r", text)
    reply = await llm.chat(text)
    _LOGGER.info("LLM reply:  %r", reply)

    # --- Step 6 hook: TTS goes here ---
    # tts_audio = await synthesize(reply)
    # await client.write_event(AudioStart(...).event())
    # for chunk in tts_audio:
    #     await client.write_event(AudioChunk(...).event())
    # await client.write_event(AudioStop().event())

    return reply


async def run_pipeline(llm: LLMClient) -> None:
    """Main event loop — connects to satellite and processes events."""
    host = config.SATELLITE_HOST
    port = config.SATELLITE_PORT
    _LOGGER.info("Connecting to satellite at %s:%d ...", host, port)

    async with AsyncTcpClient(host, port) as client:
        # Tell the satellite to start its pipeline
        await client.write_event(RunSatellite().event())
        _LOGGER.info("Connected. Waiting for wake word...")

        while True:
            event: Event | None = await client.read_event()
            if event is None:
                _LOGGER.warning("Satellite disconnected.")
                break

            if Detection.is_type(event.type):
                detection = Detection.from_event(event)
                _LOGGER.info("Wake word detected: %s", detection.name)

            elif Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                if transcript.text.strip():
                    await handle_transcript(transcript.text.strip(), llm)
                else:
                    _LOGGER.debug("Empty transcript — ignoring.")


async def main() -> None:
    llm = LLMClient()

    while True:
        try:
            await run_pipeline(llm)
        except ConnectionRefusedError:
            _LOGGER.error(
                "Could not reach satellite at %s:%d — retrying in %.0fs",
                config.SATELLITE_HOST, config.SATELLITE_PORT, _RECONNECT_DELAY,
            )
        except Exception:
            _LOGGER.exception("Pipeline error — retrying in %.0fs", _RECONNECT_DELAY)

        await asyncio.sleep(_RECONNECT_DELAY)


if __name__ == "__main__":
    asyncio.run(main())
