"""
Brain pipeline controller — Steps 5 & 6.

Connects to the wyoming-satellite as a Wyoming client and drives the
full voice assistant loop:

  Detection → Transcript → LLM → Synthesize → satellite plays TTS audio

The satellite handles its own ASR (via --asr-uri → asr_whisper.py) and
speaker playback (via --tts-uri → tts_piper.py). This controller handles
only the Transcript → LLM → Synthesize orchestration.

Run:
    python brain/pipeline.py
    SATELLITE_HOST=192.168.1.50 python brain/pipeline.py
"""

import asyncio
import logging
import sys

from wyoming.asr import Transcript
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.satellite import RunSatellite
from wyoming.tts import Synthesize
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


async def handle_transcript(
    text: str, llm: LLMClient, client: AsyncTcpClient
) -> str:
    """Transcript → LLM → Synthesize sent to satellite."""
    _LOGGER.info("Transcript: %r", text)
    reply = await llm.chat(text)
    _LOGGER.info("LLM reply:  %r", reply)

    # Send Synthesize to the satellite; it calls --tts-uri (tts_piper.py)
    # and plays the resulting audio on the speaker.
    await client.write_event(Synthesize(text=reply).event())

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
                    await handle_transcript(transcript.text.strip(), llm, client)
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
