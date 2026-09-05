"""
Brain pipeline controller — Steps 5, 6 & 7.

Connects to the wyoming-satellite as a Wyoming client and drives the
full voice assistant loop:

  Detection → Transcript → LLM (streaming) → Synthesize per sentence
                                           → satellite plays TTS audio

Streaming TTS (Step 7): LLM tokens are accumulated into a buffer. Each
time a sentence boundary is detected the sentence is sent as a Synthesize
event immediately — Piper starts speaking sentence 1 while the LLM is
still generating sentence 2, cutting perceived latency significantly.

The satellite handles ASR (--asr-uri → asr_whisper.py) and speaker
playback (--tts-uri → tts_piper.py). This controller handles only the
Transcript → LLM → Synthesize orchestration.

Run:
    python brain/pipeline.py
    SATELLITE_HOST=192.168.1.50 python brain/pipeline.py
"""

import asyncio
import logging
import re
import sys
from typing import List, Tuple

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

_RECONNECT_DELAY = 5.0

# Sentence boundary: .!? followed by whitespace.
# Simple but sufficient — the system prompt keeps responses short.
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# Don't synthesize fragments shorter than this (avoids sending "OK." alone
# when the model starts with an acknowledgement before the real answer).
_MIN_SENTENCE_CHARS = 8


def _extract_sentences(buf: str) -> Tuple[List[str], str]:
    """Split buf at sentence boundaries.

    Returns (complete_sentences, remaining_buffer). Each item in
    complete_sentences ends with punctuation and is ready for TTS.
    remaining_buffer has no trailing sentence boundary yet.
    """
    parts = _SENTENCE_END.split(buf)
    if len(parts) == 1:
        return [], buf
    sentences = [s.strip() for s in parts[:-1] if s.strip()]
    return sentences, parts[-1]


async def _synthesize_sentence(sentence: str, client: AsyncTcpClient) -> None:
    """Send one sentence to the satellite for TTS playback."""
    _LOGGER.info("TTS ← %r", sentence)
    await client.write_event(Synthesize(text=sentence).event())


async def handle_transcript(
    text: str, llm: LLMClient, client: AsyncTcpClient
) -> None:
    """Transcript → streaming LLM → per-sentence Synthesize events."""
    _LOGGER.info("Transcript: %r", text)

    buf = ""
    full_reply_parts: List[str] = []

    async for chunk in llm.stream_chat(text):
        buf += chunk
        full_reply_parts.append(chunk)
        sentences, buf = _extract_sentences(buf)
        for sentence in sentences:
            await _synthesize_sentence(sentence, client)

    # Flush any remaining text (last sentence may lack trailing whitespace)
    tail = buf.strip()
    if len(tail) >= _MIN_SENTENCE_CHARS:
        await _synthesize_sentence(tail, client)
    elif tail:
        # Too short to synthesise on its own — append to log only
        full_reply_parts.append(tail)

    _LOGGER.info("LLM reply (full): %r", "".join(full_reply_parts))


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
