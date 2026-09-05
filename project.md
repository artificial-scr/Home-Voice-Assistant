# Project: Custom Voice Assistant

## Architecture

```
SATELLITE (Raspberry Pi)          BRAIN (Linux desktop, RTX 3060)
  Mic → VAD → Wake word    --->     STT → LLM → TTS
  Speaker  <---------------------   audio out
                (Wyoming protocol)
```

## Components

| Layer | Tech |
|---|---|
| Transport | Wyoming protocol |
| Satellite hardware | Raspberry Pi 4/5 + mic (ReSpeaker HAT recommended) + speaker |
| Wake word | openWakeWord (runs on satellite) |
| VAD | webrtcvad or silero-vad (runs on satellite) |
| STT | faster-whisper, GPU, `small` or `medium` model (runs on brain) |
| LLM serving | vLLM, OpenAI-compatible API (runs on brain) |
| LLM model | Qwen3-8B-AWQ (fallback: Qwen3-4B if VRAM-constrained) |
| TTS | Piper, stock voice (runs on brain) |
| Orchestration | Pipecat (fallback: hand-rolled asyncio) |
| Tool-calling | OpenAI-style function calling via vLLM |

## Scope

- Fully local. No cloud LLM in v1.
- General-assistant tool surface: web search, calculator, timers, notes, weather.
- Tool registry must be extensible (adding a tool = no changes to core loop).
- Single satellite, single brain for v1. Multi-satellite is out of scope for v1.

## Build order

1. Confirm RTX 3060 VRAM (8GB or 12GB). Start vLLM standalone with Qwen3-8B-AWQ. Measure VRAM usage.
2. Set up Wyoming client (satellite) ↔ server (brain) audio streaming, no AI components. Verify bidirectional audio over LAN.
3. Add openWakeWord on satellite. Wake word triggers "start streaming" to brain.
4. Add faster-whisper on brain. Receive streamed audio, output transcript. Confirm VRAM coexistence with vLLM.
5. Send transcript to vLLM. Print text response. No tools, no TTS yet.
6. Add Piper on brain. Synthesize response, stream audio back to satellite, play on speaker.
7. Optimize: chunk LLM output by sentence, start TTS per sentence before full response completes. Stream TTS audio back incrementally.
8. Build tool registry. Implement: web search, calculator, timer. Wire to vLLM function-calling.
9. Add conversation history management (rolling window, capped to `max_model_len`).
10. Add remaining tools (notes, weather) via the registry.

## Constraints / config notes

- Cap `max_model_len` and `gpu_memory_utilization` in vLLM to leave room for faster-whisper on the same GPU.
- If VRAM contention blocks both models running concurrently: run faster-whisper on CPU instead.
- LAN connection between satellite and brain must be wired or stable Wi-Fi.

## Out of scope for v1

- Cloud LLM fallback (revisit after evaluating local model quality)
- Voice cloning / non-stock TTS voices
- Multiple satellites
- Android satellite
