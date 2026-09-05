# Home Voice Assistant

A fully local, self-hosted voice assistant. No cloud LLM. Wake word → speech recognition → language model → text-to-speech, all running on your own hardware.

---

## Architecture

```
SATELLITE (Raspberry Pi)                    BRAIN (Linux desktop, RTX 3060 8 GB)
──────────────────────────────              ──────────────────────────────────────
Mic → VAD → Wake word (openWakeWord)  ───▶  faster-whisper (STT, CPU)
                                             │
Speaker  ◀───────────────────────────────   vLLM  (Qwen3-8B-AWQ, GPU)
                                             │
                   Wyoming protocol          Piper (TTS, CPU)
```

| Layer | Technology |
|---|---|
| Transport | Wyoming protocol (TCP) |
| Satellite hardware | Raspberry Pi 4/5 · ReSpeaker HAT (recommended) · speaker |
| Wake word | openWakeWord (on satellite) |
| VAD | webrtcvad / silero-vad (on satellite) |
| STT | faster-whisper `small` model, CPU (on brain) |
| LLM serving | vLLM, OpenAI-compatible API (on brain) |
| LLM model | Qwen3-8B-AWQ (8 GB VRAM budget; fallback: Qwen3-4B) |
| TTS | Piper, `en_US-lessac-medium` (on brain) |
| Orchestration | Hand-rolled asyncio pipeline |

---

## Hardware requirements

| Component | Minimum |
|---|---|
| Brain CPU | Any modern x86-64 |
| Brain GPU | NVIDIA RTX 3060 **8 GB** (or better) |
| Brain RAM | 16 GB |
| Satellite | Raspberry Pi 4 / 5 |
| Satellite mic | USB mic or ReSpeaker HAT |
| Network | Wired LAN recommended (Wi-Fi works) |

---

## Repository layout

```
brain/
  config.py          Shared config; LLM settings read from env vars
  asr_whisper.py     Wyoming ASR server (faster-whisper)
  asr_stub.py        ASR stub for pipe testing (returns hardcoded transcript)
  tts_stub.py        TTS stub for pipe testing (returns 0.5 s silence)
  llm_client.py      Async OpenAI-compatible LLM client
  pipeline.py        Pipeline controller (Transcript → LLM → TTS)

satellite/
  detect_mic.py      Auto-detect best microphone via sounddevice
  run_satellite.sh   Launch wyoming-satellite + wyoming-openwakeword

install_brain.sh     One-shot dependency installer for the brain
install_satellite.sh One-shot dependency installer for the satellite
requirements-brain.txt  Pinned Python deps for the brain venv
run_brain.sh         Launch all brain services
run_stubs.sh         Launch ASR + TTS stubs only (no ML, for pipe testing)
```

---

## Setup

### Brain (Linux desktop)

**Prerequisites:** NVIDIA driver and CUDA must already be installed.
Verify with `nvidia-smi` before proceeding.

```bash
# 1. Install system deps and Python packages
bash install_brain.sh

# 2. Activate the venv
source ~/brain/.venv/bin/activate

# 3. Install the project's pinned deps
pip install -r requirements-brain.txt

# 4. Pull the LLM model (first run downloads ~5 GB)
vllm serve Qwen/Qwen3-8B-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.70
```

> **8 GB VRAM note:** vLLM is capped at 70 % GPU memory so faster-whisper
> can run on CPU without contention. If you see OOM errors reduce
> `--gpu-memory-utilization` or switch to `Qwen/Qwen3-4B-AWQ`.

### Satellite (Raspberry Pi)

```bash
# Install system deps, wyoming-satellite, wyoming-openwakeword
bash install_satellite.sh

# Verify your mic and speaker
arecord -L    # list capture devices
aplay -L      # list playback devices
```

---

## Running

### Step 1 — Validate the audio pipe (no ML)

Start the brain stubs (no model loading, instant):

```bash
source ~/brain/.venv/bin/activate
bash run_stubs.sh
```

Start the satellite pointing at the brain (replace `BRAIN_IP`):

```bash
# On the Pi
source ~/satellite/.venv/bin/activate
bash satellite/run_satellite.sh --brain-ip BRAIN_IP
```

Trigger the wake word. You should see the stub transcript logged on the brain
and hear 0.5 s of silence from the Pi speaker — confirming bidirectional audio
over the Wyoming pipe.

---

### Step 2 — Full stack (Steps 4 + 5 + 6)

On the brain, start vLLM in one terminal:

```bash
source ~/brain/.venv/bin/activate
vllm serve Qwen/Qwen3-8B-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.70
```

In a second terminal, start the brain services:

```bash
source ~/brain/.venv/bin/activate
SATELLITE_HOST=192.168.1.50 bash run_brain.sh
```

On the satellite:

```bash
bash satellite/run_satellite.sh --brain-ip BRAIN_IP
```

Trigger the wake word → speak → the brain transcribes, calls the LLM, and the satellite
speaks the reply aloud via Piper TTS.

---

## Configuration

### LLM endpoint (env vars)

All LLM settings are overridable via environment variables — no code change needed.

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |
| `LLM_MODEL` | `Qwen/Qwen3-8B-AWQ` | Model name as served by vLLM |
| `LLM_API_KEY` | `EMPTY` | API key (vLLM ignores the value; set for proxies) |
| `PIPER_MODEL_PATH` | `~/brain/models/piper/en_US-lessac-medium.onnx` | Full path to the Piper `.onnx` model |

### Satellite connection (env vars)

| Variable | Default | Description |
|---|---|---|
| `SATELLITE_HOST` | `127.0.0.1` | IP of the Raspberry Pi |
| `SATELLITE_PORT` | `10700` | Wyoming port on the satellite |

Example — brain and satellite on different machines:

```bash
LLM_BASE_URL=http://localhost:8000/v1 \
SATELLITE_HOST=192.168.1.50 \
bash run_brain.sh
```

### run_brain.sh flags

| Flag | Effect |
|---|---|
| `--stub-asr` | Use the hardcoded-transcript stub instead of faster-whisper |
| `--stub-tts` | Use the silence stub instead of Piper (default until Step 6) |
| `--no-pipeline` | Skip the pipeline controller (isolated ASR/TTS testing) |

### run_satellite.sh flags

| Flag | Default | Description |
|---|---|---|
| `--brain-ip IP` | *(required)* | IP of the brain machine |
| `--asr-port N` | `10300` | Wyoming ASR port on brain |
| `--tts-port N` | `10200` | Wyoming TTS port on brain |
| `--wake-word NAME` | `hey_jarvis` | openWakeWord model name |
| `--mic-device N` | auto-detected | sounddevice index (skip auto-detect) |
| `--name NAME` | `home-satellite` | Satellite name shown in Wyoming |

### Mic auto-detection

The satellite launch script auto-detects the best microphone using `sounddevice`.
Priority order: ReSpeaker HAT → USB mic → first input device → system default.

To see what it would pick before launching:

```bash
python satellite/detect_mic.py --list
```

To override:

```bash
bash satellite/run_satellite.sh --brain-ip BRAIN_IP --mic-device 2
```

---

## Port reference

| Port | Service | Direction |
|---|---|---|
| `10200` | TTS (Piper / stub) | satellite → brain |
| `10300` | ASR (faster-whisper / stub) | satellite → brain |
| `10400` | Wake word (openWakeWord, localhost) | satellite-internal |
| `10700` | Satellite Wyoming server | brain pipeline → satellite |
| `8000` | vLLM OpenAI API | pipeline → localhost |

---

## Build progress

| Step | Status | What it adds |
|---|---|---|
| 1 — VRAM check | ✅ | Config tuned for 8 GB (vLLM 70 %, Whisper on CPU) |
| 2 — Wyoming pipe stubs | ✅ | Bidirectional audio over LAN, no ML |
| 3 — Wake word | ✅ | openWakeWord launch script, mic auto-detection |
| 4 — faster-whisper ASR | ✅ | Real speech-to-text on brain |
| 5 — LLM integration | ✅ | Transcript → vLLM → logged reply |
| 6 — Piper TTS | ✅ | Spoken response on satellite speaker |
| 7 — Streaming TTS | ✅ | Per-sentence TTS before full LLM response |
| 8 — Tool registry | 🔲 | Web search, calculator, timer |
| 9 — Conversation history | 🔲 | Rolling window, token-capped |
| 10 — Remaining tools | 🔲 | Notes, weather |
