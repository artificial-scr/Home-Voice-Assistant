#!/usr/bin/env bash
# Launch all brain services: ASR + TTS + pipeline controller.
# Swap in real implementations as each step completes:
#   Step 4: real ASR (asr_whisper.py)   — TTS still a stub
#   Step 5: + pipeline controller (pipeline.py)
#   Step 6: real TTS (tts_piper.py)     — both real
#
# Run from the repo root inside the brain venv:
#   source ~/brain/.venv/bin/activate
#   SATELLITE_HOST=192.168.1.50 bash run_brain.sh
#
# Optional flags:
#   --stub-asr       use asr_stub.py instead of asr_whisper.py
#   --stub-tts       use tts_stub.py instead of tts_piper.py  (default until Step 6)
#   --no-pipeline    skip pipeline.py (useful when testing ASR/TTS in isolation)
#
# LLM env vars (see brain/config.py for docs):
#   LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
#   SATELLITE_HOST, SATELLITE_PORT
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USE_STUB_ASR=false
USE_STUB_TTS=true   # TTS stub is still the default until Step 6
RUN_PIPELINE=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --stub-asr)    USE_STUB_ASR=true;   shift ;;
    --stub-tts)    USE_STUB_TTS=true;   shift ;;
    --no-pipeline) RUN_PIPELINE=false;  shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

trap 'echo "[brain] Shutting down..."; kill $(jobs -p) 2>/dev/null; wait' EXIT

# --- ASR ---
if [[ "$USE_STUB_ASR" == "true" ]]; then
  echo "[brain] Starting ASR stub (port 10300)..."
  python "$SCRIPT_DIR/brain/asr_stub.py" &
else
  echo "[brain] Starting faster-whisper ASR (port 10300)..."
  python "$SCRIPT_DIR/brain/asr_whisper.py" &
fi

# --- TTS ---
if [[ "$USE_STUB_TTS" == "true" ]]; then
  echo "[brain] Starting TTS stub (port 10200)..."
  python "$SCRIPT_DIR/brain/tts_stub.py" &
else
  echo "[brain] Starting Piper TTS (port 10200)..."
  python "$SCRIPT_DIR/brain/tts_piper.py" &
fi

# --- Pipeline controller ---
if [[ "$RUN_PIPELINE" == "true" ]]; then
  echo "[brain] Starting pipeline controller (satellite: ${SATELLITE_HOST:-127.0.0.1}:${SATELLITE_PORT:-10700})..."
  python "$SCRIPT_DIR/brain/pipeline.py" &
fi

wait
