#!/usr/bin/env bash
# Launch brain services.
# Swap in real implementations as each step completes:
#   Step 4: real ASR (asr_whisper.py)   — TTS still a stub
#   Step 6: real TTS (tts_piper.py)     — both real
#
# Run from the repo root inside the brain venv:
#   source ~/brain/.venv/bin/activate
#   bash run_brain.sh
#
# Optional flags:
#   --stub-asr     use asr_stub.py instead of asr_whisper.py
#   --stub-tts     use tts_stub.py instead of tts_piper.py  (default until Step 6)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USE_STUB_ASR=false
USE_STUB_TTS=true   # TTS stub is still the default until Step 6

while [[ $# -gt 0 ]]; do
  case $1 in
    --stub-asr) USE_STUB_ASR=true; shift ;;
    --stub-tts) USE_STUB_TTS=true; shift ;;
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

wait
