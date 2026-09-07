#!/usr/bin/env bash
# Launch both Wyoming stub servers (ASR + TTS) for audio-pipe validation (Step 2).
# Run from the repo root inside the brain venv:
#   source ~/brain/.venv/bin/activate
#   bash run_stubs.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run_stubs] Starting ASR stub (port 10300) and TTS stub (port 10200)..."

# Run both in background, kill both on exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

python "$SCRIPT_DIR/brain/asr_stub.py" &
python "$SCRIPT_DIR/brain/tts_stub.py" &

wait
