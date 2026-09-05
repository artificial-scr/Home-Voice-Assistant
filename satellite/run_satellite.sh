#!/usr/bin/env bash
# Launch the satellite: wyoming-openwakeword + wyoming-satellite.
#
# Usage:
#   bash satellite/run_satellite.sh --brain-ip <IP>
#
# Optional overrides:
#   --brain-ip   IP of the brain machine          (required)
#   --asr-port   Wyoming ASR port on brain         (default: 10300)
#   --tts-port   Wyoming TTS port on brain         (default: 10200)
#   --wake-word  openWakeWord model name           (default: hey_jarvis)
#   --mic-device sounddevice index (skip auto-detect)
#   --name       Satellite name shown in Wyoming   (default: home-satellite)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Defaults ---
BRAIN_IP=""
ASR_PORT=10300
TTS_PORT=10200
WAKE_WORD="hey_jarvis"
SAT_NAME="home-satellite"
MIC_DEVICE=""
WAKEWORD_PORT=10400

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --brain-ip)   BRAIN_IP="$2";   shift 2 ;;
    --asr-port)   ASR_PORT="$2";   shift 2 ;;
    --tts-port)   TTS_PORT="$2";   shift 2 ;;
    --wake-word)  WAKE_WORD="$2";  shift 2 ;;
    --mic-device) MIC_DEVICE="$2"; shift 2 ;;
    --name)       SAT_NAME="$2";   shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$BRAIN_IP" ]]; then
  echo "ERROR: --brain-ip is required."
  echo "Usage: bash satellite/run_satellite.sh --brain-ip <IP>"
  exit 1
fi

# --- Activate venv ---
VENV="$HOME/satellite/.venv"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: satellite venv not found at $VENV. Run install_satellite.sh first."
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# --- Auto-detect mic if not given ---
if [[ -z "$MIC_DEVICE" ]]; then
  echo "[satellite] Auto-detecting microphone..."
  MIC_DEVICE=$(python "$SCRIPT_DIR/detect_mic.py" --list)
  # --list prints the table then the index; we want only the last line
  MIC_DEVICE=$(echo "$MIC_DEVICE" | tail -n1)
  echo "[satellite] Using mic device index: $MIC_DEVICE"
fi

# --- Clean up on exit ---
trap 'echo "[satellite] Shutting down..."; kill $(jobs -p) 2>/dev/null; wait' EXIT

# --- Start wyoming-openwakeword ---
OWW_DIR="$HOME/satellite/wyoming-openwakeword"
if [[ ! -d "$OWW_DIR" ]]; then
  echo "ERROR: wyoming-openwakeword not found at $OWW_DIR. Run install_satellite.sh first."
  exit 1
fi

echo "[satellite] Starting wyoming-openwakeword (wake word: $WAKE_WORD) on port $WAKEWORD_PORT..."
"$OWW_DIR/script/run" \
  --uri "tcp://127.0.0.1:$WAKEWORD_PORT" \
  --preload-model "$WAKE_WORD" &

# Give it a moment to bind
sleep 2

# --- Start wyoming-satellite ---
SAT_DIR="$HOME/satellite/wyoming-satellite"
if [[ ! -d "$SAT_DIR" ]]; then
  echo "ERROR: wyoming-satellite not found at $SAT_DIR. Run install_satellite.sh first."
  exit 1
fi

echo "[satellite] Starting wyoming-satellite → brain at $BRAIN_IP..."
"$SAT_DIR/script/run" \
  --name "$SAT_NAME" \
  --uri "tcp://0.0.0.0:10700" \
  --mic-device "$MIC_DEVICE" \
  --wake-uri "tcp://127.0.0.1:$WAKEWORD_PORT" \
  --wake-word-name "$WAKE_WORD" \
  --asr-uri "tcp://$BRAIN_IP:$ASR_PORT" \
  --tts-uri "tcp://$BRAIN_IP:$TTS_PORT" \
  --debug &

wait
