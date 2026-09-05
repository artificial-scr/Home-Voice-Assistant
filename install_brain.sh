#!/usr/bin/env bash
# Brain setup — Linux desktop, RTX 3060 (Ubuntu/Debian-based)
set -e

# Requires NVIDIA driver + CUDA already installed. Verify:
nvidia-smi

sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  build-essential git ffmpeg

mkdir -p ~/brain
cd ~/brain

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install vllm
pip install faster-whisper
pip install piper-tts
pip install wyoming
pip install pipecat-ai

# Piper voice model (stock, English US, medium quality)
mkdir -p models/piper
python3 -m piper.download_voices en_US-lessac-medium --dir models/piper

echo "Brain dependencies installed."
echo "Pull LLM model with: vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 8192 --gpu-memory-utilization 0.7"
