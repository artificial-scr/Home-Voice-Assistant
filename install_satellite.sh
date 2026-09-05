#!/usr/bin/env bash
# Satellite setup — Raspberry Pi (Raspberry Pi OS / Debian-based)
set -e

sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  portaudio19-dev libatlas-base-dev \
  ffmpeg alsa-utils git \
  netcat-openbsd

mkdir -p ~/satellite
cd ~/satellite

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install wyoming webrtcvad pysilero-vad openwakeword

git clone https://github.com/rhasspy/wyoming-satellite.git
cd wyoming-satellite
script/setup

git clone https://github.com/rhasspy/wyoming-openwakeword.git
cd ../wyoming-openwakeword
script/setup

echo "Satellite dependencies installed."
echo "Test mic/speaker with: arecord -L / aplay -L"
