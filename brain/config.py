"""Shared configuration for brain services.

LLM settings are read from environment variables so the endpoint and model
can be overridden without touching code:

    LLM_BASE_URL   vLLM OpenAI-compatible base URL  (default: http://localhost:8000/v1)
    LLM_MODEL      Model name served by vLLM         (default: Qwen/Qwen3-8B-AWQ)
    LLM_API_KEY    API key (vLLM ignores it, but some proxies need one)

Satellite connection (used by pipeline.py):
    SATELLITE_HOST  IP of the Raspberry Pi            (default: 127.0.0.1)
    SATELLITE_PORT  Wyoming port on the satellite      (default: 10700)
"""

import os

# Wyoming service ports (standard conventions)
ASR_PORT = 10300
TTS_PORT = 10200

# Bind to all interfaces so the satellite can reach us over LAN
BIND_HOST = "0.0.0.0"

# --- VRAM budget (8 GB RTX 3060) ---
# vLLM gets ~70% of VRAM; faster-whisper runs on CPU to avoid contention.
VLLM_GPU_MEMORY_UTILIZATION = 0.70
VLLM_MAX_MODEL_LEN = 8192
WHISPER_DEVICE = "cpu"          # change to "cuda" only if VRAM allows
WHISPER_MODEL = "small"

# LLM — all overridable via environment variables
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "Qwen/Qwen3-8B-AWQ")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "EMPTY")  # vLLM accepts any non-empty key

# Satellite Wyoming server (pipeline.py connects here)
SATELLITE_HOST = os.getenv("SATELLITE_HOST", "127.0.0.1")
SATELLITE_PORT = int(os.getenv("SATELLITE_PORT", "10700"))

# Piper TTS — overridable via PIPER_MODEL_PATH
# Default matches the path created by install_brain.sh
PIPER_MODEL_PATH = os.getenv(
    "PIPER_MODEL_PATH",
    os.path.expanduser("~/brain/models/piper/en_US-lessac-medium.onnx"),
)
