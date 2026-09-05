"""Shared configuration for brain services."""

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

# LLM (vLLM OpenAI-compatible endpoint)
LLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "Qwen/Qwen3-8B-AWQ"

# Piper voice model
PIPER_VOICE = "en_US-lessac-medium"
PIPER_MODELS_DIR = "models/piper"
