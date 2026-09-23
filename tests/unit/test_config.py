"""Unit tests for environment-variable loading in brain/config.py."""

import importlib
import os

import pytest


def _reload_config(monkeypatch, overrides: dict) -> object:
    """Set env vars in overrides, clear the rest, then reload config."""
    env_keys = [
        "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
        "SATELLITE_HOST", "SATELLITE_PORT", "SATELLITE_READ_TIMEOUT",
        "PIPER_MODEL_PATH",
    ]
    for key in env_keys:
        if key in overrides:
            monkeypatch.setenv(key, overrides[key])
        else:
            monkeypatch.delenv(key, raising=False)

    import config
    importlib.reload(config)
    return config


class TestLLMConfig:
    def test_default_base_url(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.LLM_BASE_URL == "http://localhost:8000/v1"

    def test_override_base_url(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LLM_BASE_URL": "http://192.168.1.100:8000/v1"})
        assert cfg.LLM_BASE_URL == "http://192.168.1.100:8000/v1"

    def test_default_model(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.LLM_MODEL == "Qwen/Qwen3-8B-AWQ"

    def test_override_model(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LLM_MODEL": "meta-llama/Llama-3-8B"})
        assert cfg.LLM_MODEL == "meta-llama/Llama-3-8B"

    def test_default_api_key(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.LLM_API_KEY == "EMPTY"

    def test_override_api_key(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LLM_API_KEY": "sk-secret"})
        assert cfg.LLM_API_KEY == "sk-secret"


class TestSatelliteConfig:
    def test_default_host(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.SATELLITE_HOST == "127.0.0.1"

    def test_override_host(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"SATELLITE_HOST": "192.168.1.50"})
        assert cfg.SATELLITE_HOST == "192.168.1.50"

    def test_default_port(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.SATELLITE_PORT == 10700

    def test_override_port(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"SATELLITE_PORT": "10800"})
        assert cfg.SATELLITE_PORT == 10800

    def test_port_is_int(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert isinstance(cfg.SATELLITE_PORT, int)

    def test_default_read_timeout(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.SATELLITE_READ_TIMEOUT == 30.0

    def test_override_read_timeout(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"SATELLITE_READ_TIMEOUT": "60"})
        assert cfg.SATELLITE_READ_TIMEOUT == 60.0

    def test_read_timeout_is_float(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert isinstance(cfg.SATELLITE_READ_TIMEOUT, float)


class TestStaticConfig:
    def test_asr_port(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.ASR_PORT == 10300

    def test_tts_port(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.TTS_PORT == 10200

    def test_bind_host(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.BIND_HOST == "0.0.0.0"

    def test_piper_model_path_default_is_onnx(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.PIPER_MODEL_PATH.endswith(".onnx")

    def test_piper_model_path_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"PIPER_MODEL_PATH": "/data/models/en.onnx"})
        assert cfg.PIPER_MODEL_PATH == "/data/models/en.onnx"
