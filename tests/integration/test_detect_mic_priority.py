"""
Integration tests for find_mic() device selection in satellite/detect_mic.py.

sounddevice is monkeypatched to return a controlled device list, so no
hardware is required.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import detect_mic
from detect_mic import find_mic


def _make_device(name: str, max_input_channels: int = 1):
    return {"name": name, "max_input_channels": max_input_channels}


def _patch_sd(devices, default_input=0):
    sd_mock = MagicMock()
    sd_mock.query_devices.return_value = devices
    sd_mock.default.device = (default_input, 0)  # (input, output)
    return patch.dict(sys.modules, {"sounddevice": sd_mock}), sd_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_respeaker_wins_over_usb():
    devices = [
        _make_device("USB Audio Device"),
        _make_device("ReSpeaker 2-Mics Pi HAT"),
    ]
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = devices
    mock_sd.default.device = (0, 0)

    with patch.object(detect_mic, "sd", mock_sd):
        idx = find_mic()

    assert idx == 1  # ReSpeaker is at index 1


def test_usb_wins_over_generic():
    devices = [
        _make_device("Generic Built-in Mic"),
        _make_device("USB Microphone"),
    ]
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = devices
    mock_sd.default.device = (0, 0)

    with patch.object(detect_mic, "sd", mock_sd):
        idx = find_mic()

    assert idx == 1  # USB at index 1


def test_hdmi_only_falls_back_to_default():
    devices = [
        _make_device("HDMI Audio Output"),
        _make_device("vc4-hdmi"),
    ]
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = devices
    mock_sd.default.device = (0, 0)

    with patch.object(detect_mic, "sd", mock_sd):
        idx = find_mic()

    assert idx == 0  # fallback to default_input


def test_no_input_devices_returns_minus_one():
    devices = [
        _make_device("Speaker Out", max_input_channels=0),
        _make_device("HDMI Out", max_input_channels=0),
    ]
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = devices
    mock_sd.default.device = (0, 0)

    with patch.object(detect_mic, "sd", mock_sd):
        idx = find_mic()

    assert idx == -1


def test_first_generic_wins_when_no_preferred():
    devices = [
        _make_device("Generic Mic A"),
        _make_device("Generic Mic B"),
    ]
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = devices
    mock_sd.default.device = (0, 0)

    with patch.object(detect_mic, "sd", mock_sd):
        idx = find_mic()

    # Both score 1; max() returns first max found, which is index 0
    assert idx == 0


def test_seeed_wins_over_usb():
    devices = [
        _make_device("USB Audio Device"),
        _make_device("seeed-voicecard"),
    ]
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = devices
    mock_sd.default.device = (0, 0)

    with patch.object(detect_mic, "sd", mock_sd):
        idx = find_mic()

    assert idx == 1
