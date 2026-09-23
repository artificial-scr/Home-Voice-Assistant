"""Unit tests for mic scoring heuristics in satellite/detect_mic.py."""

import pytest
from detect_mic import _PREFER, _SKIP_KEYWORDS, _score


class TestScore:
    # --- Priority ordering ---

    def test_respeaker_beats_usb(self):
        assert _score("ReSpeaker 2-Mics Pi HAT") > _score("USB Audio Device")

    def test_usb_beats_generic(self):
        assert _score("USB Audio Device") > _score("Generic Input Device")

    def test_seeed_same_rank_as_respeaker(self):
        assert _score("seeed-voicecard") == _score("ReSpeaker HAT")

    def test_respeaker_beats_generic(self):
        assert _score("ReSpeaker HAT") > _score("Some Generic Mic")

    # --- Deprioritised keywords return 0 ---

    def test_hdmi_is_zero(self):
        assert _score("HDMI Output") == 0

    def test_vc4_is_zero(self):
        assert _score("vc4-hdmi") == 0

    def test_bcm_is_zero(self):
        assert _score("bcm2835 ALSA") == 0

    def test_dummy_is_zero(self):
        assert _score("snd_dummy") == 0

    def test_null_is_zero(self):
        assert _score("null sink") == 0

    def test_loop_is_zero(self):
        assert _score("loopback device") == 0

    # --- Generic input devices get positive score ---

    def test_generic_input_positive(self):
        assert _score("Built-in Microphone") > 0

    def test_empty_name_positive(self):
        # An unnamed device is generic, not skip-listed
        assert _score("") > 0

    # --- Case insensitivity ---

    def test_respeaker_lowercase_still_matches(self):
        assert _score("respeaker hat") == _score("ReSpeaker HAT")

    def test_usb_uppercase_still_matches(self):
        assert _score("USB MICROPHONE") == _score("usb microphone")

    def test_hdmi_uppercase_still_zero(self):
        assert _score("HDMI AUDIO") == 0

    # --- Constants sanity ---

    def test_prefer_list_not_empty(self):
        assert len(_PREFER) >= 2

    def test_skip_keywords_not_empty(self):
        assert len(_SKIP_KEYWORDS) >= 1
