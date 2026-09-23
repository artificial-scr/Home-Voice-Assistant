"""Unit tests for PCM conversion helpers in brain/asr_whisper.py."""

import struct

import numpy as np
import pytest

from asr_whisper import _EXPECTED_WIDTH, _pcm_to_float


class TestPcmToFloat:
    def test_zero_samples_produce_zero_floats(self):
        raw = b"\x00\x00" * 100
        result = _pcm_to_float(raw)
        assert np.all(result == 0.0)

    def test_output_length_matches_sample_count(self):
        n = 64
        raw = b"\x00\x00" * n
        result = _pcm_to_float(raw)
        assert len(result) == n

    def test_dtype_is_float32(self):
        raw = b"\x00\x00" * 10
        result = _pcm_to_float(raw)
        assert result.dtype == np.float32

    def test_max_positive_int16_near_one(self):
        raw = struct.pack("<1h", 32767)
        result = _pcm_to_float(raw)
        assert result[0] == pytest.approx(32767.0 / 32768.0, abs=1e-5)

    def test_min_negative_int16_is_minus_one(self):
        raw = struct.pack("<1h", -32768)
        result = _pcm_to_float(raw)
        assert result[0] == pytest.approx(-1.0, abs=1e-5)

    def test_values_within_range(self):
        # All possible int16 values should map to [-1, 1]
        samples = list(range(-32768, 32768, 256))
        raw = struct.pack(f"<{len(samples)}h", *samples)
        result = _pcm_to_float(raw)
        assert float(result.min()) >= -1.0
        assert float(result.max()) <= 1.0

    def test_known_values(self):
        raw = struct.pack("<4h", 0, 16384, -16384, 32767)
        result = _pcm_to_float(raw)
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(16384.0 / 32768.0, abs=1e-4)
        assert result[2] == pytest.approx(-16384.0 / 32768.0, abs=1e-4)

    def test_expected_width_constant(self):
        assert _EXPECTED_WIDTH == 2  # 16-bit PCM = 2 bytes per sample

    def test_even_truncation_safety(self):
        # After the & ~1 truncation in handle_event, _pcm_to_float always
        # receives an even-length buffer.  Verify no struct error on exact multiples.
        for n in [1, 2, 4, 100, 1024]:
            raw = b"\x00\x00" * n
            result = _pcm_to_float(raw)
            assert len(result) == n
