"""
Shared test configuration.

Heavy dependencies (wyoming, faster_whisper, piper, sounddevice, openai) are
stubbed out at the sys.modules level so the brain/satellite modules can be
imported without the full ML stack installed.  This must run at module level
(before any test file imports) — pytest loads conftest.py first.
"""

import sys
from unittest.mock import MagicMock

_SIMPLE_STUBS = [
    # ML / hardware dependencies only — wyoming is the real library
    "faster_whisper",
    "piper",
    "sounddevice",
    "openai",
    "openai.types",
    "openai.types.chat",
]

for _mod in _SIMPLE_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
