"""
Detect the best available microphone device.

Priority order:
  1. ReSpeaker HAT / seeed devices (common Pi HAT)
  2. Any USB audio input device
  3. First device with input channels (excluding HDMI / vc4)
  4. sounddevice default input

Usage (standalone — prints device index):
    python detect_mic.py

Usage (verbose — show all candidates):
    python detect_mic.py --list

The printed index can be passed directly to wyoming-satellite's --mic-device.
"""

import argparse
import sys

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
    sys.exit(1)


# Keyword heuristics — highest priority first
_PREFER = [
    ("respeaker", "seeed"),        # ReSpeaker HAT
    ("usb",),                      # generic USB mic
]
# Keywords that indicate non-mic inputs to deprioritise
_SKIP_KEYWORDS = ("hdmi", "vc4", "bcm", "dummy", "null", "loop")


def _score(name: str) -> int:
    """Higher = better candidate. 0 = deprioritised."""
    lower = name.lower()
    if any(kw in lower for kw in _SKIP_KEYWORDS):
        return 0
    for rank, keywords in enumerate(reversed(_PREFER), start=1):
        if any(kw in lower for kw in keywords):
            return rank + 1
    return 1  # generic input device


def find_mic(verbose: bool = False) -> int:
    """Return the sounddevice index of the best mic, or -1 for the default."""
    devices = sd.query_devices()
    default_input = sd.default.device[0]

    candidates = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        score = _score(dev["name"])
        candidates.append((score, idx, dev["name"]))

    if verbose:
        print("Available input devices:")
        for score, idx, name in sorted(candidates, key=lambda x: -x[0]):
            marker = " ← default" if idx == default_input else ""
            print(f"  [{idx:2d}] score={score}  {name}{marker}")
        print()

    if not candidates:
        print("No input devices found — falling back to system default.", file=sys.stderr)
        return -1

    best_score, best_idx, best_name = max(candidates, key=lambda x: x[0])

    if best_score == 0:
        # All candidates are deprioritised; fall back to default
        best_idx = default_input
        best_name = devices[default_input]["name"] if default_input >= 0 else "default"

    if verbose:
        print(f"Selected: [{best_idx}] {best_name}")

    return best_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect best microphone device index.")
    parser.add_argument("--list", action="store_true", help="Show all candidates, then print selected index.")
    args = parser.parse_args()

    idx = find_mic(verbose=args.list)
    # Print just the index on stdout so shell scripts can capture it cleanly
    print(idx)
