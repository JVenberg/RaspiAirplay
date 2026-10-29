"""Stand-in for sox's rec: paced s16le stereo PCM on stdout, steered by a JSON control file.

Control keys: mode (tone|silence|noise|ramp|stall|exit|partial), amplitude, freq,
rate (pacing multiplier, 0 for unpaced), code (exit status), stderr (line per chunk).
Every generated frame advances a counter so "ramp" output can be checked for continuity.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RATE = 44100
CHUNK = 441


def ramp(start, n):
    i = np.arange(start, start + n, dtype=np.int64)
    left = (i % 65536 - 32768).astype(np.int16)
    return np.stack([left, ~left], axis=1)


def main():
    control_path = Path(sys.argv[1])
    with open(f"{control_path}.starts", "a") as f:
        f.write(f"{os.getpid()}\n")
    rng = np.random.default_rng()
    control = {}
    frame = 0
    started = time.monotonic()
    paced_frames = 0
    out = sys.stdout.buffer
    while True:
        try:
            control = json.loads(control_path.read_text())
        except (OSError, ValueError):
            pass
        mode = control.get("mode", "silence")
        amplitude = control.get("amplitude", 8000)
        if control.get("stderr"):
            print(control["stderr"], file=sys.stderr, flush=True)
        if mode == "exit":
            sys.exit(control.get("code", 1))
        if mode == "stall":
            time.sleep(0.05)
            started, paced_frames = time.monotonic(), 0
            continue
        if mode == "partial":
            out.write(b"\x00" * 6)
            out.flush()
            sys.exit(0)
        if mode == "tone":
            t = np.arange(frame, frame + CHUNK) / RATE
            wave = (amplitude * np.sin(2 * np.pi * control.get("freq", 440.0) * t)).astype(np.int16)
            samples = np.stack([wave, wave], axis=1)
        elif mode == "noise":
            samples = rng.normal(0, amplitude, (CHUNK, 2)).clip(-32768, 32767).astype(np.int16)
        elif mode == "ramp":
            samples = ramp(frame, CHUNK)
        else:
            samples = np.zeros((CHUNK, 2), dtype=np.int16)
        try:
            out.write(samples.astype("<i2").tobytes())
            out.flush()
        except BrokenPipeError:
            sys.exit(0)
        frame += CHUNK
        paced_frames += CHUNK
        rate = control.get("rate", 1.0)
        if rate > 0:
            delay = started + paced_frames / (RATE * rate) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        else:
            started, paced_frames = time.monotonic(), 0


if __name__ == "__main__":
    main()
