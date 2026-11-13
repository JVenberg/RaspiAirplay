"""Randomized fault injection for SOAK_SECONDS; checks recovery after every fault and resource growth."""

import asyncio
import collections
import gc
import logging
import os
import random
import threading
import time
import tracemalloc
from pathlib import Path

import pytest
from fakes import App

import stream

pytestmark = pytest.mark.soak

SOAK_SECONDS = float(os.environ.get("SOAK_SECONDS", "120"))
HEALTHY = {"scan": "found", "connect": "ok", "stream": "play", "close": "ok"}
RECEIVER_FAULTS = [
    ("scan", "none"),
    ("scan", "hang"),
    ("scan", "raise"),
    ("connect", "hang"),
    ("connect", "raise"),
    ("stream", "hang_setup"),
    ("stream", "stall"),
    ("stream", "raise"),
    ("stream", "end"),
    ("stream", "disconnect"),
    ("close", "hang"),
    ("close", "raise"),
]
CAPTURE_FAULTS = [
    {"mode": "exit", "code": 1},
    {"mode": "stall"},
    {"mode": "partial"},
    {"mode": "ramp", "rate": 4.0},
    {"mode": "ramp", "rate": 0.8},
    {"mode": "noise", "amplitude": 20000},
    {"mode": "silence"},
]


def children():
    pid = str(os.getpid())
    states = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[1] == pid:
            states.append(fields[0])
    return states


def resources():
    gc.collect()
    status = Path("/proc/self/status").read_text()
    rss = next(line for line in status.splitlines() if line.startswith("VmRSS"))
    kids = children()
    return {
        "rss_mb": int(rss.split()[1]) / 1024,
        "fds": len(os.listdir("/proc/self/fd")),
        "threads": threading.active_count(),
        "tasks": len(asyncio.all_tasks()),
        "children": sum(state != "Z" for state in kids),
        "zombies": kids.count("Z"),
    }


class LogTail(logging.Handler):
    def __init__(self):
        super().__init__()
        self.tail = collections.deque(maxlen=300)
        self.counts = collections.Counter()

    def emit(self, record):
        self.counts[record.levelname] += 1
        self.tail.append(f"{record.levelname} {record.getMessage()}")


async def wait_streaming(receiver, log, timeout=15.0):
    """A session is up, has sent at least a second of real audio and is still being read."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        source = getattr(receiver.sessions[-1], "source", None) if receiver.sessions else None
        if (
            receiver.active == 1
            and source is not None
            and source.frames_sent > 44100
            and time.monotonic() - source.last_read < 0.1
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("did not recover:\n" + "\n".join(log.tail))


@pytest.mark.timeout(SOAK_SECONDS + 300)
async def test_random_faults_never_wedge_or_leak(make_config, recorder, receiver, monkeypatch):
    seed = int(os.environ.get("SOAK_SEED", time.time_ns() % 2**32))
    rng = random.Random(seed)
    log = LogTail()
    monkeypatch.setattr(stream.logger, "propagate", False)
    monkeypatch.setattr(stream.logger, "handlers", [log])
    receiver.keep_frames = False
    config = make_config(
        scan_grace_seconds=0.2,
        capture_max_backoff_seconds=0.2,
        max_backlog_seconds=1.0,
        buffer_seconds=5.0,
    )
    tracemalloc.start(10)
    ignore = [tracemalloc.Filter(False, tracemalloc.__file__), tracemalloc.Filter(False, "*/tests/*")]
    async with App(config) as app:
        recorder.set(mode="ramp")
        await wait_streaming(receiver, log)
        await asyncio.sleep(3)
        before = resources()
        snapshot = tracemalloc.take_snapshot().filter_traces(ignore)
        cpu = time.process_time()
        faults = collections.Counter()
        deadline = time.monotonic() + SOAK_SECONDS
        while time.monotonic() < deadline:
            roll = rng.random()
            if roll < 0.45:
                attr, behaviour = rng.choice(RECEIVER_FAULTS)
                setattr(receiver, attr, behaviour)
                faults[f"{attr}:{behaviour}"] += 1
            elif roll < 0.85:
                control = rng.choice(CAPTURE_FAULTS)
                recorder.set(**control)
                faults[f"capture:{control['mode']}"] += 1
            else:
                faults["none"] += 1
            await asyncio.sleep(rng.uniform(0.1, 2.5))
            for attr, behaviour in HEALTHY.items():
                setattr(receiver, attr, behaviour)
            recorder.set(mode="ramp")
            await wait_streaming(receiver, log)
            app.check()
            del receiver.sessions[:-1]
            receiver.scans.clear()
            receiver.scan_times.clear()
        await asyncio.sleep(3)
        after = resources()
        growth = tracemalloc.take_snapshot().filter_traces(ignore).compare_to(snapshot, "lineno")
        tracemalloc.stop()
        cpu = time.process_time() - cpu
    grew = sum(stat.size_diff for stat in growth)
    print(f"\nseed={seed} faults={sum(faults.values())} sessions={app.stats.sessions} cpu={cpu:.1f}s")
    print(f"log levels: {dict(log.counts)}")
    print(f"faults: {dict(faults)}")
    print(f"before: {before}\nafter:  {after}\ntraced growth: {grew / 1024:.1f} KiB")
    for stat in growth[:8]:
        print(f"  {stat}")
    assert grew < 256 * 1024
    assert after["rss_mb"] - before["rss_mb"] < 8
    assert after["fds"] <= before["fds"] + 1
    assert after["threads"] <= before["threads"] + 1
    assert after["tasks"] <= before["tasks"]
    assert after["children"] == 1 and after["zombies"] == 0
