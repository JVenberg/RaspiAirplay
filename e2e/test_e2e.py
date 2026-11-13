"""End to end against a RAOP receiver in Docker: real pyatv, zeroconf, RTSP and RTP. Run on the host."""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(not shutil.which("docker"), reason="needs docker")]

ROOT = Path(__file__).parent.parent
APP_IMAGE = os.environ.get("E2E_APP_IMAGE", "airplay:test")
APP_CONFIG = {
    "start_window_seconds": 0.5,
    "preroll_seconds": 0.5,
    "scan_timeout_seconds": 3.0,
    "scan_grace_seconds": 5.0,
    "connect_timeout_seconds": 10.0,
    "setup_timeout_seconds": 10.0,
    "stream_stall_seconds": 5.0,
    "stop_timeout_seconds": 6.0,
    "min_backoff_seconds": 0.5,
    "max_backoff_seconds": 4.0,
    "healthy_session_seconds": 10.0,
    "status_interval_seconds": 2.0,
}


def docker(*args, check=True):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout.strip()


class Logs:
    def __init__(self, container):
        self.lines = []
        self.proc = subprocess.Popen(
            ["docker", "logs", "-f", container], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)

    @property
    def text(self):
        return "".join(self.lines)

    def count(self, text):
        return self.text.count(text)

    def wait_for(self, text, count=1, timeout=30):
        deadline = time.monotonic() + timeout
        while self.count(text) < count:
            if time.monotonic() > deadline:
                raise AssertionError(f"{text!r} x{count} not seen in {timeout}s; logs:\n{self.text[-6000:]}")
            time.sleep(0.05)

    def statuses(self):
        pattern = re.compile(r"status: .*")
        rows = []
        for match in pattern.finditer(self.text):
            fields = dict(item.split("=", 1) for item in match.group(0).split()[1:] if "=" in item)
            rows.append(fields)
        return rows


class E2E:
    def __init__(self, workdir):
        self.id = uuid.uuid4().hex[:8]
        self.workdir = workdir
        self.network = f"airplay-e2e-{self.id}"
        self.subnet = f"172.31.{int(self.id[:2], 16)}"
        self.receiver = f"airplay-e2e-receiver-{self.id}"
        self.app = f"airplay-e2e-app-{self.id}"
        self.receiver_logs = self.app_logs = None
        self.set_capture(mode="silence")
        docker("network", "create", "--subnet", f"{self.subnet}.0/24", self.network)

    def set_capture(self, **control):
        tmp = self.workdir / "control.tmp"
        tmp.write_text(json.dumps(control))
        tmp.replace(self.workdir / "control.json")

    def start_receiver(self, ip_suffix=10):
        docker("rm", "-f", self.receiver, check=False)
        docker(
            "run", "-d", "--name", self.receiver, "--network", self.network, "--ip", f"{self.subnet}.{ip_suffix}",
            "-v", f"{ROOT}:/app:ro", "-v", f"{self.workdir}:/out", APP_IMAGE,
            "python", "/app/e2e/raop_receiver.py", "--out", "/out", "--idle-timeout", "5",
        )  # fmt: skip
        self.receiver_logs = Logs(self.receiver)
        self.receiver_logs.wait_for("receiver: ready", timeout=30)

    def start_app(self, **config):
        spec = {
            "real_receiver": True,
            "config": {
                **APP_CONFIG,
                **config,
                "capture_cmd": ["python", "/app/tests/fake_rec.py", "/work/control.json"],
                "heartbeat_path": "/tmp/heartbeat",
            },
        }
        (self.workdir / "spec.json").write_text(json.dumps(spec))
        docker(
            "run", "-d", "--name", self.app, "--network", self.network, "--init",
            "-e", "PYTHONPATH=/app:/app/tests", "-v", f"{ROOT}:/app:ro", "-v", f"{self.workdir}:/work",
            APP_IMAGE, "python", "/app/tests/run_app.py", "/work/spec.json",
            "--threshold", "100", "--silence-seconds", "1.5", "--identifier", "AA:BB:CC:DD:EE:FF",
        )  # fmt: skip
        self.app_logs = Logs(self.app)
        self.app_logs.wait_for("capture started")

    def audio(self):
        data = np.fromfile(self.workdir / "audio.raw", dtype="<i2")
        return data[: len(data) // 2 * 2].reshape(-1, 2)

    def close(self):
        if self.app_logs:
            print(f"\n--- app logs ---\n{self.app_logs.text[-8000:]}")
        if self.receiver_logs:
            print(f"\n--- receiver logs ---\n{self.receiver_logs.text[-4000:]}")
        for name in (self.app, self.receiver):
            docker("rm", "-f", name, check=False)
        docker("network", "rm", self.network, check=False)


@pytest.fixture
def e2e(tmp_path):
    env = E2E(tmp_path)
    try:
        yield env
    finally:
        env.close()


def analyse(frames):
    """Every frame must be silence or the fake recorder's ramp; returns ramp runs and in-run breaks."""
    ramp = frames[:, 1] == ~frames[:, 0]
    silent = (frames == 0).all(axis=1)
    index = (frames[:, 0].astype(np.int64) + 32768) % 65536
    both = ramp[1:] & ramp[:-1]
    breaks = int(np.sum(both & ((index[1:] - index[:-1]) % 65536 != 1)))
    starts = int(np.sum(ramp[1:] & ~ramp[:-1]) + ramp[0]) if len(frames) else 0
    return {
        "frames": len(frames),
        "ramp": int(ramp.sum()),
        "silent": int(silent.sum()),
        "other": int(np.sum(~ramp & ~silent)),
        "runs": starts,
        "breaks": breaks,
    }


def stream_once(e2e, seconds, count):
    e2e.set_capture(mode="ramp")
    e2e.app_logs.wait_for("streaming started", count=count, timeout=60)
    time.sleep(seconds)
    e2e.set_capture(mode="silence")
    e2e.app_logs.wait_for("stream ended: silence", count=count, timeout=30)


def test_audio_arrives_bit_exact(e2e):
    e2e.start_receiver()
    e2e.start_app()
    stream_once(e2e, 6, 1)
    time.sleep(1)
    result = analyse(e2e.audio())
    print(result)
    assert result["other"] == 0
    assert result["breaks"] == 0
    assert result["runs"] == 1
    assert result["ramp"] > 44100 * 6
    assert "resetting on progress" not in e2e.receiver_logs.text


@pytest.mark.parametrize("downtime", [0, 30])
def test_recovers_when_receiver_dies(e2e, downtime):
    e2e.start_receiver()
    e2e.start_app()
    e2e.set_capture(mode="ramp")
    e2e.app_logs.wait_for("streaming started")
    time.sleep(2)
    docker("kill", e2e.receiver)
    killed = time.monotonic()
    if downtime:
        e2e.app_logs.wait_for("stream ended: receiver", timeout=downtime)
        print(f"\ndead receiver detected after {time.monotonic() - killed:.1f}s")
        time.sleep(killed + downtime - time.monotonic())
    docker("start", e2e.receiver)
    e2e.receiver_logs = Logs(e2e.receiver)
    e2e.app_logs.wait_for("stream ended:", count=1, timeout=30)
    e2e.app_logs.wait_for("streaming started", count=2, timeout=90)
    time.sleep(3)
    assert "Traceback" not in e2e.app_logs.text


@pytest.mark.parametrize("seconds", [3, 20])
def test_recovers_from_frozen_receiver(e2e, seconds):
    e2e.start_receiver()
    e2e.start_app()
    e2e.set_capture(mode="ramp")
    e2e.app_logs.wait_for("streaming started")
    time.sleep(2)
    docker("pause", e2e.receiver)
    time.sleep(seconds)
    docker("unpause", e2e.receiver)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        before = len(e2e.audio())
        time.sleep(2)
        if len(e2e.audio()) - before > 44100:
            break
    else:
        raise AssertionError("audio never resumed")
    print(e2e.app_logs.text[-3000:])


def test_recovers_from_network_loss(e2e):
    e2e.start_receiver()
    e2e.start_app()
    e2e.set_capture(mode="ramp")
    e2e.app_logs.wait_for("streaming started")
    time.sleep(2)
    docker("network", "disconnect", e2e.network, e2e.receiver)
    lost = time.monotonic()
    e2e.app_logs.wait_for("stream ended: receiver closed connection", timeout=30)
    print(f"\nnetwork loss detected after {time.monotonic() - lost:.1f}s")
    time.sleep(5)
    docker("network", "connect", "--ip", f"{e2e.subnet}.10", e2e.network, e2e.receiver)
    e2e.app_logs.wait_for("streaming started", count=2, timeout=120)


def test_finds_receiver_after_ip_change(e2e):
    e2e.start_receiver(ip_suffix=10)
    e2e.start_app()
    stream_once(e2e, 2, 1)
    e2e.start_receiver(ip_suffix=20)
    stream_once(e2e, 2, 2)
    assert f"{e2e.subnet}.20" in e2e.app_logs.text


def test_many_sessions_do_not_leak(e2e):
    sessions = int(os.environ.get("E2E_SESSIONS", "30"))
    e2e.start_receiver()
    e2e.start_app()
    for i in range(1, sessions + 1):
        stream_once(e2e, 1.5, i)
    time.sleep(3)
    rows = e2e.app_logs.statuses()
    first, last = rows[len(rows) // 5], rows[-1]
    print(f"\nsessions={last['sessions']} first={first}\nlast={last}")
    assert int(last["sessions"]) == sessions and last["failures"] == "0"
    assert float(last["rss_mb"]) - float(first["rss_mb"]) < 8
    assert int(last["fds"]) <= int(first["fds"]) + 2
    assert int(last["threads"]) <= int(first["threads"]) + 1
    assert int(last["tasks"]) <= int(first["tasks"]) + 1
    result = analyse(e2e.audio())
    print(result)
    assert result["other"] == 0 and result["breaks"] == 0
