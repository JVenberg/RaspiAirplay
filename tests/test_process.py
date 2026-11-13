"""Runs the whole program as a subprocess to check signals, exit codes and child cleanup."""

import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
ROOT = TESTS.parent
CLI_FIELDS = {"gain", "identifier", "device", "threshold", "silence_seconds"}


class AppProcess:
    def __init__(self, spec_path, args=(), env=None):
        self.proc = subprocess.Popen(
            [sys.executable, str(TESTS / "run_app.py"), str(spec_path), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{TESTS}", **(env or {})},
        )
        self.lines = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)

    @property
    def output(self):
        return "".join(self.lines)

    def wait_for(self, text, timeout=10):
        deadline = time.monotonic() + timeout
        while text not in self.output:
            if time.monotonic() > deadline or self.proc.poll() is not None:
                time.sleep(0.2)
                raise AssertionError(f"{text!r} not seen; output:\n{self.output}")
            time.sleep(0.02)

    def wait(self, timeout=10):
        code = self.proc.wait(timeout)
        self.reader.join(5)
        return code


def process_state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return None


def wait_dead(pids, timeout=5):
    deadline = time.monotonic() + timeout
    while any(process_state(pid) not in (None, "Z") for pid in pids):
        if time.monotonic() > deadline:
            raise AssertionError({pid: process_state(pid) for pid in pids})
        time.sleep(0.05)


@pytest.fixture
def launch(tmp_path, make_config):
    procs = []

    def start(config=None, args=(), env=None, **spec):
        values = dataclasses.asdict(make_config(**(config or {})))
        spec["config"] = {k: v for k, v in values.items() if k not in CLI_FIELDS}
        path = tmp_path / f"spec{len(procs)}.json"
        path.write_text(json.dumps(spec))
        env = {"RASPI_AIRPLAY_THRESHOLD": "100", "RASPI_AIRPLAY_SILENCE_SECONDS": "0.6", **(env or {})}
        procs.append(AppProcess(path, args, env))
        return procs[-1]

    yield start
    for app in procs:
        if app.proc.poll() is None:
            app.proc.kill()
            app.wait()


@pytest.mark.parametrize("state", ["idle", "streaming", "connecting"])
def test_sigterm_exits_cleanly_and_reaps_recorder(launch, recorder, state):
    receiver = {"connect": "hang"} if state == "connecting" else {}
    recorder.set(mode="silence" if state == "idle" else "ramp")
    app = launch(receiver=receiver)
    app.wait_for("capture started")
    if state != "idle":
        app.wait_for("connecting to Fake")
    if state == "streaming":
        app.wait_for("streaming started")
    time.sleep(0.3)
    app.proc.send_signal(signal.SIGTERM)
    assert app.wait() == 0, app.output
    assert "shutting down" in app.output
    assert "Traceback" not in app.output
    assert all(process_state(int(pid)) is None for pid in recorder.starts)


def test_cli_options_and_env_vars_reach_config(launch, recorder):
    app = launch(args=["--log-level", "debug", "--device", "hw:9"], env={"RASPI_AIRPLAY_GAIN": "1.5"})
    app.wait_for("starting with")
    for expected in ("gain=1.5", "device='hw:9'", "threshold=100.0", "silence_seconds=0.6"):
        assert expected in app.output
    app.proc.send_signal(signal.SIGINT)
    assert app.wait() == 0


def test_blocked_event_loop_is_killed_by_watchdog(launch, recorder):
    recorder.set(mode="ramp")
    app = launch(config={"watchdog_seconds": 0.5}, block_loop_after=1.0)
    started = time.monotonic()
    assert app.wait() == 3
    assert time.monotonic() - started < 5
    assert "event loop unresponsive" in app.output
    assert "in run_until_complete" in app.output
    wait_dead([int(pid) for pid in recorder.starts])


def test_main_loop_without_progress_exits(launch, recorder):
    app = launch(config={"stuck_seconds": 0.5, "watchdog_seconds": 2.0}, hang_main=True)
    assert app.wait() == 3
    assert "made no progress" in app.output
    assert "hang" in app.output
    wait_dead([int(pid) for pid in recorder.starts])


def test_unstoppable_stream_exits_for_restart(launch, recorder):
    recorder.set(mode="ramp")
    app = launch(receiver={"stream": "zombie"})
    assert app.wait(timeout=15) == 1
    assert "fatal:" in app.output and "did not stop" in app.output
    wait_dead([int(pid) for pid in recorder.starts])
