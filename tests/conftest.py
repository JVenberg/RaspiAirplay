import pytest
from fakes import FakeReceiver, Recorder

import stream


@pytest.fixture
def recorder(tmp_path):
    return Recorder(tmp_path / "control.json")


@pytest.fixture
def make_config(tmp_path, recorder):
    def make(**overrides):
        values = dict(
            capture_cmd=recorder.cmd,
            threshold=100.0,
            silence_seconds=0.6,
            start_window_seconds=0.2,
            preroll_seconds=0.2,
            buffer_seconds=5.0,
            max_backlog_seconds=2.0,
            capture_stall_seconds=1.0,
            underrun_timeout_seconds=1.5,
            scan_timeout_seconds=0.3,
            connect_timeout_seconds=0.5,
            setup_timeout_seconds=0.5,
            stream_stall_seconds=0.5,
            stop_timeout_seconds=0.9,
            min_backoff_seconds=0.05,
            max_backoff_seconds=0.2,
            healthy_session_seconds=1.5,
            poll_seconds=0.02,
            status_interval_seconds=0.5,
            heartbeat_path=str(tmp_path / "heartbeat"),
            watchdog_seconds=5.0,
            stuck_seconds=30.0,
        )
        values.update(overrides)
        return stream.Config(**values)

    return make


@pytest.fixture
def receiver(monkeypatch):
    fake = FakeReceiver()
    fake.install(monkeypatch)
    return fake
