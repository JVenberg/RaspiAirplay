import asyncio
import ipaddress
import socket
import time

import numpy as np
import pytest
from fakes import App, decode_frames, ramp_index, wait_until

import stream


def assert_contiguous(session):
    frames = decode_frames(session.frames)
    ramp = frames[frames[:, 1] == ~frames[:, 0]]
    assert len(ramp) > 0
    assert np.all(np.diff(ramp_index(ramp)) % 65536 == 1)
    return len(ramp)


async def test_streams_on_sound_and_stops_on_silence(make_config, recorder, receiver, caplog):
    async with App(make_config()) as app:
        await asyncio.sleep(0.5)
        assert receiver.connects == 0
        recorder.set(mode="ramp")
        await wait_until(lambda: receiver.active == 1)
        await asyncio.sleep(1.5)
        recorder.set(mode="silence")
        await wait_until(lambda: receiver.active == 0 and receiver.closes == 1)
        app.check()
        assert app.stats.sessions == 1 and app.stats.failures == 0
    sent = assert_contiguous(receiver.sessions[0])
    assert sent > 44100 * 1.3
    assert "stream ended: silence" in caplog.text


async def test_quiet_noise_below_threshold_does_not_stream(make_config, recorder, receiver):
    recorder.set(mode="noise", amplitude=40)
    async with App(make_config()) as app:
        await asyncio.sleep(1.0)
        assert app.capture.meter.rms(5) == pytest.approx(40, rel=0.2)
        assert receiver.connects == 0


async def test_restarts_stream_while_sound_continues(make_config, recorder, receiver, caplog):
    receiver.stream = "end"
    recorder.set(mode="ramp")
    async with App(make_config()) as app:
        await wait_until(lambda: receiver.connects >= 3)
        receiver.stream = "play"
        await wait_until(lambda: receiver.active == 1)
        app.check()
    assert "receiver ended stream" in caplog.text


@pytest.mark.parametrize(
    ("attr", "behaviour", "message"),
    [
        ("scan", "none", "DeviceNotFound"),
        ("scan", "hang", "TimeoutError"),
        ("scan", "raise", "OSError: network unreachable"),
        ("connect", "hang", "TimeoutError"),
        ("connect", "raise", "ConnectionRefusedError"),
        ("stream", "hang_setup", "stream setup timed out"),
        ("stream", "stall", "stream stalled"),
        ("stream", "raise", "RuntimeError: protocol error"),
        ("stream", "disconnect", "receiver closed connection"),
    ],
)
async def test_recovers_from_receiver_fault(make_config, recorder, receiver, caplog, attr, behaviour, message):
    setattr(receiver, attr, behaviour)
    recorder.set(mode="ramp")
    async with App(make_config(scan_grace_seconds=0.2)) as app:
        await wait_until(lambda: message in caplog.text, timeout=10)
        await asyncio.sleep(0.5)
        setattr(receiver, attr, {"scan": "found", "connect": "ok", "stream": "play"}[attr])
        start = receiver.connects
        await wait_until(lambda: receiver.connects > start and receiver.active == 1, timeout=10)
        await asyncio.sleep(0.5)
        assert receiver.active == 1
        app.check()
    assert receiver.active == 0


@pytest.mark.parametrize("behaviour", ["hang", "raise"])
async def test_recovers_from_connection_close_fault(make_config, recorder, receiver, caplog, behaviour):
    receiver.close = behaviour
    recorder.set(mode="ramp")
    async with App(make_config()) as app:
        await wait_until(lambda: receiver.active == 1)
        recorder.set(mode="silence")
        await wait_until(lambda: "stream ended: silence" in caplog.text, timeout=10)
        recorder.set(mode="ramp")
        await wait_until(lambda: len(receiver.sessions) == 2 and receiver.active == 1, timeout=10)
        app.check()


async def test_failures_back_off_exponentially(make_config, recorder, receiver):
    receiver.scan = "none"
    recorder.set(mode="ramp")
    config = make_config(min_backoff_seconds=0.1, max_backoff_seconds=0.4)
    async with App(config) as app:
        await asyncio.sleep(2.0)
        app.check()
    gaps = np.diff(receiver.scan_times)
    assert len(receiver.scans) <= 8
    assert gaps[-1] >= 0.4


async def test_rediscovers_receiver_after_ip_change(make_config, recorder, receiver):
    recorder.set(mode="ramp")
    async with App(make_config()) as app:
        await wait_until(lambda: receiver.active == 1)
        recorder.set(mode="silence")
        await wait_until(lambda: receiver.active == 0)
        assert app.streamer.last_address == "10.0.0.5"
        receiver.address = ipaddress.ip_address("10.0.0.9")
        receiver.scans.clear()
        recorder.set(mode="ramp")
        await wait_until(lambda: receiver.active == 1)
        assert receiver.scans[:2] == [["10.0.0.5"], None]
        assert app.streamer.last_address == "10.0.0.9"


async def test_capture_stall_ends_stream_then_recovers(make_config, recorder, receiver, caplog):
    recorder.set(mode="ramp")
    async with App(make_config()) as app:
        await wait_until(lambda: receiver.active == 1)
        recorder.set(mode="stall")
        await wait_until(lambda: "stream ended: capture lost" in caplog.text, timeout=10)
        assert receiver.active == 0
        recorder.set(mode="ramp")
        await wait_until(lambda: receiver.active == 1, timeout=10)
        app.check()
    assert app.capture.restarts >= 1


async def test_stream_rides_through_capture_restart(make_config, recorder, receiver):
    recorder.set(mode="ramp")
    async with App(make_config()) as app:
        await wait_until(lambda: receiver.active == 1)
        await asyncio.sleep(0.5)
        recorder.set(mode="exit", code=1)
        await wait_until(lambda: app.capture.restarts >= 1, timeout=10)
        recorder.set(mode="ramp")
        await asyncio.sleep(2.0)
        source = receiver.sessions[0].source
        assert receiver.active == 1 and len(receiver.sessions) == 1
        assert source.underruns == 1
        assert app.capture.ring.end - source.pos >= source.preroll - 4410
        app.check()


async def test_enables_dead_receiver_detection(make_config, recorder, receiver):
    recorder.set(mode="ramp")
    async with App(make_config(receiver_timeout_seconds=10)) as app:
        await wait_until(lambda: receiver.active == 1)
        await wait_until(lambda: receiver.sessions[0].socket.options)
        app.check()
    options = receiver.sessions[0].socket.options
    assert options[socket.SO_KEEPALIVE] == 1
    if hasattr(socket, "TCP_USER_TIMEOUT"):
        assert options[socket.TCP_USER_TIMEOUT] == 10000
        assert options[socket.TCP_KEEPIDLE] + options[socket.TCP_KEEPINTVL] * options[socket.TCP_KEEPCNT] == 10


async def test_slow_receiver_skips_ahead_instead_of_lagging(make_config, recorder, receiver, caplog):
    receiver.speed = 0.5
    recorder.set(mode="ramp")
    async with App(make_config(max_backlog_seconds=0.5, preroll_seconds=0.1)) as app:
        await wait_until(lambda: receiver.active == 1)
        await asyncio.sleep(2.5)
        app.check()
        assert receiver.active == 1
    assert "skipping ahead" in caplog.text


async def test_fast_capture_burst_does_not_break_stream(make_config, recorder, receiver):
    recorder.set(mode="ramp")
    async with App(make_config(max_backlog_seconds=1.0)) as app:
        await wait_until(lambda: receiver.active == 1)
        recorder.set(mode="ramp", rate=4.0)
        await asyncio.sleep(1.0)
        recorder.set(mode="ramp")
        await asyncio.sleep(1.0)
        assert receiver.active == 1
        app.check()


async def test_unstoppable_stream_is_fatal(make_config, recorder, receiver):
    receiver.stream = "zombie"
    recorder.set(mode="ramp")
    try:
        async with App(make_config()) as app:
            main = app.tasks[1]
            await wait_until(main.done, timeout=10)
            with pytest.raises(stream.FatalError, match="did not stop"):
                main.result()
    finally:
        receiver.released = True


async def test_monitor_ignores_unknown_pyatv_layout(make_config, recorder, receiver, monkeypatch):
    monkeypatch.setattr(stream, "receiver_connected", lambda atv: None)
    recorder.set(mode="ramp")
    async with App(make_config()) as app:
        await wait_until(lambda: receiver.active == 1)
        await asyncio.sleep(0.5)
        assert receiver.active == 1
        app.check()


async def test_heartbeat_touches_file_and_logs_status(make_config, recorder, caplog, tmp_path):
    config = make_config(status_interval_seconds=0.3, watchdog_seconds=0.4)
    stats = stream.Stats()
    recorder.set(mode="tone")
    async with App(config, stream_loop=False) as app:
        watchdog = stream.Watchdog(60)
        task = asyncio.create_task(stream.heartbeat(config, app.capture, stats, watchdog))
        await wait_until(lambda: "status:" in caplog.text, timeout=3)
        task.cancel()
    heartbeat = tmp_path / "heartbeat"
    assert time.time() - heartbeat.stat().st_mtime < 5
    line = next(line for line in caplog.text.splitlines() if "status:" in line)
    for field in ("rms_1s=", "peak=", "clipped=", "drift_ppm=", "rss_mb=", "fds=", "threads=", "tasks="):
        assert field in line


async def test_heartbeat_dies_when_main_loop_is_stuck(make_config, recorder, monkeypatch):
    deaths = []

    def fake_die(reason, **kwargs):
        deaths.append(reason)
        raise SystemExit

    monkeypatch.setattr(stream, "die", fake_die)
    config = make_config(stuck_seconds=0.2, watchdog_seconds=0.4)
    stats = stream.Stats()
    stats.progress -= 1
    async with App(config, stream_loop=False) as app:
        with pytest.raises(SystemExit):
            await stream.heartbeat(config, app.capture, stats, stream.Watchdog(60))
    assert deaths and "no progress" in deaths[0]


def test_watchdog_fires_once_when_loop_stops_beating(monkeypatch):
    deaths = []
    monkeypatch.setattr(stream, "die", lambda reason, **kw: deaths.append(reason))
    watchdog = stream.Watchdog(0.2)
    watchdog.start()
    time.sleep(0.6)
    assert len(deaths) == 1 and "unresponsive" in deaths[0]
    assert not watchdog._thread.is_alive()


def test_watchdog_stays_quiet_while_beating(monkeypatch):
    deaths = []
    monkeypatch.setattr(stream, "die", lambda reason, **kw: deaths.append(reason))
    watchdog = stream.Watchdog(0.2)
    watchdog.start()
    for _ in range(12):
        watchdog.beat()
        time.sleep(0.05)
    watchdog.last = float("inf")
    assert deaths == []
