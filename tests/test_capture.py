import asyncio

import numpy as np
import psutil
import pytest
from fakes import App, ramp_index, wait_until


def alive(pid):
    try:
        return psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


async def test_meters_tone(make_config, recorder):
    recorder.set(mode="tone", amplitude=8000)
    async with App(make_config(), stream_loop=False) as app:
        await wait_until(lambda: app.capture.ring.end > 44100 * 0.6)
        assert app.capture.meter.rms(2) == pytest.approx(8000 / np.sqrt(2), rel=0.02)
        assert not app.capture.stalled()


async def test_ring_holds_contiguous_audio(make_config, recorder):
    recorder.set(mode="ramp")
    async with App(make_config(), stream_loop=False) as app:
        ring = app.capture.ring
        await wait_until(lambda: ring.end > 44100)
        idx = ramp_index(ring.read(ring.start, ring.end - ring.start))
        assert np.all(np.diff(idx) % 65536 == 1)


@pytest.mark.parametrize("mode", ["exit", "partial"])
async def test_restarts_when_recorder_dies(make_config, recorder, caplog, mode):
    recorder.set(mode=mode, code=2)
    async with App(make_config(), stream_loop=False) as app:
        await wait_until(lambda: app.capture.restarts >= 3)
        assert len(recorder.starts) >= 3
        recorder.set(mode="tone")
        await wait_until(lambda: app.capture.ring.end > 44100 * 0.3)
        assert not app.capture.stalled()
        app.check()
    assert "capture process exited" in caplog.text


async def test_kills_and_restarts_stalled_recorder(make_config, recorder, caplog):
    recorder.set(mode="ramp")
    async with App(make_config(capture_stall_seconds=0.5), stream_loop=False) as app:
        await wait_until(lambda: app.capture.ring.end > 4410)
        first = recorder.starts[0]
        recorder.set(mode="stall")
        await wait_until(lambda: app.capture.stalled())
        await wait_until(lambda: len(recorder.starts) >= 2)
        await wait_until(lambda: not alive(first))
        recorder.set(mode="ramp")
        end = app.capture.ring.end
        await wait_until(lambda: app.capture.ring.end > end + 4410)
        assert not app.capture.stalled()
    assert "capture produced no audio" in caplog.text


async def test_missing_recorder_binary_is_retried(make_config, caplog):
    config = make_config(capture_cmd=("/nonexistent/rec",), capture_max_backoff_seconds=0.1)
    async with App(config, stream_loop=False) as app:
        await wait_until(lambda: app.capture.restarts >= 3)
        app.check()
    assert "FileNotFoundError" in caplog.text


async def test_stderr_is_rate_limited(make_config, recorder, caplog):
    recorder.set(mode="tone", stderr="rec WARN alsa: over-run")
    async with App(make_config(), stream_loop=False) as app:
        await wait_until(lambda: app.capture.ring.end > 44100)
    assert 1 <= caplog.text.count("capture: rec WARN alsa: over-run") <= 5


async def test_benign_sox_warning_is_hidden(make_config, recorder, caplog):
    recorder.set(mode="tone", stderr="rec WARN wav: Length in output .wav header will be wrong; can't encode 0-bit")
    async with App(make_config(), stream_loop=False) as app:
        await wait_until(lambda: app.capture.ring.end > 4410 * 3)
    assert "can't encode" not in caplog.text


async def test_no_leaked_processes_or_fds_after_many_restarts(make_config, recorder):
    me = psutil.Process()
    recorder.set(mode="exit", code=0)
    config = make_config(capture_max_backoff_seconds=0.05)
    fds_before = me.num_fds()
    async with App(config, stream_loop=False) as app:
        await wait_until(lambda: app.capture.restarts >= 20, timeout=60)
        recorder.set(mode="tone")
        await wait_until(lambda: not app.capture.stalled() and app.capture.ring.end > 4410, timeout=10)
        children = me.children()
        assert len(children) == 1 and children[0].status() != psutil.STATUS_ZOMBIE
        assert me.num_fds() <= fds_before + 6
    await asyncio.sleep(0.2)
    assert me.children() == []
    assert me.num_fds() <= fds_before + 1
    assert all(not alive(pid) for pid in recorder.starts)
