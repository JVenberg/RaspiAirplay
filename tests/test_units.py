import asyncio
import shutil
import socket
import subprocess
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import wait_until

import stream


def frames_from(start, n):
    i = np.arange(start, start + n)
    return np.stack([i % 30000, -(i % 30000)], axis=1).astype(np.int16)


class TestRingBuffer:
    def test_read_back_across_wraparound(self):
        ring = stream.RingBuffer(100)
        pos = 0
        for n in [30, 50, 40, 99, 1, 100, 7]:
            ring.write(frames_from(pos, n))
            pos += n
        assert ring.end == pos
        assert ring.start == pos - 100
        np.testing.assert_array_equal(ring.read(ring.start, 100), frames_from(ring.start, 100))
        np.testing.assert_array_equal(ring.read(pos - 10, 10), frames_from(pos - 10, 10))

    def test_oversized_write_keeps_newest(self):
        ring = stream.RingBuffer(10)
        ring.write(frames_from(0, 25))
        assert (ring.start, ring.end) == (15, 25)
        np.testing.assert_array_equal(ring.read(15, 10), frames_from(15, 10))

    def test_out_of_range_reads_raise(self):
        ring = stream.RingBuffer(10)
        ring.write(frames_from(0, 15))
        with pytest.raises(IndexError):
            ring.read(4, 2)
        with pytest.raises(IndexError):
            ring.read(14, 2)

    def test_read_returns_copy(self):
        ring = stream.RingBuffer(10)
        ring.write(frames_from(0, 5))
        out = ring.read(0, 5)
        ring.write(frames_from(100, 10))
        np.testing.assert_array_equal(out, frames_from(0, 5))

    def test_matches_reference_model_under_random_ops(self):
        rng = np.random.default_rng(1)
        capacity = 97
        ring = stream.RingBuffer(capacity)
        model = deque(maxlen=capacity)
        pos = 0
        for _ in range(3000):
            n = int(rng.integers(0, 250))
            chunk = frames_from(pos, n)
            ring.write(chunk)
            model.extend(map(tuple, chunk))
            pos += n
            assert ring.end == pos
            if model:
                start = int(rng.integers(ring.start, ring.end))
                length = int(rng.integers(0, ring.end - start + 1))
                expected = np.array(list(model)[start - ring.start : start - ring.start + length], dtype=np.int16)
                np.testing.assert_array_equal(ring.read(start, length), expected.reshape(-1, 2))


class TestVolumeMeter:
    def test_full_scale_does_not_overflow(self):
        meter = stream.VolumeMeter(10)
        meter.add(np.full((4410, 2), -32768, dtype=np.int16))
        assert meter.rms(1) == pytest.approx(32768)

    def test_rms_of_sine(self):
        meter = stream.VolumeMeter(10)
        t = np.arange(4410) / 44100
        wave = (10000 * np.sin(2 * np.pi * 441 * t)).astype(np.int16)
        meter.add(np.stack([wave, wave], axis=1))
        assert meter.rms(1) == pytest.approx(10000 / np.sqrt(2), rel=1e-3)

    def test_window_only_covers_recent_blocks(self):
        meter = stream.VolumeMeter(5)
        loud = np.full((10, 2), 1000, dtype=np.int16)
        quiet = np.zeros((10, 2), dtype=np.int16)
        for _ in range(7):
            meter.add(loud)
        meter.add(quiet)
        assert meter.rms(1) == 0
        assert meter.rms(2) == pytest.approx(1000 / np.sqrt(2))
        assert meter.rms(50) == pytest.approx(1000 * np.sqrt(4 / 5))

    def test_empty_meter_is_silent(self):
        assert stream.VolumeMeter(5).rms(3) == 0.0

    def test_startup_transient_is_diluted_by_the_full_window(self):
        meter = stream.VolumeMeter(50)
        meter.add(np.full((10, 2), 10, dtype=np.int16))
        assert meter.rms(1) == pytest.approx(10)
        assert meter.rms(4) == pytest.approx(5)
        assert meter.rms(100) == pytest.approx(10 / np.sqrt(50))

    def test_boot_transient_stays_below_threshold(self):
        config = stream.Config()
        meter = stream.VolumeMeter(config.blocks(60))
        start = config.blocks(config.start_window_seconds)
        rng = np.random.default_rng(0)
        for i in range(start * 2):
            # The HiFiBerry ADC read about rms 5.2 for its first second after a cold boot.
            rms = 5.2 if i < config.blocks(1.0) else 0.9
            meter.add(rng.normal(0, rms, (config.block_frames, 2)).round().astype(np.int16))
            assert meter.rms(start) <= config.threshold

    def test_tracks_peak_and_clipping(self):
        meter = stream.VolumeMeter(5)
        meter.add(np.array([[16384, -100], [32767, -32768]], dtype=np.int16))
        peak, clipped = meter.take_peak()
        assert peak == pytest.approx(0, abs=0.01)
        assert clipped == 2
        meter.add(np.array([[16384, -16384]], dtype=np.int16))
        assert meter.take_peak() == (pytest.approx(-6.02, abs=0.01), 0)


class TestRateLimiter:
    def test_burst_then_suppress_then_reset(self, monkeypatch, caplog):
        now = [1000.0]
        monkeypatch.setattr(stream.time, "monotonic", lambda: now[0])
        limiter = stream.RateLimiter(burst=3, window=10)
        assert [limiter.allow() for _ in range(5)] == [True, True, True, False, False]
        now[0] += 10
        with caplog.at_level("WARNING"):
            assert limiter.allow()
        assert "suppressed 2 similar log lines" in caplog.text


def test_backoff_doubles_caps_and_resets():
    backoff = stream.Backoff(1, 5)
    assert [backoff.next() for _ in range(5)] == [1, 2, 4, 5, 5]
    backoff.reset()
    assert backoff.next() == 1


def test_default_capture_cmd():
    cmd = stream.default_capture_cmd(1.5, "hw:1")
    assert cmd[:4] == ("sox", "-q", "-V2", "-t")
    assert cmd[cmd.index("alsa") + 9] == "hw:1"
    assert cmd[-2:] == ("vol", "1.5")
    assert "vol" not in stream.default_capture_cmd(1.0)


def run_capture_pipeline(tmp_path, samples32, gain):
    """Runs the real sox command with the ALSA input swapped for a raw 32-bit file."""
    src = tmp_path / "in.raw"
    src.write_bytes(samples32.astype("<i4").tobytes())
    cmd = list(stream.default_capture_cmd(gain, str(src)))
    cmd[cmd.index("alsa")] = "raw"
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=30).stdout
    return np.frombuffer(out, dtype="<i2").reshape(-1, 2)


@pytest.mark.skipif(shutil.which("sox") is None, reason="sox not installed")
class TestCapturePipeline:
    def sine(self, dbfs, seconds=1.0, freq=1000.0):
        t = np.arange(int(44100 * seconds)) / 44100
        wave = 10 ** (dbfs / 20) * 2**31 * np.sin(2 * np.pi * freq * t)
        return np.stack([wave, wave], axis=1).round().astype(np.int64)

    def test_applies_gain_and_keeps_little_endian_frames(self, tmp_path):
        out = run_capture_pipeline(tmp_path, self.sine(-20), 1.5)
        assert out.shape == (44100, 2)
        rms = np.sqrt(np.mean(out.astype(np.float64) ** 2))
        assert rms == pytest.approx(32768 * 0.1 * 1.5 / np.sqrt(2), rel=0.01)

    def test_keeps_detail_below_16_bits(self, tmp_path):
        """A -100 dBFS tone vanishes if truncated to 16 bits first; dither keeps it audible."""
        out = run_capture_pipeline(tmp_path, self.sine(-100, seconds=2.0), 1.0)
        spectrum = np.abs(np.fft.rfft(out[:, 0].astype(np.float64)))
        tone_bin = round(1000 * len(out) / 44100)
        assert spectrum[tone_bin] > 20 * np.median(spectrum)

    def test_loud_input_clips_and_is_metered(self, tmp_path):
        out = run_capture_pipeline(tmp_path, self.sine(-1), 1.5)
        meter = stream.VolumeMeter(10)
        meter.add(out)
        peak, clipped = meter.take_peak()
        assert peak == pytest.approx(0, abs=0.01)
        assert clipped > 1000
        assert meter.take_peak() == (-np.inf, 0)


def test_drift_ppm_compares_captured_audio_to_elapsed_time(make_config):
    capture = stream.Capture(make_config())
    assert capture.drift_ppm() is None
    capture.ring.write(np.zeros((44100, 2), np.int16))
    capture.clock_start = (capture.last_data - 30, 0)
    assert capture.drift_ppm() is None
    capture.clock_start = (capture.last_data - 100, capture.ring.end - 4410441)
    assert capture.drift_ppm() == pytest.approx(100, abs=0.01)


class FakeCapture:
    def __init__(self, config):
        self.ring = stream.RingBuffer(config.frames(config.buffer_seconds))
        self.pos = 0

    def feed(self, n):
        self.ring.write(frames_from(self.pos, n))
        self.pos += n


def decode(data):
    return np.frombuffer(data, dtype=">i2").astype(np.int16).reshape(-1, 2)


class TestLiveSource:
    def make(self, make_config, preroll=0.1, **kw):
        config = make_config(preroll_seconds=preroll, **kw)
        capture = FakeCapture(config)
        return config, capture

    async def test_starts_preroll_behind_and_is_big_endian(self, make_config):
        config, capture = self.make(make_config)
        capture.feed(44100)
        source = stream.LiveSource(capture, config, None)
        data = await source.readframes(352)
        assert len(data) == 352 * 4
        np.testing.assert_array_equal(decode(data), frames_from(44100 - 4410, 352))
        assert source.frames_sent == 352

    async def test_consecutive_reads_are_contiguous(self, make_config):
        config, capture = self.make(make_config)
        capture.feed(44100)
        source = stream.LiveSource(capture, config, None)
        out = [decode(await source.readframes(352)) for _ in range(10)]
        np.testing.assert_array_equal(np.concatenate(out), frames_from(44100 - 4410, 3520))

    async def test_underrun_plays_silence_until_preroll_refills(self, make_config):
        config, capture = self.make(make_config)
        capture.feed(100)
        source = stream.LiveSource(capture, config, None)
        assert await source.readframes(352) == bytes(352 * 4)
        capture.feed(4000)
        assert await source.readframes(352) == bytes(352 * 4)
        capture.feed(310)
        np.testing.assert_array_equal(decode(await source.readframes(352)), frames_from(0, 352))
        assert source.underruns == 1 and source.silence_sent == 704 and not source.capture_lost

    async def test_cushion_survives_a_capture_gap(self, make_config):
        """After a gap the stream resumes one preroll behind capture instead of at the head."""
        config, capture = self.make(make_config)
        capture.feed(44100)
        source = stream.LiveSource(capture, config, None)
        while source.pos + 352 <= capture.ring.end:
            await source.readframes(352)
        resume = source.pos
        assert await source.readframes(352) == bytes(352 * 4)
        capture.feed(source.preroll)
        np.testing.assert_array_equal(decode(await source.readframes(352)), frames_from(resume, 352))
        assert capture.ring.end - source.pos >= source.preroll - 352
        assert source.underruns == 1

    async def test_underrun_timeout_marks_capture_lost(self, make_config):
        config, capture = self.make(make_config, underrun_timeout_seconds=0.1)
        capture.feed(4410)
        source = stream.LiveSource(capture, config, None)
        while source.pos + 352 <= capture.ring.end:
            await source.readframes(352)
        assert await source.readframes(352) == bytes(352 * 4)
        await asyncio.sleep(0.15)
        assert await source.readframes(352) == b""
        assert source.capture_lost
        assert await source.readframes(352) == b""

    async def test_closed_source_returns_no_frames(self, make_config):
        config, capture = self.make(make_config)
        capture.feed(44100)
        source = stream.LiveSource(capture, config, None)
        await source.close()
        assert await source.readframes(352) == b""
        assert not source.capture_lost

    async def test_skips_ahead_when_far_behind(self, make_config):
        config, capture = self.make(make_config, max_backlog_seconds=0.5)
        capture.feed(4410)
        source = stream.LiveSource(capture, config, None)
        capture.feed(44100)
        data = await source.readframes(352)
        assert source.skips == 1
        np.testing.assert_array_equal(decode(data), frames_from(4410 + 44100 - 4410, 352))

    async def test_skips_ahead_when_overwritten(self, make_config):
        config, capture = self.make(make_config, buffer_seconds=1.0, max_backlog_seconds=10)
        capture.feed(4410)
        source = stream.LiveSource(capture, config, None)
        capture.feed(44100)
        await source.readframes(352)
        assert source.skips == 1

    def test_reports_format(self, make_config):
        config, capture = self.make(make_config)
        source = stream.LiveSource(capture, config, "meta")
        assert (source.sample_rate, source.channels, source.sample_size, source.duration) == (44100, 2, 2, 0)
        assert asyncio.run(source.get_metadata()) == "meta"


class TestOpenSourcePatch:
    def test_installed_on_pyatv(self):
        assert stream.raop.open_source is stream._open_source

    async def test_passes_live_source_through(self, make_config):
        config = make_config()
        source = stream.LiveSource(FakeCapture(config), config, None)
        assert await stream.raop.open_source(source, 44100, 2, 2) is source

    async def test_rejects_format_mismatch(self, make_config):
        config = make_config()
        source = stream.LiveSource(FakeCapture(config), config, None)
        with pytest.raises(ValueError):
            await stream.raop.open_source(source, 48000, 2, 2)

    async def test_delegates_other_sources(self, monkeypatch):
        calls = []

        async def original(*args):
            calls.append(args)
            return "opened"

        monkeypatch.setattr(stream, "_original_open_source", original)
        assert await stream.raop.open_source("file.mp3", 44100, 2, 2) == "opened"
        assert calls == [("file.mp3", 44100, 2, 2)]


class TestReceiverConnected:
    def test_unknown_when_internals_missing(self):
        assert stream.receiver_connected(object()) is None

    async def test_tracks_transport(self, receiver):
        atv = await stream.pyatv.connect(None, None)
        assert stream.receiver_connected(atv) is None
        session = receiver.sessions[-1]
        session.manager.stream_client = session.client
        assert stream.receiver_connected(atv) is True
        session.connection.transport = None
        assert stream.receiver_connected(atv) is False

    async def test_real_pyatv_layout(self):
        """Guards the attribute path against pyatv upgrades."""
        from pyatv.protocols.raop import RaopPlaybackManager, RaopStream
        from pyatv.protocols.raop.stream_client import StreamClient
        from pyatv.support.rtsp import RtspSession

        assert "playback_manager" in RaopStream.__init__.__code__.co_names
        assert isinstance(RaopPlaybackManager.stream_client, property)
        assert "rtsp" in StreamClient.__init__.__code__.co_names
        assert "connection" in RtspSession.__init__.__code__.co_names


class TestDetectDeadPeer:
    async def test_sets_kernel_options_on_real_socket(self):
        loop = asyncio.get_running_loop()
        server = await loop.create_server(asyncio.Protocol, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        transport, _ = await loop.create_connection(asyncio.Protocol, "127.0.0.1", port)
        try:
            stream.detect_dead_peer(transport, 15.0)
            sock = transport.get_extra_info("socket")
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
            if hasattr(socket, "TCP_USER_TIMEOUT"):
                assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT) == 15000
                assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 3
                assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 3
                assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 4
        finally:
            transport.close()
            server.close()
            await server.wait_closed()


class TestDropProgressMetadata:
    def test_real_pyatv_conf_stops_advertising_progress(self):
        from pyatv.conf import AppleTV, ManualService
        from pyatv.protocols.raop.parsers import MetadataType, get_metadata_types

        conf = AppleTV("10.0.0.5", "Bose")
        conf.add_service(ManualService("AA:BB", stream.Protocol.RAOP, 7000, {"md": "0,1,2", "et": "0"}))
        conf.add_service(ManualService("AA:BB", stream.Protocol.AirPlay, 7000, {"features": "0x1"}))
        stream.drop_progress_metadata(conf)
        types = get_metadata_types(conf.get_service(stream.Protocol.RAOP).properties)
        assert types == MetadataType.Text | MetadataType.Artwork
        assert "md" not in conf.get_service(stream.Protocol.AirPlay).properties

    @pytest.mark.parametrize("md, expected", [("2", ""), ("2,0", "0"), ("0, 2", "0"), ("0,1", "0,1")])
    def test_removes_only_progress(self, md, expected):
        service = SimpleNamespace(properties={"md": md})
        stream.drop_progress_metadata(SimpleNamespace(services=[service]))
        assert service.properties["md"] == expected


class TestStats:
    def test_touch_advances_progress(self):
        stats = stream.Stats()
        before = stats.progress
        time.sleep(0.01)
        stats.touch()
        assert stats.progress > before


class TestLogLoopError:
    def test_network_error_is_one_warning_line(self, caplog):
        loop = asyncio.new_event_loop()
        try:
            loop.set_exception_handler(stream.log_loop_error)
            ex = OSError(113, "Connect call failed")
            loop.call_exception_handler({"message": "Task exception was never retrieved", "exception": ex})
        finally:
            loop.close()
        assert "background task failed: OSError: [Errno 113] Connect call failed" in caplog.text
        assert "Traceback" not in caplog.text

    def test_other_errors_use_default_handler(self, caplog):
        loop = asyncio.new_event_loop()
        try:
            loop.set_exception_handler(stream.log_loop_error)
            loop.call_exception_handler({"message": "boom", "exception": ValueError("bad")})
        finally:
            loop.close()
        assert "boom" in caplog.text and "ValueError: bad" in caplog.text


async def test_wait_until_helper_times_out():
    with pytest.raises(AssertionError):
        await wait_until(lambda: False, timeout=0.05)
