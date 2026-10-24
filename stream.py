#!/usr/bin/env python3
"""Stream line-in audio to an AirPlay receiver whenever sound is detected."""

import asyncio
import contextlib
import dataclasses
import faulthandler
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import click
import numpy as np
import pyatv
import pyatv.protocols.raop as raop
from pyatv.const import Protocol
from pyatv.interface import MediaMetadata
from pyatv.protocols.raop.audio_source import AudioSource

logger = logging.getLogger("airplay")

SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_BYTES = 2


@dataclasses.dataclass
class Config:
    gain: float = 1.0
    identifier: str = ""
    device: str = "default"
    threshold: float = 5.0
    silence_seconds: float = 10.0
    capture_cmd: tuple = ()
    block_seconds: float = 0.1
    start_window_seconds: float = 2.0
    preroll_seconds: float = 1.0
    buffer_seconds: float = 30.0
    max_backlog_seconds: float = 15.0
    capture_stall_seconds: float = 5.0
    underrun_timeout_seconds: float = 8.0
    scan_timeout_seconds: float = 10.0
    scan_grace_seconds: float = 10.0
    connect_timeout_seconds: float = 30.0
    setup_timeout_seconds: float = 30.0
    stream_stall_seconds: float = 10.0
    receiver_timeout_seconds: float = 15.0
    stop_timeout_seconds: float = 15.0
    min_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    capture_max_backoff_seconds: float = 30.0
    healthy_session_seconds: float = 60.0
    poll_seconds: float = 0.25
    status_interval_seconds: float = 300.0
    heartbeat_path: str = "/tmp/heartbeat"
    watchdog_seconds: float = 60.0
    stuck_seconds: float = 300.0

    @property
    def block_frames(self):
        return round(SAMPLE_RATE * self.block_seconds)

    def blocks(self, seconds):
        return max(1, round(seconds / self.block_seconds))

    def frames(self, seconds):
        return round(seconds * SAMPLE_RATE)


def default_capture_cmd(gain, device="default"):
    """Captures 32-bit from ALSA so gain is applied before sox dithers down to 16-bit."""
    fmt = ["-r", str(SAMPLE_RATE), "-c", str(CHANNELS), "-e", "signed-integer"]
    cmd = ["sox", "-q", "-V2", "-t", "alsa", *fmt, "-b", "32", device]
    cmd += ["-t", "raw", *fmt, "-b", "16", "-L", "-"]
    if gain != 1.0:
        cmd += ["vol", str(gain)]
    return tuple(cmd)


class FatalError(Exception):
    """Unrecoverable state; exit so the container restart policy starts fresh."""


class DeviceNotFound(Exception):
    pass


class RingBuffer:
    """Fixed-size circular store of interleaved int16 frames, indexed by absolute frame."""

    def __init__(self, capacity, channels=CHANNELS):
        self.capacity = capacity
        self.data = np.zeros((capacity, channels), dtype=np.int16)
        self.end = 0

    @property
    def start(self):
        return max(0, self.end - self.capacity)

    def write(self, frames):
        n = len(frames)
        if n > self.capacity:
            self.end += n - self.capacity
            frames = frames[-self.capacity :]
            n = self.capacity
        i = self.end % self.capacity
        first = min(n, self.capacity - i)
        self.data[i : i + first] = frames[:first]
        self.data[: n - first] = frames[first:]
        self.end += n

    def read(self, pos, n):
        if pos < self.start or pos + n > self.end:
            raise IndexError(f"frames {pos}..{pos + n} not in {self.start}..{self.end}")
        i = pos % self.capacity
        first = min(n, self.capacity - i)
        if first == n:
            return self.data[i : i + n].copy()
        return np.concatenate((self.data[i:], self.data[: n - first]))


class VolumeMeter:
    """RMS level over recent fixed-size blocks, computed in float to avoid int16 overflow."""

    def __init__(self, history_blocks):
        self.mean_squares = np.zeros(history_blocks, dtype=np.float64)
        self.count = 0
        self.peak = 0
        self.clipped = 0

    def add(self, frames):
        x = frames.astype(np.float32).ravel()
        if x.size == 0:
            return
        self.mean_squares[self.count % len(self.mean_squares)] = np.dot(x, x) / x.size
        self.count += 1
        self.peak = max(self.peak, int(-x.min()), int(x.max()))
        self.clipped += int(np.count_nonzero((x >= 32767) | (x <= -32768)))

    def take_peak(self):
        """Peak in dBFS and clipped sample count since the last call."""
        peak, clipped = self.peak, self.clipped
        self.peak = self.clipped = 0
        return (20 * np.log10(peak / 32768) if peak else -np.inf), clipped

    def rms(self, blocks):
        n = min(blocks, self.count, len(self.mean_squares))
        if n == 0:
            return 0.0
        idx = np.arange(self.count - n, self.count) % len(self.mean_squares)
        return float(np.sqrt(self.mean_squares[idx].mean()))


class RateLimiter:
    """Allows a burst of events per window and counts what it drops."""

    def __init__(self, burst=5, window=60.0):
        self.burst, self.window = burst, window
        self.window_start, self.used, self.suppressed = -float("inf"), 0, 0

    def allow(self):
        now = time.monotonic()
        if now - self.window_start >= self.window:
            self.window_start, self.used = now, 0
            if self.suppressed:
                logger.warning("suppressed %d similar log lines", self.suppressed)
                self.suppressed = 0
        if self.used < self.burst:
            self.used += 1
            return True
        self.suppressed += 1
        return False


class Backoff:
    def __init__(self, minimum, maximum):
        self.minimum, self.maximum = minimum, maximum
        self.current = minimum

    def next(self):
        delay = self.current
        self.current = min(self.current * 2, self.maximum)
        return delay

    def reset(self):
        self.current = self.minimum


class Capture:
    """Runs the recorder subprocess forever, feeding a ring buffer and volume meter."""

    def __init__(self, config):
        self.config = config
        self.cmd = config.capture_cmd or default_capture_cmd(config.gain, config.device)
        self.ring = RingBuffer(config.frames(config.buffer_seconds))
        self.meter = VolumeMeter(config.blocks(max(60.0, config.silence_seconds * 2)))
        self.last_data = time.monotonic()
        self.clock_start = None
        self.restarts = 0
        self._stderr_limit = RateLimiter()

    def stalled(self):
        return time.monotonic() - self.last_data > self.config.capture_stall_seconds

    def drift_ppm(self):
        """ADC clock rate relative to the system clock since capture started, once measurable."""
        if self.clock_start is None:
            return None
        started, start_frames = self.clock_start
        elapsed = self.last_data - started
        if elapsed < 60:
            return None
        return ((self.ring.end - start_frames) / SAMPLE_RATE / elapsed - 1) * 1e6

    async def run(self):
        backoff = Backoff(self.config.min_backoff_seconds, self.config.capture_max_backoff_seconds)
        while True:
            started = time.monotonic()
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("capture failed")
            if time.monotonic() - started > self.config.healthy_session_seconds:
                backoff.reset()
            self.restarts += 1
            delay = backoff.next()
            logger.warning("restarting capture in %.1fs", delay)
            await asyncio.sleep(delay)

    async def _run_once(self):
        block_bytes = self.config.block_frames * CHANNELS * SAMPLE_BYTES
        proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("capture started (pid %d): %s", proc.pid, " ".join(self.cmd))
        self.clock_start = None
        stderr_task = asyncio.create_task(self._log_stderr(proc.stderr))
        problem = None
        try:
            while True:
                data = await asyncio.wait_for(proc.stdout.readexactly(block_bytes), self.config.capture_stall_seconds)
                frames = np.frombuffer(data, dtype="<i2").reshape(-1, CHANNELS)
                self.ring.write(frames)
                self.meter.add(frames)
                self.last_data = time.monotonic()
                if self.clock_start is None:
                    self.clock_start = (self.last_data, self.ring.end)
        except asyncio.IncompleteReadError:
            problem = "capture process exited"
        except TimeoutError:
            problem = f"capture produced no audio for {self.config.capture_stall_seconds:.0f}s"
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), 5.0)
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
        logger.error("%s (exit code %s)", problem, proc.returncode)

    async def _log_stderr(self, stream):
        while line := await stream.readline():
            text = line.decode(errors="replace").strip()
            if text and "can't encode 0-bit" not in text and self._stderr_limit.allow():
                logger.warning("capture: %s", text)


class LiveSource(AudioSource):
    """pyatv source playing the ring buffer; underruns play silence until the preroll refills."""

    def __init__(self, capture, config, metadata):
        self.capture = capture
        self.config = config
        self.metadata = metadata
        self.preroll = config.frames(config.preroll_seconds)
        ring = capture.ring
        self.pos = max(ring.start, ring.end - self.preroll)
        self.created = time.monotonic()
        self.last_read = None
        self.rebuffering_since = None
        self.frames_sent = 0
        self.silence_sent = 0
        self.underruns = 0
        self.skips = 0
        self.capture_lost = False
        self.closed = False

    async def readframes(self, nframes):
        self.last_read = now = time.monotonic()
        if self.closed or self.capture_lost:
            return AudioSource.NO_FRAMES
        ring = self.capture.ring
        if self.rebuffering_since is None and ring.end - self.pos < nframes:
            self.underruns += 1
            self.rebuffering_since = now
        if self.rebuffering_since is not None:
            if ring.end - self.pos >= max(self.preroll, nframes):
                self.rebuffering_since = None
            elif now - self.rebuffering_since > self.config.underrun_timeout_seconds:
                self.capture_lost = True
                return AudioSource.NO_FRAMES
            else:
                self.silence_sent += nframes
                return bytes(nframes * CHANNELS * SAMPLE_BYTES)
        if self.pos < ring.start or ring.end - self.pos > self.config.frames(self.config.max_backlog_seconds):
            self.skips += 1
            new_pos = ring.end - self.preroll
            logger.warning("stream fell %.1fs behind capture; skipping ahead", (new_pos - self.pos) / SAMPLE_RATE)
            self.pos = new_pos
        frames = ring.read(self.pos, nframes)
        self.pos += nframes
        self.frames_sent += nframes
        # pyatv sends samples big-endian (see _to_audio_samples in pyatv's audio_source)
        return frames.astype(">i2").tobytes()

    async def close(self):
        self.closed = True

    async def get_metadata(self):
        return self.metadata

    @property
    def sample_rate(self):
        return SAMPLE_RATE

    @property
    def channels(self):
        return CHANNELS

    @property
    def sample_size(self):
        return SAMPLE_BYTES

    @property
    def duration(self):
        return 0


_original_open_source = raop.open_source


async def _open_source(source, sample_rate, channels, sample_size):
    if not isinstance(source, LiveSource):
        return await _original_open_source(source, sample_rate, channels, sample_size)
    wanted = (source.sample_rate, source.channels, source.sample_size)
    if (sample_rate, channels, sample_size) != wanted:
        raise ValueError(f"receiver wants {(sample_rate, channels, sample_size)}, capture is {wanted}")
    return source


raop.open_source = _open_source


def rtsp_connection(atv):
    """pyatv's RTSP connection for the active stream, or None if not streaming or internals changed."""
    try:
        client = atv.stream.get(Protocol.RAOP).playback_manager.stream_client
        return None if client is None else client.rtsp.connection
    except AttributeError:
        return None


def receiver_connected(atv):
    """Whether the RTSP connection is still open, or None if unknown."""
    connection = rtsp_connection(atv)
    return None if connection is None else connection.transport is not None


def detect_dead_peer(transport, timeout):
    """Audio is UDP, so only TCP probes on the RTSP socket notice a receiver that vanished."""
    sock = transport.get_extra_info("socket")
    interval = max(1, int(timeout / 5))
    options = [
        (socket.SOL_SOCKET, "SO_KEEPALIVE", 1),
        (socket.IPPROTO_TCP, "TCP_KEEPIDLE", interval),
        (socket.IPPROTO_TCP, "TCP_KEEPINTVL", interval),
        (socket.IPPROTO_TCP, "TCP_KEEPCNT", max(1, int(timeout / interval) - 1)),
        (socket.IPPROTO_TCP, "TCP_USER_TIMEOUT", int(timeout * 1000)),
    ]
    for level, name, value in options:
        if hasattr(socket, name):
            sock.setsockopt(level, getattr(socket, name), value)


class Stats:
    def __init__(self):
        self.started = self.progress = time.monotonic()
        self.state = "starting"
        self.sessions = 0
        self.failures = 0

    def touch(self):
        self.progress = time.monotonic()


def process_stats():
    stats = {"threads": threading.active_count()}
    with contextlib.suppress(OSError):
        stats["fds"] = len(os.listdir("/proc/self/fd"))
    with contextlib.suppress(OSError, ValueError, IndexError):
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        stats["rss_mb"] = round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    return stats


class Streamer:
    def __init__(self, config, capture, stats):
        self.config = config
        self.capture = capture
        self.stats = stats
        self.last_address = None

    async def discover(self):
        loop = asyncio.get_running_loop()
        identifier = self.config.identifier or None
        protocols = {Protocol.RAOP, Protocol.AirPlay}
        attempts = []
        if self.last_address:
            attempts.append(dict(hosts=[self.last_address], timeout=3))
        attempts.append(dict(timeout=self.config.scan_timeout_seconds))
        for kwargs in attempts:
            atvs = await asyncio.wait_for(
                pyatv.scan(loop, identifier=identifier, protocol=protocols, **kwargs),
                kwargs["timeout"] + self.config.scan_grace_seconds,
            )
            if atvs:
                return atvs[0]
        raise DeviceNotFound(f"no AirPlay receiver found (identifier={identifier!r})")

    async def run_session(self):
        """Stream until silence or failure; returns the reason the session ended."""
        conf = await self.discover()
        self.last_address = str(conf.address)
        logger.info("connecting to %s (%s)", conf.name, conf.address)
        loop = asyncio.get_running_loop()
        atv = await asyncio.wait_for(pyatv.connect(conf, loop), self.config.connect_timeout_seconds)
        source = LiveSource(
            self.capture,
            self.config,
            MediaMetadata(artist="Raspberry PI", title="Streaming from Raspberry PI"),
        )
        stream_task = asyncio.create_task(atv.stream.stream_file(source))
        reason = "cancelled"
        try:
            reason = await self._monitor(atv, stream_task, source)
            return reason
        except Exception as ex:
            reason = f"{type(ex).__name__}: {ex}"
            raise
        finally:
            await self._shutdown(atv, stream_task, source)
            logger.info(
                "stream ended: %s (%.0fs sent, %d underruns, %.1fs silence inserted, %d skips)",
                reason,
                source.frames_sent / SAMPLE_RATE,
                source.underruns,
                source.silence_sent / SAMPLE_RATE,
                source.skips,
            )

    async def _monitor(self, atv, stream_task, source):
        silence_blocks = self.config.blocks(self.config.silence_seconds)
        started = time.monotonic()
        announced = False
        while True:
            await asyncio.wait({stream_task}, timeout=self.config.poll_seconds)
            self.stats.touch()
            now = time.monotonic()
            if stream_task.done():
                if source.capture_lost:
                    return "capture lost"
                if not stream_task.cancelled() and (ex := stream_task.exception()):
                    logger.debug("stream task failed", exc_info=ex)
                    return f"receiver ended stream ({type(ex).__name__}: {ex})"
                return "receiver ended stream"
            if source.last_read is None:
                if now - source.created > self.config.setup_timeout_seconds:
                    return "stream setup timed out"
                continue
            if not announced:
                announced = True
                logger.info("streaming started after %.1fs", source.last_read - source.created)
                connection = rtsp_connection(atv)
                if connection is not None and connection.transport is not None:
                    try:
                        detect_dead_peer(connection.transport, self.config.receiver_timeout_seconds)
                    except OSError as ex:
                        logger.warning("could not enable dead receiver detection: %s", ex)
            if now - source.last_read > self.config.stream_stall_seconds:
                return "stream stalled"
            if receiver_connected(atv) is False:
                return "receiver closed connection"
            if (
                now - started >= self.config.silence_seconds
                and self.capture.meter.rms(silence_blocks) < self.config.threshold
            ):
                return "silence"

    async def _shutdown(self, atv, stream_task, source):
        # A closed source makes pyatv pad out its latency and send TEARDOWN on its own.
        await source.close()
        if not stream_task.done():
            await asyncio.wait({stream_task}, timeout=self.config.stop_timeout_seconds / 3)
        if not stream_task.done():
            stream_task.cancel()
            await asyncio.wait({stream_task}, timeout=self.config.stop_timeout_seconds)
        if not stream_task.done():
            raise FatalError("stream task did not stop")
        if not stream_task.cancelled() and stream_task.exception():
            logger.debug("stream task error during shutdown", exc_info=stream_task.exception())
        close_tasks = atv.close() or set()
        if not close_tasks:
            return
        _, pending = await asyncio.wait(close_tasks, timeout=self.config.stop_timeout_seconds)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=5.0)
        for task in close_tasks:
            if not task.done():
                raise FatalError("connection close did not finish")
            if not task.cancelled() and task.exception():
                logger.debug("error closing connection", exc_info=task.exception())


def die(reason, code=3, dump_stacks=True):
    """Exit immediately, skipping cleanup that could itself hang; Docker restarts us."""
    logger.critical("%s; exiting for restart", reason)
    if dump_stacks:
        faulthandler.dump_traceback(all_threads=True)
    logging.shutdown()
    os._exit(code)


def log_loop_error(loop, context):
    """pyatv leaves some failed network tasks unretrieved; those are noise, not crashes."""
    ex = context.get("exception")
    if isinstance(ex, OSError):
        logger.warning("background task failed: %s: %s", type(ex).__name__, ex)
    else:
        loop.default_exception_handler(context)


class Watchdog:
    """Kills the process if the event loop stops beating, so Docker restarts it."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.last = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)

    def start(self):
        self._thread.start()

    def beat(self):
        self.last = time.monotonic()

    def _run(self):
        while True:
            time.sleep(self.timeout / 4)
            stalled = time.monotonic() - self.last
            if stalled > self.timeout:
                die(f"event loop unresponsive for {stalled:.0f}s")
                return


async def heartbeat(config, capture, stats, watchdog):
    path = Path(config.heartbeat_path)
    last_status = time.monotonic()
    while True:
        watchdog.beat()
        stuck = time.monotonic() - stats.progress
        if stuck > config.stuck_seconds:
            for task in asyncio.all_tasks():
                task.print_stack(file=sys.stdout)
            die(f"main loop made no progress for {stuck:.0f}s")
        with contextlib.suppress(OSError):
            path.touch()
        if time.monotonic() - last_status >= config.status_interval_seconds:
            last_status = time.monotonic()
            log_status(config, capture, stats)
        await asyncio.sleep(min(5.0, config.watchdog_seconds / 4))


def log_status(config, capture, stats):
    meter = capture.meter
    peak_dbfs, clipped = meter.take_peak()
    drift = capture.drift_ppm()
    logger.info(
        "status: state=%s rms_1s=%.1f rms_%.0fs=%.1f peak=%.1fdBFS clipped=%d sessions=%d failures=%d "
        "capture_restarts=%d drift_ppm=%s tasks=%d uptime=%.1fh %s",
        stats.state,
        meter.rms(config.blocks(1.0)),
        config.silence_seconds,
        meter.rms(config.blocks(config.silence_seconds)),
        peak_dbfs,
        clipped,
        stats.sessions,
        stats.failures,
        capture.restarts,
        "n/a" if drift is None else f"{drift:+.1f}",
        len(asyncio.all_tasks()),
        (time.monotonic() - stats.started) / 3600,
        " ".join(f"{k}={v}" for k, v in process_stats().items()),
    )


async def wait_for_sound(config, capture, stats):
    stats.state = "idle"
    start_blocks = config.blocks(config.start_window_seconds)
    while capture.stalled() or capture.meter.rms(start_blocks) <= config.threshold:
        stats.touch()
        await asyncio.sleep(config.poll_seconds)


async def stream_forever(config, capture, streamer, stats):
    backoff = Backoff(config.min_backoff_seconds, config.max_backoff_seconds)
    error_limit = RateLimiter(burst=3, window=3600.0)
    while True:
        await wait_for_sound(config, capture, stats)
        stats.state = "streaming"
        stats.sessions += 1
        logger.info("sound detected (rms %.1f); starting stream", capture.meter.rms(config.blocks(1.0)))
        started = time.monotonic()
        try:
            reason = await streamer.run_session()
        except (DeviceNotFound, TimeoutError) as ex:
            reason = f"{type(ex).__name__}: {ex}"
            logger.warning("stream failed: %s", reason)
        except FatalError:
            raise
        except Exception as ex:
            reason = f"{type(ex).__name__}: {ex}"
            if error_limit.allow():
                logger.exception("stream failed")
            else:
                logger.warning("stream failed: %s", reason)
        stats.touch()
        if reason == "silence" or time.monotonic() - started >= config.healthy_session_seconds:
            backoff.reset()
            continue
        stats.failures += 1
        stats.state = "backoff"
        delay = backoff.next()
        logger.info("retrying in %.1fs", delay)
        await asyncio.sleep(delay)


async def run(config, watchdog):
    stats = Stats()
    capture = Capture(config)
    streamer = Streamer(config, capture, stats)
    tasks = [
        asyncio.create_task(capture.run(), name="capture"),
        asyncio.create_task(heartbeat(config, capture, stats, watchdog), name="heartbeat"),
        asyncio.create_task(stream_forever(config, capture, streamer, stats), name="main"),
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        raise FatalError(f"{next(iter(done)).get_name()} task exited")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.wait(tasks, timeout=config.stop_timeout_seconds * 2)


async def amain(config):
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(log_loop_error)
    task = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    watchdog = Watchdog(config.watchdog_seconds)
    watchdog.start()
    try:
        await run(config, watchdog)
    except asyncio.CancelledError:
        logger.info("shutting down")
    except FatalError as ex:
        die(f"fatal: {ex}", code=1, dump_stacks=False)


@click.command()
@click.option("--gain", default=1.0, type=float, help="Gain for recording")
@click.option("--identifier", default="", help="Identifier (MAC) of the AirPlay receiver")
@click.option("--device", default="default", help="ALSA capture device")
@click.option("--threshold", default=5.0, type=float, help="RMS level (int16 units) that counts as sound")
@click.option("--silence-seconds", default=10.0, type=float, help="Seconds of silence before stopping")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(log_level, **kwargs):
    logging.basicConfig(
        level=log_level.upper(),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    if logging.getLogger().level > logging.DEBUG:
        for name in ("pyatv", "zeroconf"):
            logging.getLogger(name).setLevel(logging.WARNING)
    faulthandler.enable()
    config = Config(**kwargs)
    logger.info("starting with %s", config)
    asyncio.run(amain(config))


if __name__ == "__main__":
    main(auto_envvar_prefix="RASPI_AIRPLAY")
