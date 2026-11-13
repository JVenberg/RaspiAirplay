"""Test doubles for the recorder process and pyatv."""

import asyncio
import contextlib
import ipaddress
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import stream

FAKE_REC = str(Path(__file__).with_name("fake_rec.py"))


class Recorder:
    """Writes the fake recorder's control file."""

    def __init__(self, path):
        self.path = path
        self.set(mode="silence")

    def set(self, **control):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(control))
        tmp.replace(self.path)

    @property
    def starts(self):
        try:
            return Path(f"{self.path}.starts").read_text().split()
        except FileNotFoundError:
            return []

    @property
    def cmd(self):
        return (sys.executable, FAKE_REC, str(self.path))


async def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(interval)


class FakeSocket:
    def __init__(self):
        self.options = {}

    def setsockopt(self, level, option, value):
        self.options[option] = value


class FakeTransport:
    def __init__(self):
        self.socket = FakeSocket()

    def get_extra_info(self, name):
        return self.socket if name == "socket" else None


class FakeReceiver:
    """Scriptable replacement for pyatv's scan, connect, stream_file and close.

    stream behaviours: play, hang_setup, stall, raise, end, disconnect, zombie.
    close behaviours: ok, hang, raise.
    """

    LATENCY_FRAMES = 4410

    def __init__(self, address="10.0.0.5"):
        self.address = ipaddress.ip_address(address)
        self.scan = "found"
        self.connect = "ok"
        self.stream = "play"
        self.close = "ok"
        self.stream_after = 0.3
        self.speed = 1.0
        self.keep_frames = True
        self.released = False
        self.scans = []
        self.scan_times = []
        self.connects = 0
        self.confs = []
        self.closes = 0
        self.active = 0
        self.received = []
        self.sessions = []

    def install(self, monkeypatch):
        monkeypatch.setattr(stream.pyatv, "scan", self._scan)
        monkeypatch.setattr(stream.pyatv, "connect", self._connect)

    async def _scan(self, loop, identifier=None, protocol=None, hosts=None, timeout=5):
        self.scans.append(hosts)
        self.scan_times.append(time.monotonic())
        if hosts and str(self.address) not in hosts:
            return []
        if self.scan == "hang":
            await asyncio.Event().wait()
        if self.scan == "raise":
            raise OSError("network unreachable")
        if self.scan == "none":
            await asyncio.sleep(min(timeout, 0.05))
            return []
        services = [SimpleNamespace(properties={"md": "0,1,2"})]
        return [SimpleNamespace(name="Fake", address=self.address, identifier="AA:BB", services=services)]

    async def _connect(self, conf, loop):
        self.connects += 1
        self.confs.append(conf)
        if self.connect == "hang":
            await asyncio.Event().wait()
        if self.connect == "raise":
            raise ConnectionRefusedError("refused")
        transport = FakeTransport()
        connection = SimpleNamespace(transport=transport)
        client = SimpleNamespace(rtsp=SimpleNamespace(connection=connection))
        manager = SimpleNamespace(stream_client=None)
        session = SimpleNamespace(
            manager=manager, client=client, connection=connection, socket=transport.socket, frames=[]
        )
        self.sessions.append(session)
        raop_stream = SimpleNamespace(playback_manager=manager)
        return SimpleNamespace(
            stream=SimpleNamespace(
                stream_file=lambda source: self._stream_file(session, source),
                get=lambda protocol: raop_stream if protocol == stream.Protocol.RAOP else None,
            ),
            close=lambda: self._close(),
        )

    async def _stream_file(self, session, source):
        self.active += 1
        try:
            if self.stream == "hang_setup":
                await asyncio.Event().wait()
            if self.stream == "raise":
                raise RuntimeError("protocol error")
            session.manager.stream_client = session.client
            source = await stream.raop.open_source(source, 44100, 2, 2)
            session.source = source
            started = time.monotonic()
            sent = padding = 0
            while padding < self.LATENCY_FRAMES:
                elapsed = time.monotonic() - started
                if self.stream == "stall" and elapsed > self.stream_after:
                    await asyncio.Event().wait()
                if self.stream == "end" and elapsed > self.stream_after:
                    return
                if self.stream == "disconnect" and elapsed > self.stream_after:
                    session.connection.transport = None
                if self.stream == "zombie" and elapsed > self.stream_after:
                    while not self.released:
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.sleep(0.01)
                frames = await source.readframes(352)
                if frames:
                    if self.keep_frames:
                        session.frames.append(frames)
                else:
                    padding += 352
                sent += 352
                delay = started + sent / (44100 * self.speed) - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            await source.close()
            session.manager.stream_client = None
            self.active -= 1

    def _close(self):
        self.closes += 1

        async def close():
            if self.close == "hang":
                await asyncio.Event().wait()
            if self.close == "raise":
                raise OSError("close failed")

        return {asyncio.ensure_future(close())}


def decode_frames(chunks):
    """Big-endian wire bytes back to an (n, 2) int16 array."""
    if not chunks:
        return np.zeros((0, 2), dtype=np.int16)
    return np.frombuffer(b"".join(chunks), dtype=">i2").astype(np.int16).reshape(-1, 2)


def ramp_index(frames):
    """Recovers the fake recorder's frame counter (mod 65536) from ramp samples."""
    return (frames[:, 0].astype(np.int64) + 32768) % 65536


class App:
    """Runs the capture and (optionally) the streaming loop as the real program does."""

    def __init__(self, config, stream_loop=True):
        self.config = config
        self.stats = stream.Stats()
        self.capture = stream.Capture(config)
        self.streamer = stream.Streamer(config, self.capture, self.stats)
        self.stream_loop = stream_loop
        self.tasks = []

    async def __aenter__(self):
        self.tasks.append(asyncio.create_task(self.capture.run(), name="capture"))
        if self.stream_loop:
            main = stream.stream_forever(self.config, self.capture, self.streamer, self.stats)
            self.tasks.append(asyncio.create_task(main, name="main"))
        return self

    def check(self):
        for task in self.tasks:
            if task.done():
                raise AssertionError(f"{task.get_name()} exited") from task.exception()

    async def __aexit__(self, *exc):
        for task in self.tasks:
            task.cancel()
        _, pending = await asyncio.wait(self.tasks, timeout=10)
        assert not pending, "tasks did not stop"
