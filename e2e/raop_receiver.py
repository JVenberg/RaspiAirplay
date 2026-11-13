"""Minimal AirPlay 1 (RAOP) receiver for e2e tests: accepts pyatv's uncompressed L16 stream and records it.

Audio is appended to <out>/audio.raw as little-endian int16 stereo, in RTP sequence order as received.
"""

import argparse
import asyncio
import plistlib
import socket
import struct
import time
from pathlib import Path

import numpy as np
from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

NAME = "E2E Receiver"
MAC = "AABBCCDDEEFF"
RTSP_PORT = 5000
AUDIO_PORT, CONTROL_PORT, TIMING_PORT = 6000, 6001, 6002
TXT = {
    "txtvers": "1",
    "ch": "2",
    "cn": "0,1",
    "et": "0",
    "sr": "44100",
    "ss": "16",
    "tp": "UDP",
    "vn": "65537",
    "md": "0,1,2",
    "am": "E2EReceiver",
}
# pyatv takes the colon identifier from AirPlay; features avoid bits 38/48 (AirPlay 2).
AIRPLAY_TXT = {
    "deviceid": ":".join(MAC[i : i + 2] for i in range(0, len(MAC), 2)),
    "features": "0x5A7FFFF7,0x1E",
    "model": "E2EReceiver",
    "srcvers": "366.0",
}


def log(message, *args):
    print(f"{time.strftime('%H:%M:%S')} receiver: {message % args}", flush=True)


class Session:
    def __init__(self, number):
        self.number = number
        self.expected = None
        self.packets = self.gaps = self.late = 0
        self.last_packet = time.monotonic()


class Receiver:
    def __init__(self, out, idle_timeout):
        self.out = out
        self.idle_timeout = idle_timeout
        self.audio = (out / "audio.raw").open("ab")
        self.sessions = 0
        self.session = None
        self.connection = None

    def start_session(self, connection):
        if self.connection and self.connection is not connection:
            log("preempting previous session")
            self.connection.close()
        self.sessions += 1
        self.session = Session(self.sessions)
        self.connection = connection
        log("session %d started", self.sessions)

    def end_session(self, connection, why):
        if self.connection is not connection or self.session is None:
            return
        s = self.session
        log("session %d ended (%s): packets=%d gaps=%d late=%d", s.number, why, s.packets, s.gaps, s.late)
        self.session = self.connection = None

    def audio_packet(self, data):
        s = self.session
        if s is None or len(data) <= 12:
            return
        seq = struct.unpack_from(">H", data, 2)[0]
        if s.expected is not None and seq != s.expected:
            ahead = (seq - s.expected) % 65536
            if ahead >= 32768:
                s.late += 1
                return
            s.gaps += ahead
        s.expected = (seq + 1) % 65536
        s.packets += 1
        s.last_packet = time.monotonic()
        self.audio.write(np.frombuffer(data, dtype=">i2", offset=12).astype("<i2").tobytes())
        self.audio.flush()

    async def watch_idle(self):
        while True:
            await asyncio.sleep(0.5)
            s, connection = self.session, self.connection
            if s and s.packets and time.monotonic() - s.last_packet > self.idle_timeout:
                self.end_session(connection, f"no audio for {self.idle_timeout:.0f}s")
                connection.close()


class Rtsp(asyncio.Protocol):
    def __init__(self, receiver):
        self.receiver = receiver
        self.buffer = b""
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        log("rtsp connection from %s", transport.get_extra_info("peername")[0])

    def connection_lost(self, exc):
        self.receiver.end_session(self, f"connection lost: {exc}" if exc else "connection closed")

    def close(self):
        self.transport.close()

    def data_received(self, data):
        self.buffer += data
        while b"\r\n\r\n" in self.buffer:
            head, rest = self.buffer.split(b"\r\n\r\n", 1)
            lines = head.decode(errors="replace").split("\r\n")
            pairs = (line.split(":", 1) for line in lines[1:] if ":" in line)
            headers = {k.strip().lower(): v.strip() for k, v in pairs}
            length = int(headers.get("content-length", 0))
            if len(rest) < length:
                return
            body, self.buffer = rest[:length], rest[length:]
            method, uri, protocol = lines[0].split(" ", 2)
            self.handle(method, uri, protocol, headers, body)

    def handle(self, method, uri, protocol, headers, body):
        extra, payload = {}, b""
        if method == "GET" and uri == "/info":
            extra["Content-Type"] = "application/x-apple-binary-plist"
            info = {"statusFlags": 4, "name": NAME, "initialVolume": -15.0}
            payload = plistlib.dumps(info, fmt=plistlib.FMT_BINARY)
        elif method == "OPTIONS":
            extra["Public"] = "ANNOUNCE, SETUP, RECORD, PAUSE, FLUSH, TEARDOWN, OPTIONS, GET_PARAMETER, SET_PARAMETER"
        elif method == "ANNOUNCE":
            self.receiver.start_session(self)
            log("announce: %s", " ".join(line for line in body.decode().split("\r\n") if line.startswith("a=")))
        elif method == "SETUP":
            ports = f"server_port={AUDIO_PORT};control_port={CONTROL_PORT};timing_port={TIMING_PORT}"
            extra["Transport"] = f"RTP/AVP/UDP;unicast;mode=record;{ports}"
            extra["Session"] = "1"
            extra["Audio-Jack-Status"] = "connected; type=analog"
        elif method == "RECORD":
            extra["Audio-Latency"] = "11025"
        elif method == "SET_PARAMETER" and body.startswith(b"progress:"):
            log("resetting on progress metadata, like a Bose soundbar")
            self.transport.abort()
            return
        elif method == "SET_PARAMETER":
            log("set_parameter: %s", body.decode(errors="replace").strip()[:80])
        elif method == "TEARDOWN":
            self.receiver.end_session(self, "teardown")
            extra["Connection"] = "close"
        elif method not in ("POST", "FLUSH", "GET_PARAMETER"):
            log("unsupported %s %s", method, uri)
        status = "RTSP/1.0" if protocol.startswith("RTSP") else protocol
        lines = [f"{status} 200 OK", f"CSeq: {headers.get('cseq', '0')}", "Server: AirTunes/366.0"]
        lines += [f"{k}: {v}" for k, v in extra.items()]
        lines.append(f"Content-Length: {len(payload)}")
        self.transport.write(("\r\n".join(lines) + "\r\n\r\n").encode() + payload)


class Audio(asyncio.DatagramProtocol):
    def __init__(self, receiver):
        self.receiver = receiver

    def datagram_received(self, data, addr):
        self.receiver.audio_packet(data)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/out"))
    parser.add_argument("--idle-timeout", type=float, default=10.0)
    args = parser.parse_args()
    ip = socket.gethostbyname(socket.gethostname())
    loop = asyncio.get_running_loop()
    receiver = Receiver(args.out, args.idle_timeout)
    server = await loop.create_server(lambda: Rtsp(receiver), "0.0.0.0", RTSP_PORT)
    await loop.create_datagram_endpoint(lambda: Audio(receiver), local_addr=("0.0.0.0", AUDIO_PORT))
    for port in (CONTROL_PORT, TIMING_PORT):
        await loop.create_datagram_endpoint(asyncio.DatagramProtocol, local_addr=("0.0.0.0", port))
    zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only, interfaces=[ip])
    for kind, name, properties in [
        ("_raop._tcp.local.", f"{MAC}@{NAME}", TXT),
        ("_airplay._tcp.local.", NAME, AIRPLAY_TXT),
    ]:
        info = ServiceInfo(
            kind,
            f"{name}.{kind}",
            addresses=[socket.inet_aton(ip)],
            port=RTSP_PORT,
            properties=properties,
            server="e2e-receiver.local.",
        )
        await zeroconf.async_register_service(info)
    log("ready on %s:%d", ip, RTSP_PORT)
    async with server:
        await receiver.watch_idle()


if __name__ == "__main__":
    asyncio.run(main())
