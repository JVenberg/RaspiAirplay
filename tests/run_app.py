"""Runs stream.main() as a real process, by default with a fake receiver; argv: spec.json then CLI args."""

import asyncio
import functools
import json
import sys
import time
from pathlib import Path

from fakes import FakeReceiver

import stream


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    if not spec.get("real_receiver"):
        receiver = FakeReceiver()
        for key, value in spec.get("receiver", {}).items():
            setattr(receiver, key, value)
        stream.pyatv.scan = receiver._scan
        stream.pyatv.connect = receiver._connect
    overrides = spec.get("config", {})
    overrides["capture_cmd"] = tuple(overrides.get("capture_cmd", ()))
    stream.Config = functools.partial(stream.Config, **overrides)
    if spec.get("hang_main"):

        async def hang(*args):
            await asyncio.Event().wait()

        stream.stream_forever = hang
    if "block_loop_after" in spec:
        amain = stream.amain

        async def blocking_amain(config):
            asyncio.get_running_loop().call_later(spec["block_loop_after"], time.sleep, 3600)
            await amain(config)

        stream.amain = blocking_amain
    stream.main(sys.argv[2:], auto_envvar_prefix="RASPI_AIRPLAY")


if __name__ == "__main__":
    main()
