#!/usr/bin/env python3
import asyncio
import asyncio.subprocess as asp
import logging
import sys
import time
from asyncio.exceptions import CancelledError

import click
import numpy as np
import pyatv
from pyatv.const import Protocol
from pyatv.interface import MediaMetadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RollingVolume:
    def __init__(self, sample_rate=48000, channels=2, time_period=10):
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_size = sample_rate * channels * time_period
        self.current_window = np.zeros(self.window_size, dtype=np.int16)
        self.current_index = 0

    def add_samples(self, samples):
        for sample in samples:
            self.current_window[self.current_index] = sample
            self.current_index += 1
            if self.current_index == self.window_size:
                self.current_index = 0

    def get_volume(self):
        return np.sqrt(np.mean(np.square(self.current_window)))


class Recorder(asyncio.StreamReader):
    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain
        self.process = None
        self.volume = RollingVolume()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    async def start(self):
        self.process = await asp.create_subprocess_shell(
            f"rec -q -t wav - vol {self.gain}",
            stdout=asp.PIPE,
        )

    async def read(self, n=-1):
        if self.process is None:
            raise Exception("Recorder not started")
        buffer = await self.process.stdout.read(n)
        samples = np.frombuffer(buffer, dtype=np.int16)
        self.volume.add_samples(samples)
        return buffer

    async def stop(self):
        if self.process is not None:
            self.process.kill()
            await self.process.communicate()

    def get_volume(self):
        return self.volume.get_volume()


async def stream(recorder, identifier=""):
    """Stream audio from recorder to Apple TV."""
    try:
        logger.debug("Discovering device on network...")
        loop = asyncio.get_event_loop()
        atvs = await pyatv.scan(
            loop,
            identifier=identifier,
            protocol={Protocol.AirPlay},
            timeout=10,
        )
        if not atvs:
            logger.debug("Device not found", file=sys.stderr)
            return

        conf = atvs[0]

        logger.debug(f"Connecting to {conf.address}")
        atv = await pyatv.connect(conf, loop)

        metadata = MediaMetadata(
            artist="Raspberry PI", title="Streaming from Raspberry PI"
        )
        await atv.stream.stream_file(recorder, metadata)
    except CancelledError:
        logger.debug("Cancelled stream")
    finally:
        await asyncio.gather(*atv.close())


async def wait_for_volume(
    recorder,
    volume_conditional_callback=None,
    interval=1,
    delay=None,
    drain_buffer=False,
    buffer_size=4096,
):
    start = time.time()

    while True:
        if drain_buffer:
            drain_start = time.time()
            while time.time() - drain_start < interval:
                await recorder.read(buffer_size)
        else:
            await asyncio.sleep(interval)

        volume = recorder.get_volume()
        logger.debug(f"Volume: {volume}")
        if (delay is None or time.time() - start > delay) and (
            volume_conditional_callback is None or volume_conditional_callback(volume)
        ):
            break


async def main_loop(gain=1.0, identifier="", threshold=5):
    while True:
        async with Recorder(gain=gain) as recorder:
            await asyncio.wait_for(
                wait_for_volume(
                    recorder,
                    volume_conditional_callback=lambda volume: volume > threshold,
                    drain_buffer=True,
                ),
                timeout=None,
            )

        logger.debug("Starting stream")
        async with Recorder(gain=gain) as recorder:
            stream_teaks = asyncio.create_task(stream(recorder, identifier))
            stop_task = asyncio.create_task(
                wait_for_volume(
                    recorder,
                    volume_conditional_callback=lambda volume: volume < threshold,
                    delay=10,
                )
            )

            _, pending = await asyncio.wait(
                [stop_task, stream_teaks], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
        logger.debug("Stream stopped")


@click.command()
@click.option("--gain", default=1.0, type=float, help="Gain for recording")
@click.option("--identifier", default="", help="MAC Identifier of Apple TV")
@click.option("--threshold", default=5, help="Volume Threshold")
def main(*args):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_loop(**args))
    loop.run_forever()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.setLevel(logging.DEBUG)
    main(auto_envvar_prefix="RASPI_AIRPLAY")
