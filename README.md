# RaspiAirplay

Streams a Raspberry Pi's line-in (HiFiBerry DAC+ ADC Pro) to an AirPlay receiver whenever sound is detected, and stops after a period of silence.

## How it works

- One long-lived `sox` process captures 32-bit audio from ALSA at 44.1 kHz, applies gain, and dithers to 16-bit.
- Audio lands in a 30 s ring buffer; a vectorized RMS meter decides when to start and stop streaming.
- pyatv streams the ring buffer over RAOP. Underruns play silence until the preroll refills, and a receiver that falls behind skips ahead.
- Every network step has a timeout, failures back off exponentially, and TCP keepalives on the RTSP socket detect a receiver that vanished mid-stream.
- A watchdog thread and a progress check exit the process if the event loop hangs; Docker restarts it.
- A status line (levels, peak, clipping, clock drift, memory, fds, tasks) is logged every 5 minutes.

## Setup

[SETUP.md](SETUP.md) takes a fresh Pi from SD card to running, and `host/status.sh` checks a Pi against it. To run just the app:

```sh
cp sample.env .env   # set RASPI_AIRPLAY_IDENTIFIER to the receiver's MAC
docker compose pull  # or: docker compose build
docker compose up -d
journalctl -t airplay -f
```

Options (CLI flag or `RASPI_AIRPLAY_*` env var):

| Option | Default | |
| --- | --- | --- |
| `--identifier` | first receiver found | Receiver MAC, as shown by `atvremote scan` |
| `--gain` | 1.0 | Linear gain applied before dithering |
| `--device` | `default` | ALSA capture device |
| `--threshold` | 5.0 | RMS level (int16 units) that counts as sound |
| `--silence-seconds` | 10 | Silence before the stream stops |
| `--log-level` | INFO | |

## Tests

```sh
docker build --target test -t airplay:test .
docker run --rm -v "$PWD":/app -w /app airplay:test pytest                 # unit, capture, process
docker run --rm -e SOAK_SECONDS=600 -v "$PWD":/app -w /app airplay:test pytest -m soak -s
uv run --no-project --with pytest==9.1.1 --with pytest-timeout --with pytest-asyncio==1.4.0 --with numpy \
  pytest -m e2e -s e2e/                                                    # needs docker
```

The end-to-end tests run the app against a minimal RAOP receiver (`e2e/raop_receiver.py`) in Docker and check the audio arrives bit-exact, then inject faults: receiver killed, frozen, disconnected, and moved to a new IP.

CI runs everything on an arm64 runner and pushes `jvenberg/airplay` from `main`.
