FROM python:3.13-slim-trixie AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends sox libsox-fmt-alsa \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

FROM python:3.13-slim-trixie AS build
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/
RUN uv venv --python /usr/local/bin/python3 /venv \
    && uv pip install --python /venv/bin/python --no-cache --require-hashes -r /tmp/requirements.txt

FROM build AS build-dev
COPY requirements-dev.txt /tmp/
RUN uv pip install --python /venv/bin/python --no-cache -r /tmp/requirements-dev.txt

FROM base AS test
COPY --from=build-dev /venv /venv
WORKDIR /app
COPY stream.py pytest.ini /app/
COPY tests /app/tests
CMD ["pytest"]

FROM base AS runtime
COPY --from=build /venv /venv
COPY stream.py /app/stream.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD test $(( $(date +%s) - $(stat -c %Y /tmp/heartbeat) )) -lt 60
CMD ["python", "/app/stream.py"]
