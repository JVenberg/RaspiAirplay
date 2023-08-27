FROM python:3.11-alpine

RUN apk add sox gcc musl-dev build-base linux-headers libffi-dev rust cargo openssl-dev git

RUN pip install setuptools-rust && \
    pip install pyatv && \
    pip install numpy && \
    pip install click

COPY stream.py /app/stream.py
RUN chmod +x /app/stream.py
