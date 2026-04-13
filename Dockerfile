# Hippo 🦛 — Lightweight local LLM manager
# Multi-stage build for minimal image

# Stage 1: Build llama-cpp-python
FROM python:3.14-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml ./
COPY hippo/ hippo/

RUN pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin/hippo /usr/local/bin/hippo
COPY hippo/ /app/hippo/

ENV HIPPO_MODELS_DIR=/models
EXPOSE 8000

ENTRYPOINT ["hippo", "serve", "--host", "0.0.0.0", "--port", "8000"]
