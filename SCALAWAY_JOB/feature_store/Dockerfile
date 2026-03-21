# syntax=docker/dockerfile:1
# Job 2 – Data Enrichment & Feature Consolidation
# Build context: SCALAWAY_JOB/  (docker build -f feature_store/Dockerfile .)

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- System dependencies (minimal — no geo stack needed) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    ; \
    rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies (cached layer) ---
COPY feature_store/pyproject.toml /app/
RUN uv pip install --system $(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(' '.join(d['project']['dependencies']))")

# --- Copy job code ---
COPY feature_store/ /app/

# Mount gabri_filters.xlsx at /data/gabri_filters.xlsx at runtime

# --- Non-root user ---
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "main.py", "--help"]
