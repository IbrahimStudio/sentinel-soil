# syntax=docker/dockerfile:1
# Job 3 – Model Training & Evaluation
# Build context: SCALAWAY_JOB/  (docker build -f training/Dockerfile .)

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- System dependencies (proj-bin for pyproj spatial blocking) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        proj-bin \
        libproj-dev \
    ; \
    rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies (cached layer) ---
COPY training/pyproject.toml /app/
RUN uv pip install --system $(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(' '.join(d['project']['dependencies']))")

# --- Copy job code ---
COPY training/ /app/

# reports/ dir is mounted or written locally at runtime

# --- Non-root user ---
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "main.py", "--help"]
