# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- System dependencies (robust geo stack + runtime libs) ---
# Notes:
# - rasterio wheels often need runtime libs (like libexpat1).
# - installing gdal/proj ensures compatibility when wheels are not fully self-contained.
# - tini helps correct PID1 behavior (signals, zombie reaping) in job runners.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        # rasterio runtime deps commonly needed by wheels
        libexpat1 \
        zlib1g \
        liblzma5 \
        libsqlite3-0 \
        libcurl4 \
        # full geo stack (safe option)
        gdal-bin \
        libgdal-dev \
        proj-bin \
        libproj-dev \
    ; \
    rm -rf /var/lib/apt/lists/*

# --- Dependencies first (best caching) ---
COPY pyproject.toml uv.lock /app/

# Create venv + install exactly locked deps
RUN uv sync --frozen --no-dev

# --- Copy app code ---
COPY . /app

# (Optional) If your project needs to be installed into the venv:
# - Keep this if you import via package name / rely on entry points
# - Remove if your imports are purely via /app on PYTHONPATH
RUN uv sync --frozen --no-dev

# --- Non-root user (recommended) ---
RUN useradd -m -u 10001 appuser && \
    chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/usr/bin/tini","--"]

# Default command (override in Scaleway Jobs)
CMD ["uv", "run", "python", "-m", "pipeline.worker", "--help"]
