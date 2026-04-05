# syntax=docker/dockerfile:1
# Job 1v2 – Process API Raster Ingestion
# Build context: SCALAWAY_JOB/  (docker build -f ingestion/process_api/Dockerfile .)
#
# Leaner than ingestion:stats-api — no sentinelhub SDK.
# Geo stack (GDAL, proj) retained because rasterio needs it to parse Process API TIFFs.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- System dependencies (geo stack: rasterio needs GDAL + proj) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        libexpat1 \
        zlib1g \
        liblzma5 \
        libsqlite3-0 \
        libcurl4 \
        gdal-bin \
        libgdal-dev \
        proj-bin \
        libproj-dev \
    ; \
    rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies ---
COPY ingestion/process_api/pyproject.toml /app/
RUN uv pip install --system $(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(' '.join(d['project']['dependencies']))")

# --- Copy job code ---
COPY ingestion/process_api/ /app/

# At runtime, mount:
#   gabri_filters.xlsx      → /data/input.xlsx        (read-only)
#   filter_config.json      → /data/filter_config.json (read-only)
#   evalscript_process_api  → /data/evalscript.js      (read-only)

# --- Non-root user ---
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "main.py", "--help"]
