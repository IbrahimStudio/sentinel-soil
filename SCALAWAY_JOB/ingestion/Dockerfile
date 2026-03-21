# syntax=docker/dockerfile:1
# Job 1 – Remote Sensing Data Acquisition
# Build context: SCALAWAY_JOB/  (docker build -f ingestion/Dockerfile .)

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- System dependencies (geo stack required by rasterio + sentinelhub) ---
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

# --- Install Python dependencies (cached layer) ---
COPY ingestion/pyproject.toml /app/
RUN uv pip install --system -r pyproject.toml || uv pip install --system $(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(' '.join(d['project']['dependencies']))")

# --- Copy job code ---
COPY ingestion/ /app/
COPY configs/ /app/configs/

# --- Non-root user ---
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]
# Override in Scaleway Jobs with actual args:
#   python main.py --xlsx /data/gabri_filters.xlsx --evalscript sh_statistics/evalscripts/only_scl.js --time_window 60 --workers 8
CMD ["python", "main.py", "--help"]
