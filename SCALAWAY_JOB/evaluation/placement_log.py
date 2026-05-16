"""
Placement audit log — written at inference time, read at Phase 4 validation.

Every placement run produces one PlacementAuditLog uploaded to S3 under
  evaluation/placement-logs/{field_id}/{run_id}.json

The schema is designed so that when sensor measurements arrive, you can:
  1. Match each selected tile to its nearest sensor (by lat/lon).
  2. Correlate predicted texture features against measured moisture per tile.
  3. Compare cross-method: which strategy led to the best zone coverage?

Nothing in this file depends on MLflow — it can be imported independently.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class TileRecord:
    tile_id: str
    lat: float
    lon: float
    # Named predicted features — e.g. {"clay": 32.1, "silt": 41.5, ...}
    # Keeping them named (not positional) makes the log self-documenting and
    # forward-compatible when the feature schema changes across pipeline versions.
    features: dict[str, float]
    # Clustering result (None for maximin which skips clustering)
    cluster_id: int | None
    distance_to_centroid: float | None
    # Placement scoring
    placement_score: float
    rank: int
    selected: bool


@dataclass
class PlacementAuditLog:
    field_id: str
    pipeline_id: str                    # e.g. "texture_v1"
    model_versions: dict[str, str]      # {"clay": "run_20260503_125342/...", ...}
    placement_method: str               # "representative" | "maximin" | "stratified"
    n_sensors: int
    config: dict[str, Any]             # full placement config for reproducibility
    tiles: list[TileRecord]
    # Auto-populated
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Clustering diagnostics — None for non-clustering methods
    n_clusters: int | None = None
    silhouette_score: float | None = None
    inertia: float | None = None

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def selected_tile_ids(self) -> list[str]:
        return [t.tile_id for t in self.tiles if t.selected]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_tiles"] = self.n_tiles
        d["selected_tile_ids"] = self.selected_tile_ids
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def upload_to_s3(
    log: PlacementAuditLog,
    *,
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> str:
    """Serialize and upload the audit log. Returns the S3 key."""
    key = f"{prefix.rstrip('/')}/{log.field_id}/{log.run_id}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=log.to_json().encode("utf-8"),
        ContentType="application/json",
    )
    return key
