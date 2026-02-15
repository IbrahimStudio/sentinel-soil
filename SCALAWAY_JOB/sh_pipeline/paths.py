# sentinel_soil/pipeline/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobPaths:
    # local
    local_root: Path
    local_intermediate_root: Path   # where extract writes (local)
    local_features_root: Path       # where features writes (local)
    local_logs_dir: Path

    # object storage
    obj_intermediate_prefix: str
    obj_features_prefix: str
    obj_logs_prefix: str
    obj_manifest_key: str


def build_job_paths(job_id: str, *, tmp_root: Path = Path("/tmp/sentinel_soil")) -> JobPaths:
    """
    Creates a deterministic local folder layout for a single job, and the matching
    object storage keys/prefixes as required:

      intermediate/<job_id>/
      features/<job_id>/
      logs/<job_id>/
      manifests/<job_id>.json
    """
    job_id = str(job_id).strip()
    if not job_id:
        raise ValueError("job_id is empty")

    local_root = tmp_root / job_id
    local_intermediate_root = local_root / "intermediate"
    local_features_root = local_root / "features"
    local_logs_dir = local_root / "logs"

    # required remote structure
    obj_intermediate_prefix = f"intermediate/{job_id}/"
    obj_features_prefix = f"features/{job_id}/"
    obj_logs_prefix = f"logs/{job_id}/"
    obj_manifest_key = f"manifests/{job_id}.json"

    return JobPaths(
        local_root=local_root,
        local_intermediate_root=local_intermediate_root,
        local_features_root=local_features_root,
        local_logs_dir=local_logs_dir,
        obj_intermediate_prefix=obj_intermediate_prefix,
        obj_features_prefix=obj_features_prefix,
        obj_logs_prefix=obj_logs_prefix,
        obj_manifest_key=obj_manifest_key,
    )
