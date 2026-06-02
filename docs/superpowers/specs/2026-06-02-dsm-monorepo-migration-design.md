# DSM Monorepo Migration — Design Spec

**Date:** 2026-06-02  
**Status:** Approved  
**Scope:** Merge `sentinel-soil` and `soil-moisture` into a single `DSM` monorepo, preserving full git history for both, with a temporary dual-push transition period.

---

## 1. Repository Structure

```
DSM/
  sentinel-soil/           ← full history rewritten under this prefix
    SCALAWAY_JOB/
    AIRFLOW_JOB/
    DSM_WEBAPP/
    soil-texture-inference/
    CLAUDE.md
    ...
  soil-moisture/           ← full history rewritten under this prefix
    moisture_zoning/
    configs/
    CLAUDE.md
    ...
  infra/                   ← new; placeholder for MLflow service (next phase)
    docker-compose.yml
    .env.example
  CLAUDE.md                ← thin orientation doc (see §4)
  README.md
  .gitignore               ← covers infra/.env and editor files
```

Each sub-project remains self-contained: its own `pyproject.toml`, `.venv`, `Dockerfile`, and `CLAUDE.md`. No shared Python packages are introduced in this migration.

---

## 2. Migration Mechanics

Tool: `git filter-repo` (preferred over `git filter-branch` — faster, safer, no backup refs needed).

### Step-by-step

```bash
# 1. Fresh clones (don't touch working copies)
git clone https://github.com/<user>/sentinel-soil sentinel-soil-migration
git clone https://github.com/<user>/soil-moisture soil-moisture-migration

# 2. Rewrite each repo so every file path is prefixed by its subdirectory name
cd sentinel-soil-migration
git filter-repo --to-subdirectory-filter sentinel-soil/

cd ../soil-moisture-migration
git filter-repo --to-subdirectory-filter soil-moisture/

# 3. Create the new DSM repo
mkdir DSM && cd DSM
git init
git commit --allow-empty -m "chore: init DSM monorepo"

# 4. Merge both histories
git remote add sentinel-soil ../sentinel-soil-migration
git fetch sentinel-soil
git merge --allow-unrelated-histories sentinel-soil/main -m "feat: import sentinel-soil history"

git remote add soil-moisture ../soil-moisture-migration
git fetch soil-moisture
git merge --allow-unrelated-histories soil-moisture/main -m "feat: import soil-moisture history"

# 5. Remove migration remotes
git remote remove sentinel-soil
git remote remove soil-moisture

# 6. Add infra/ scaffold and top-level CLAUDE.md
mkdir infra
# ... create placeholder files ...
git add infra/ CLAUDE.md README.md .gitignore
git commit -m "chore: add infra/ placeholder and top-level CLAUDE.md"

# 7. Push to new DSM remote
git remote add origin https://github.com/<user>/dsm
git push -u origin main
```

---

## 3. Transition Period

Old repos remain alive on GitHub until all dependents (CI, Docker Hub, team members) have migrated.

### Dual-push setup

```bash
# Add old repos as secondary remotes (run once after migration)
git remote add origin-sentinel https://github.com/<user>/sentinel-soil
git remote add origin-moisture https://github.com/<user>/soil-moisture
```

### Push subtrees to old repos

```bash
# Run from DSM/ root when you want to sync
git subtree push --prefix=sentinel-soil origin-sentinel main
git subtree push --prefix=soil-moisture origin-moisture main
```

### Archive checklist

Before archiving each old repo, confirm:
- [ ] No active CI/CD pipelines pointing at the old remote
- [ ] Docker images are built from DSM or a registry (not the old repo)
- [ ] Any collaborators have switched to DSM
- [ ] Old repo is archived on GitHub (Settings → Archive repository)
- [ ] Secondary remotes removed from local DSM clone

---

## 4. Build Contexts and Import Paths

| Concern | Impact | Action required |
|---------|--------|-----------------|
| Docker Compose run from sub-project dir | None | Run Compose from `sentinel-soil/` or `soil-moisture/` as today |
| Docker Compose run from monorepo root | Build context path changes | Adjust `build.context` from `.` to `../sentinel-soil` (future `infra/` Compose) |
| Python imports | None | Both projects use isolated `.venv`s; no cross-project imports exist |
| Runtime paths (`FEATURES_PATH`, S3 keys) | None | All env-var injected; not hardcoded to old repo root |
| `.gitignore` | Minor | Each sub-project keeps its own; root adds `infra/.env` and editor files |

---

## 5. Top-level `CLAUDE.md`

Thin orientation document — does not duplicate sub-project docs:

```markdown
# DSM Monorepo

Two sub-projects live here. Enter the relevant directory before running any commands.

| Sub-project      | What it does                                                         | Details                   |
|------------------|----------------------------------------------------------------------|---------------------------|
| sentinel-soil/   | Sentinel-2 → soil texture (Statistics API + Process API pipelines, ML training) | sentinel-soil/CLAUDE.md   |
| soil-moisture/   | OPTRAM-based soil moisture zoning                                    | soil-moisture/CLAUDE.md   |
| infra/           | Shared infrastructure (MLflow server, future services)               | infra/docker-compose.yml  |
```

---

## 6. Out of Scope

- Shared Python library (`dsm-common`): not introduced in this migration; both projects stay independent
- MLflow Docker Compose service: placeholder `infra/docker-compose.yml` is created but left empty; this is the next design phase
- Merging `DSM_WEBAPP/` or `soil-texture-inference/` as top-level sub-projects: they remain inside `sentinel-soil/` for now

---

## 7. Success Criteria

- `git log sentinel-soil/` in DSM shows full sentinel-soil commit history
- `git log soil-moisture/` in DSM shows full soil-moisture commit history
- Both projects build and run identically from their new paths
- Old repos still accept pushes via `git subtree push` during transition
