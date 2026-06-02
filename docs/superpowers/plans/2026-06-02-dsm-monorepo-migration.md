# DSM Monorepo Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `sentinel-soil` and `soil-moisture` into a single `DSM` monorepo on GitHub, preserving both git histories, with a temporary dual-push transition.

**Architecture:** Use `git filter-repo` to rewrite each repo so every file path is prefixed by its subdirectory name, then merge both into a fresh `DSM` repo via `git merge --allow-unrelated-histories`. Old repos remain alive as secondary remotes for the transition period.

**Tech Stack:** git, git-filter-repo (pip package), GitHub (`IbrahimStudio` org)

---

## Task 1: Resolve uncommitted changes in both repos

**Files:**
- Modify: `sentinel-soil` working tree (commit or discard before migration)
- Modify: `soil-moisture` working tree (commit or discard before migration)

The migration uses **fresh clones from GitHub**, so only committed state carries over. Decide now what to keep.

- [ ] **Step 1: Inspect sentinel-soil changes**

```bash
cd /home/ibrahim/Projects/sentinel-soil
git status
git diff
```

Expected: you will see modifications to `.gitignore`, `SCALAWAY_JOB/docker-compose.yaml`, `SCALAWAY_JOB/ingestion/process_api/main.py`, `SCALAWAY_JOB/ingestion/process_api/sh_clients.py` and untracked files `DSM_WEBAPP/evalscript_multitemporal.js`, `soil-texture-inference/`.

- [ ] **Step 2: Commit sentinel-soil changes you want to preserve**

Commit any modifications and untracked files that belong in the monorepo. At minimum commit the spec and plan docs added today:

```bash
cd /home/ibrahim/Projects/sentinel-soil
git add .gitignore SCALAWAY_JOB/docker-compose.yaml \
    SCALAWAY_JOB/ingestion/process_api/main.py \
    SCALAWAY_JOB/ingestion/process_api/sh_clients.py \
    DSM_WEBAPP/evalscript_multitemporal.js
# Add soil-texture-inference/ only if it has content worth keeping:
git add soil-texture-inference/
git commit -m "chore: stage pending changes before monorepo migration"
git push origin main
```

If you want to discard any file instead, run `git checkout -- <file>` or `git clean -fd <path>`.

- [ ] **Step 3: Inspect soil-moisture changes**

```bash
cd /home/ibrahim/Projects/soil-moisture
git status
git diff
```

Expected: modifications to `soil-moisture-inference/soil_moisture_inference.egg-info/` and `soil-moisture-inference/uv.lock` — these are build artifacts, safe to discard.

- [ ] **Step 4: Discard soil-moisture build artifacts and push**

```bash
cd /home/ibrahim/Projects/soil-moisture
git checkout -- .
git status
```

Expected: `nothing to commit, working tree clean`

- [ ] **Step 5: Verify both repos are clean and up to date**

```bash
git -C /home/ibrahim/Projects/sentinel-soil log --oneline -3
git -C /home/ibrahim/Projects/soil-moisture log --oneline -3
```

Expected: both show recent commits with no pending local changes.

---

## Task 2: Install git-filter-repo

**Files:** None (system tool install)

- [ ] **Step 1: Check if already installed**

```bash
git filter-repo --version
```

Expected output: `git filter-repo == 2.x.x` — if you see this, skip to Task 3.

- [ ] **Step 2: Install via pip**

```bash
pip install git-filter-repo
```

- [ ] **Step 3: Verify**

```bash
git filter-repo --version
```

Expected: `git filter-repo == 2.x.x`

---

## Task 3: Prepare migration workspace

**Files:**
- Create: `/tmp/dsm-migration/` (temp working directory)

- [ ] **Step 1: Create workspace**

```bash
mkdir -p /tmp/dsm-migration
cd /tmp/dsm-migration
ls
```

Expected: empty directory.

- [ ] **Step 2: Clone sentinel-soil fresh**

```bash
cd /tmp/dsm-migration
git clone https://github.com/IbrahimStudio/sentinel-soil.git sentinel-soil-rewrite
```

Expected: `Cloning into 'sentinel-soil-rewrite'...` completes successfully.

- [ ] **Step 3: Clone soil-moisture fresh**

```bash
cd /tmp/dsm-migration
git clone https://github.com/IbrahimStudio/soil-moisture.git soil-moisture-rewrite
```

Expected: `Cloning into 'soil-moisture-rewrite'...` completes successfully.

- [ ] **Step 4: Verify both clones have full history**

```bash
git -C /tmp/dsm-migration/sentinel-soil-rewrite log --oneline | wc -l
git -C /tmp/dsm-migration/soil-moisture-rewrite log --oneline | wc -l
```

Expected: both print a count greater than 5 (sentinel-soil has ~15 commits, soil-moisture more).

---

## Task 4: Rewrite sentinel-soil history under subdirectory prefix

**Files:**
- Modify: `/tmp/dsm-migration/sentinel-soil-rewrite/` (rewritten in place)

`git filter-repo --to-subdirectory-filter` rewrites every commit so that all file paths gain a `sentinel-soil/` prefix. After this, `README.md` in the old repo appears as `sentinel-soil/README.md` in every commit.

- [ ] **Step 1: Run filter-repo**

```bash
cd /tmp/dsm-migration/sentinel-soil-rewrite
git filter-repo --to-subdirectory-filter sentinel-soil/
```

Expected: prints progress, completes in a few seconds. No errors.

- [ ] **Step 2: Verify rewrite**

```bash
cd /tmp/dsm-migration/sentinel-soil-rewrite
git log --oneline -5
git show HEAD --stat | head -20
```

Expected: `git show` lists files like `sentinel-soil/CLAUDE.md`, `sentinel-soil/SCALAWAY_JOB/...` — the prefix is present on every file.

- [ ] **Step 3: Verify commit count is unchanged**

```bash
git -C /tmp/dsm-migration/sentinel-soil-rewrite log --oneline | wc -l
```

Expected: same count as the clone in Task 3 Step 4. History is preserved, only paths changed.

---

## Task 5: Rewrite soil-moisture history under subdirectory prefix

**Files:**
- Modify: `/tmp/dsm-migration/soil-moisture-rewrite/` (rewritten in place)

- [ ] **Step 1: Run filter-repo**

```bash
cd /tmp/dsm-migration/soil-moisture-rewrite
git filter-repo --to-subdirectory-filter soil-moisture/
```

Expected: completes without errors.

- [ ] **Step 2: Verify rewrite**

```bash
cd /tmp/dsm-migration/soil-moisture-rewrite
git show HEAD --stat | head -20
```

Expected: files listed as `soil-moisture/moisture_zoning/...`, `soil-moisture/configs/...`, etc.

- [ ] **Step 3: Verify commit count is unchanged**

```bash
git -C /tmp/dsm-migration/soil-moisture-rewrite log --oneline | wc -l
```

Expected: same count as the clone in Task 3 Step 4.

---

## Task 6: Create DSM repo and merge both histories

**Files:**
- Create: `/tmp/dsm-migration/DSM/` (new monorepo)

- [ ] **Step 1: Create the new repo**

```bash
cd /tmp/dsm-migration
mkdir DSM
cd DSM
git init
git commit --allow-empty -m "chore: init DSM monorepo"
```

Expected: `[main (root-commit) xxxxxxx] chore: init DSM monorepo`

- [ ] **Step 2: Merge sentinel-soil history**

```bash
cd /tmp/dsm-migration/DSM
git remote add sentinel-soil ../sentinel-soil-rewrite
git fetch sentinel-soil
git merge --allow-unrelated-histories sentinel-soil/main \
    -m "feat: import sentinel-soil history under sentinel-soil/"
```

Expected: merge succeeds, no conflicts. Output ends with `Fast-forward` or a merge commit message.

- [ ] **Step 3: Verify sentinel-soil files and history**

```bash
cd /tmp/dsm-migration/DSM
ls sentinel-soil/
git log --oneline sentinel-soil/CLAUDE.md | head -5
```

Expected: `ls` shows `CLAUDE.md`, `SCALAWAY_JOB/`, etc. `git log` shows commits from the original sentinel-soil history.

- [ ] **Step 4: Merge soil-moisture history**

```bash
cd /tmp/dsm-migration/DSM
git remote add soil-moisture ../soil-moisture-rewrite
git fetch soil-moisture
git merge --allow-unrelated-histories soil-moisture/main \
    -m "feat: import soil-moisture history under soil-moisture/"
```

Expected: merge succeeds, no conflicts.

- [ ] **Step 5: Verify soil-moisture files and history**

```bash
cd /tmp/dsm-migration/DSM
ls soil-moisture/
git log --oneline soil-moisture/moisture_zoning/ | head -5
```

Expected: `ls` shows `moisture_zoning/`, `configs/`, etc. `git log` shows commits from the original soil-moisture history.

- [ ] **Step 6: Remove migration remotes**

```bash
cd /tmp/dsm-migration/DSM
git remote remove sentinel-soil
git remote remove soil-moisture
git remote -v
```

Expected: no remotes listed.

---

## Task 7: Add top-level files

**Files:**
- Create: `/tmp/dsm-migration/DSM/CLAUDE.md`
- Create: `/tmp/dsm-migration/DSM/README.md`
- Create: `/tmp/dsm-migration/DSM/.gitignore`
- Create: `/tmp/dsm-migration/DSM/infra/docker-compose.yml`
- Create: `/tmp/dsm-migration/DSM/infra/.env.example`

- [ ] **Step 1: Create CLAUDE.md**

```bash
cat > /tmp/dsm-migration/DSM/CLAUDE.md << 'EOF'
# DSM Monorepo

Two sub-projects live here. Enter the relevant directory before running any commands.

| Sub-project    | What it does                                                                    | Details                 |
|----------------|---------------------------------------------------------------------------------|-------------------------|
| sentinel-soil/ | Sentinel-2 → soil texture (Statistics API + Process API pipelines, ML training) | sentinel-soil/CLAUDE.md |
| soil-moisture/ | OPTRAM-based soil moisture zoning                                               | soil-moisture/CLAUDE.md |
| infra/         | Shared infrastructure (MLflow server, future services)                          | infra/docker-compose.yml|
EOF
```

- [ ] **Step 2: Create README.md**

```bash
cat > /tmp/dsm-migration/DSM/README.md << 'EOF'
# DSM — Digital Soil Mapping

Monorepo for the DSM project family.

- **sentinel-soil/** — soil texture prediction from Sentinel-2 imagery
- **soil-moisture/** — OPTRAM-based soil moisture zoning
- **infra/** — shared infrastructure (MLflow tracking server)

See each sub-project's `CLAUDE.md` for dev commands and architecture notes.
EOF
```

- [ ] **Step 3: Create .gitignore**

```bash
cat > /tmp/dsm-migration/DSM/.gitignore << 'EOF'
# Infrastructure secrets
infra/.env

# Editor
.idea/
.vscode/
*.swp
.DS_Store
EOF
```

- [ ] **Step 4: Create infra/ placeholder files**

```bash
mkdir -p /tmp/dsm-migration/DSM/infra

cat > /tmp/dsm-migration/DSM/infra/docker-compose.yml << 'EOF'
# Placeholder — MLflow tracking server will be defined here in the next phase.
# See docs/superpowers/specs/2026-06-02-dsm-monorepo-migration-design.md §6.
version: "3.9"
services: {}
EOF

cat > /tmp/dsm-migration/DSM/infra/.env.example << 'EOF'
# Copy to infra/.env and fill in values.
# Used by the shared Docker Compose services in this directory.

# MLflow (populated in next phase)
# MLFLOW_TRACKING_URI=
# MLFLOW_ARTIFACT_ROOT=
EOF
```

- [ ] **Step 5: Commit top-level files**

```bash
cd /tmp/dsm-migration/DSM
git add CLAUDE.md README.md .gitignore infra/
git commit -m "chore: add top-level CLAUDE.md, README, .gitignore, infra/ placeholder"
```

Expected: commit succeeds showing the 5 new files.

- [ ] **Step 6: Verify final structure**

```bash
cd /tmp/dsm-migration/DSM
find . -maxdepth 2 -not -path './.git/*' | sort
```

Expected: shows `./CLAUDE.md`, `./README.md`, `./.gitignore`, `./infra/`, `./sentinel-soil/`, `./soil-moisture/` at the top level.

---

## Task 8: Create DSM repo on GitHub and push

**Files:** None (GitHub operation)

Before this step: create an empty repo named `dsm` (or `DSM`) under the `IbrahimStudio` org on GitHub. Do **not** initialise it with a README — it must be completely empty.

- [ ] **Step 1: Create the GitHub repo**

Go to https://github.com/organizations/IbrahimStudio/repositories/new  
Name: `dsm`  
Visibility: Private (or Public — your choice)  
**Do not** tick "Add a README file" or "Add .gitignore"  
Click "Create repository"

- [ ] **Step 2: Add remote and push**

```bash
cd /tmp/dsm-migration/DSM
git remote add origin https://github.com/IbrahimStudio/dsm.git
git push -u origin main
```

Expected: push succeeds, GitHub shows the new repo with both sub-project directories.

- [ ] **Step 3: Verify on GitHub**

Open https://github.com/IbrahimStudio/dsm in a browser.  
Expected: you can see `sentinel-soil/`, `soil-moisture/`, `infra/`, `CLAUDE.md`, `README.md` in the file tree.

---

## Task 9: Set up transition dual-push remotes

**Files:** None (git config only, applied to local working copies)

You will continue doing daily work from the existing local paths (`/home/ibrahim/Projects/sentinel-soil` and `/home/ibrahim/Projects/soil-moisture`). To use the monorepo as primary, clone it as the new working copy. The dual-push setup adds the old remotes to the DSM clone so you can sync back to the old repos during the transition.

- [ ] **Step 1: Clone DSM as your new working copy**

```bash
cd /home/ibrahim/Projects
git clone https://github.com/IbrahimStudio/dsm.git DSM
cd DSM
ls
```

Expected: `sentinel-soil/`, `soil-moisture/`, `infra/`, `CLAUDE.md`, `README.md`.

- [ ] **Step 2: Add old repos as secondary remotes**

```bash
cd /home/ibrahim/Projects/DSM
git remote add origin-sentinel https://github.com/IbrahimStudio/sentinel-soil.git
git remote add origin-moisture https://github.com/IbrahimStudio/soil-moisture.git
git remote -v
```

Expected: three remotes listed — `origin` (DSM), `origin-sentinel`, `origin-moisture`.

- [ ] **Step 3: Test subtree push to sentinel-soil**

```bash
cd /home/ibrahim/Projects/DSM
git subtree push --prefix=sentinel-soil origin-sentinel main
```

Expected: pushes the `sentinel-soil/` subtree to the old repo. This may take a minute.

**If rejected with "non-fast-forward"** (likely after a filter-repo rewrite, since commit SHAs diverge): use the force-push fallback instead:

```bash
cd /home/ibrahim/Projects/DSM
git subtree split --prefix=sentinel-soil -b _sentinel-split
git push origin-sentinel _sentinel-split:main --force
git branch -D _sentinel-split
```

- [ ] **Step 4: Test subtree push to soil-moisture**

```bash
cd /home/ibrahim/Projects/DSM
git subtree push --prefix=soil-moisture origin-moisture main
```

Expected: pushes the `soil-moisture/` subtree to the old repo.

**If rejected with "non-fast-forward"**: force-push fallback:

```bash
cd /home/ibrahim/Projects/DSM
git subtree split --prefix=soil-moisture -b _moisture-split
git push origin-moisture _moisture-split:main --force
git branch -D _moisture-split
```

- [ ] **Step 5: Commit the archive checklist to DSM**

Add a tracking note so the transition doesn't quietly linger:

```bash
cat >> /home/ibrahim/Projects/DSM/infra/.env.example << 'EOF'

# === TRANSITION NOTE ===
# Old repos are still alive as secondary remotes. Archive them when:
#   - No active CI/CD pipelines point at IbrahimStudio/sentinel-soil or IbrahimStudio/soil-moisture
#   - Docker images are built from DSM or a registry, not the old repos
#   - Both repos are archived on GitHub (Settings -> Archive repository)
#   - origin-sentinel and origin-moisture remotes are removed from local DSM clone
EOF
git -C /home/ibrahim/Projects/DSM add infra/.env.example
git -C /home/ibrahim/Projects/DSM commit -m "chore: add transition archive checklist to .env.example"
git -C /home/ibrahim/Projects/DSM push origin main
```

---

## Task 10: Smoke-test verification

**Files:** None (read-only checks)

- [ ] **Step 1: Verify sentinel-soil history depth**

```bash
cd /home/ibrahim/Projects/DSM
git log --oneline sentinel-soil/ | wc -l
git log --oneline /home/ibrahim/Projects/sentinel-soil | wc -l
```

Expected: both print the same number (DSM preserved all commits).

- [ ] **Step 2: Verify soil-moisture history depth**

```bash
cd /home/ibrahim/Projects/DSM
git log --oneline soil-moisture/ | wc -l
git log --oneline /home/ibrahim/Projects/soil-moisture | wc -l
```

Expected: both print the same number.

- [ ] **Step 3: Smoke-test sentinel-soil Python environment**

```bash
cd /home/ibrahim/Projects/DSM/sentinel-soil/SCALAWAY_JOB
uv run python -c "import sh_statistics; print('ok')"
```

Expected: prints `ok`. If the `.venv` doesn't exist yet, run `uv sync` first.

- [ ] **Step 4: Smoke-test soil-moisture Python environment**

```bash
cd /home/ibrahim/Projects/DSM/soil-moisture
uv run python -c "import moisture_zoning; print('ok')"
```

Expected: prints `ok`. Run `uv sync` first if needed.

- [ ] **Step 5: Final commit — copy updated plan into DSM**

The plan and spec docs currently live in the old `sentinel-soil` repo. Copy them into the DSM monorepo:

```bash
cp -r /home/ibrahim/Projects/sentinel-soil/docs \
      /home/ibrahim/Projects/DSM/sentinel-soil/docs
cd /home/ibrahim/Projects/DSM
git add sentinel-soil/docs/
git commit -m "docs: copy superpowers specs and plans into DSM monorepo"
git push origin main
```

Expected: commit succeeds. The specs and plans are now versioned inside DSM.

---

## Archive checklist (post-transition, no deadline)

Run these when you're ready to retire the old repos:

```bash
# Remove secondary remotes from DSM clone
cd /home/ibrahim/Projects/DSM
git remote remove origin-sentinel
git remote remove origin-moisture

# Archive on GitHub (do this via the web UI):
# https://github.com/IbrahimStudio/sentinel-soil → Settings → Archive
# https://github.com/IbrahimStudio/soil-moisture → Settings → Archive
```
