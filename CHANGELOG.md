# Changelog

All notable changes to Sulci are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.3] — 2026-03-09

### Added
-- No change .... Bump version in pyproject.toml  →  version = "0.1.3"

## [0.1.2] — 2026-03-09

### Added
-- just a chore to get the workflow captured as below
--  # 1. Bump version in pyproject.toml  →  version = "0.1.2"
-- # 2. Update CHANGELOG.mdpypi-
-- # 3. Commit, tag, push
-- git add pyproject.toml CHANGELOG.md
-- git commit -m "chore: bump to v0.1.2"
-- git tag v0.1.2
-- git push origin main --tags   # --tags pushes tag + triggers auto-publish

## [0.1.1] — 2026-03-09

### Added
- Full library structure: `sulci/`, `backends/`, `embeddings/`
- Six vector backends: ChromaDB, Qdrant, FAISS, Redis, SQLite, Milvus
- Two embedding providers: MiniLM/MPNet/BGE (local), OpenAI API
- `Cache.cached_call()` — drop-in LLM wrapper
- `Cache.get()` / `set()` — manual cache control
- `Cache.stats()` — hit rate, cost savings tracking
- TTL-based cache expiry
- Per-user personalized caching via `user_id`
- GitHub Actions: auto-publish on tag, test matrix (Python 3.9–3.12, 3 OS)
- pytest suite: 20 core tests + backend contract tests
- Examples: `basic_usage.py`, `anthropic_example.py`

### Fixed
- `pyproject.toml` build backend changed from `setuptools.backends.legacy`
  to correct `setuptools.build_meta`
- Removed mandatory `numpy>=1.24` core dependency (now optional per backend)

---

## [0.1.0] — 2026-03-09

### Added
- Initial release
- Basic semantic cache engine with ChromaDB backend
- MiniLM embeddings
- `cached_call()` wrapper
