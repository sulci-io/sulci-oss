# Changelog

All notable changes to Sulci are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.1] — 2025-03-01

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

## [0.1.0] — 2025-02-15

### Added
- Initial release
- Basic semantic cache engine with ChromaDB backend
- MiniLM embeddings
- `cached_call()` wrapper
