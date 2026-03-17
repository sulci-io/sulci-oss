# Changelog

All notable changes to Sulci are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.2.4] - 2026-03-16

- Release v0.2.4 — Developer Edition baseline — pre-enterprise transition

## [0.2.3] - 2026-03-16

- Release v0.2.3 — correct test counts, updated docs

## [0.2.2] - 2026-03-15

- Packaging fix: re-publish of 0.2.1 (PyPI file conflict resolution)

## [0.2.1] - 2026-03-11

- Context-aware benchmark suite: --context flag,
- 25 session pools, brute-force cosine scan.
- Results: +20.8pp resolution accuracy.

## [0.2.0] — 2026-03-10

### Added

- **Context-aware caching** for multi-turn LLM conversations
- `sulci/context.py` — new module with `ContextWindow` and `SessionStore`
  - `ContextWindow`: sliding window of turns per session with exponential
    decay blending (`lookup_vec = α·query + (1-α)·Σwᵢ·historyᵢ`)
  - `SessionStore`: concurrent session manager with TTL-based eviction
- `Cache` gains four new init parameters:
  - `context_window` — turns to remember per session (0 = stateless, default)
  - `query_weight` — current query weight vs blended history (default: 0.70)
  - `context_decay` — exponential decay per turn (default: 0.50)
  - `session_ttl` — idle session eviction in seconds (default: 3600)
- `cached_call()`, `get()`, `set()` now accept `session_id` parameter
- All results include `context_depth` field (0 = no context used)
- New context management methods: `get_context()`, `clear_context()`,
  `context_summary()`
- `sulci/__init__.py` now exports `ContextWindow` and `SessionStore`
- `examples/context_aware.py` — 4-demo walkthrough, no API key required
- `tests/test_context.py` — 27 tests covering ContextWindow, SessionStore,
  and Cache integration
- Updated `anthropic_example.py` with `session_id` and `Chat` wrapper

### Fixed

- `tests/test_core.py` — all `cache.get()` call sites updated to unpack
  3-tuple `(response, sim, context_depth)` instead of 2-tuple
- CI workflow updated to also run `test_context.py`

### Changed

- Version bumped to `0.2.0`
- `README.md` updated with context-awareness section and full API reference

### Backward compatibility

- `context_window=0` (default) is identical to v0.1.x behaviour
- No breaking changes — existing code requires zero modifications

---

## [0.1.1] — 2026-03-07

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

## [0.1.0] — 2026-03-07

### Added

- Initial release
- Initial release — 6 backends, MiniLM, TTL, personalization, stats.
