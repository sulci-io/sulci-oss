# Changelog

All notable changes to Sulci are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- **`make smoke-fast` and `make checkin-fast` Makefile targets** — opt-in
  CPU-forced smoke runs for macOS Apple Silicon (closes sulci-oss #48).
  Sets `SENTENCE_TRANSFORMERS_DEVICE=cpu` + `CUDA_VISIBLE_DEVICES=""` +
  `PYTORCH_ENABLE_MPS_FALLBACK=1` + `TOKENIZERS_PARALLELISM=false` +
  `SULCI_SMOKE_FAST=1` before invoking each smoke script. Reduces total
  smoke wall time on M2/M3 laptops from ~8 min to ~30 s by skipping the
  MPS warm-up pass on each of the four smoke processes. Default
  `make smoke` and `make checkin` are unchanged — CI continues to
  exercise the accelerator path on macos-latest matrix rows. Full
  rationale + alternatives-considered documented in
  [`docs/architecture/adrs/0002-smoke-fast-cpu-mode.md`](docs/architecture/adrs/0002-smoke-fast-cpu-mode.md).
  `make checkin` success message now points macOS contributors at
  `make checkin-fast` explicitly.

### Fixed

- **`examples/langchain_example.py` and `examples/llamaindex_example.py`
  now fail fast with a useful message when an API key is rejected** —
  extends the pattern shipped in v0.5.4 (`anthropic_example.py` +
  `async_example.py`, sulci-oss #20) to the two multi-provider integration
  examples. Both now catch `AuthenticationError` (matched by class name to
  avoid hard-importing `openai` / `anthropic` — they're optional deps),
  print a one-line `"<Provider> rejected the API key (HTTP 401). Verify
  your key at <URL>. Falling back to mock LLM for the rest of this demo."`
  message on first rejection, flip an internal `_key_state["rejected"]`
  sentinel, and route all subsequent LLM calls in the run to the deterministic
  mock. Same sentinel + fallback shape as v0.5.4. The rest of the demo
  still runs so the reader gets to see semantic hits, context blending, and
  the summary stats even with a stale key. Previously a stale or wrong key
  surfaced as a raw `HTTPStatusError` traceback mid-output.

  In the LangChain example this involved refactoring the mock LLM out of
  the `if _llm is None:` branch to top-level so the rejected-key path can
  reach it too. In the LlamaIndex example, the mock class had to move to
  top level and become abstract-method-complete (all `complete` / `chat` /
  `stream_*` / `a*` variants) so `SulciCacheLLM(llm=_MockLLM(), ...)` can
  wrap it as a genuine drop-in replacement for the real provider LLM
  mid-run.

### Docs

- **`sulci.connect(prompt=False)` — explain the sustained default** rather
  than promise the v0.6.0 flip that v0.5.3's CHANGELOG note pre-committed
  to. Updates the docstring in `sulci/__init__.py::connect()` and adds a
  clarifying note in the `[0.5.3]` CHANGELOG section. The full OSS-Connect
  chain (SDK device-code client D12 in v0.5.3, gateway endpoints D4/D4.5/D5
  in sulci-platform PR #51 shipped 2026-05-04, dashboard `/oss-connect`
  page D7 in sulci-platform PR #67 shipped 2026-05-05, cutover 2026-05-08)
  has been live end-to-end for two months. The default nonetheless stays
  `prompt=False`, and this change makes that decision permanent and
  auditable rather than a stale promise. Rationale for keeping opt-in as
  the default:
    1. **Non-interactive default is the safe default** for a library called
       from LangChain / LlamaIndex agents, FastAPI request handlers,
       LangGraph nodes, and CI runners — none of which have a tty or a
       browser. `prompt=True` at import time would block those callers on a
       15-minute device-code timeout with no visible cause.
    2. **v0.7.0 shipped `Cache()` auto-connect** — passing an `api_key=` to
       the `Cache` constructor now attaches telemetry automatically, which
       is the ergonomic "make it easy" path users actually reach for. That
       covers the goal the pre-committed flip was aimed at without
       introducing a blocking browser prompt as an import-time side effect.
    3. **Explicit `prompt=True` remains a first-class supported call** —
       nothing changed there; users on interactive machines can still opt
       in per call with one keyword argument.

  No API or behavior change from v0.7.4; just documentation truth-in-labeling.

---

## [0.7.4] — 2026-07-06 — SULCI_GATEWAY on cloud backend + CHANGELOG housekeeping

### Fixed

- **`SULCI_GATEWAY` env var now redirects cache traffic too** (closes the
  v0.5.5 `#TBD-2` follow-up). Previously the env var only redirected
  telemetry POSTs; `Cache(backend="sulci")` cache reads and writes
  continued hitting `https://api.sulci.io` regardless of `SULCI_GATEWAY`,
  creating a split-brain state in staging environments (telemetry to
  staging URL, cache traffic to prod). `sulci/backends/cloud.py::__init__`
  now honors the env var as the second rung in a 3-tier precedence chain:
    1. explicit `gateway_url=` kwarg (VPC customers, unchanged)
    2. `SULCI_GATEWAY` env var (new)
    3. `https://api.sulci.io` default

  Env var is read at instance construction time, not import time, so
  `os.environ["SULCI_GATEWAY"] = "..."; Cache(...)` works without needing
  to set the var before `import sulci`. Backward-compatible: explicit
  `gateway_url=` still wins, so anyone currently passing the kwarg is
  unaffected. Six new tests in `tests/test_cloud_backend.py::TestConstruction`
  lock in the precedence chain, including a regression guard against a
  future refactor that would promote the read to module level.
  (sulci-oss #106)

### Docs

- Restored missing `## [0.6.5]` and `## [0.5.2]` version headers to
  CHANGELOG.md (sulci-oss #105). Both versions shipped and their bodies
  were documented, but the `## [X.Y.Z]` headings had been dropped,
  leaving content under wrong parent versions and breaking GitHub
  anchor links `#065` / `#052`. Zero content changes — only headings
  added, plus one missing `---` separator above 0.6.5.

---

## [0.7.3] — 2026-06-06 — Release integrity: PyPI Trusted Publishing (OIDC)

No library changes — `sulci==0.7.3` is 0.7.2 plus the version bump. This
release exists to live-verify the new credential-free publishing path.

### Changed

- **`publish.yml` publishes via PyPI Trusted Publishing (OIDC).** The
  publish job now declares `environment: pypi` + `permissions: id-token:
  write` and uploads with `pypa/gh-action-pypi-publish@release/v1` — no
  password input, no `twine`. The long-lived `PYPI_TOKEN` was deleted from
  repo secrets **and revoked on PyPI** (the `sulci-github-actions`
  all-projects token). Releases can now only originate from this repo's
  `publish.yml` on a `v*` tag. PEP 740 attestations are generated
  automatically — **0.7.3 is the first attested release** (verified on the
  PyPI file listing and the repo Attestations panel). (PR #100; live
  verification: Publish to PyPI run #48 on tag `v0.7.3`.)

### Added

- **`.gitleaks.toml`** — allowlists the deliberately fake `sk-sulci-*`
  test-fixture credentials so `gitleaks detect --log-opts="--all"` is a
  clean, usable gate. Full-history scan 2026-06-05: 148 commits, 7 hits,
  all fixtures, zero real secrets.
- **`docs/OSS_BOUNDARY_POLICY.md`** — the per-PR checklist for what may
  never land in sulci-oss (fleet-derived tuning, billing/quota/plan logic,
  platform internals) plus the release-integrity runbook this release
  executed.

---

## [0.7.2] — 2026-06-05 — Per-entry hit attribution, agent benchmark mode, agent examples

> Note: no 0.7.1 was published to PyPI — the fixes below that reference
> 0.7.1-era behavior shipped here.

### Added

- **Per-entry hit attribution (`v0.7.2`)** — three additive pieces enabling
  downstream consumers (sulci-platform's Top Queries pipeline) to count how
  many times each cached entry is actually SERVED, aligning the dashboard's
  "Hits" column with the hit-rate stat computed from the same events:
  - `QdrantBackend.store()` writes a `created` timestamp in the payload, so
    aggregators can report an honest "last seen" for entries never served.
  - `QdrantBackend.search_match()` — like `search()`, plus the matched
    entry's STORED query text as a third element. `search()` keeps its
    2-tuple contract by delegating; the `Backend` protocol and the other
    backends are untouched (feature-detected via `hasattr`).
  - `CacheEvent.matched_query_hash` — populated on hits when the backend
    exposes `search_match()` (additive-with-default per ADR 0005, same
    pattern as `plan` in v0.5.6). Privacy: a hash, never text, and
    deliberately NOT on the sink allowlist — `TelemetrySink` and
    `RedisStreamSink` scrub it, so it is consumable only by in-process
    sinks injected by the caller.
  - `sulci.sinks.query_hash()` — the hash scheme (sha256, first 32 hex
    chars), exported as a documented cross-repo contract and pinned with a
    literal-value test on both sides.
  - New test module `tests/test_hit_attribution.py` (12 tests against
    embedded Qdrant).

- **`benchmark/run.py --agent` — measured agent-workload deduplication.** New
  benchmark mode simulates 50 sessions × 200 LLM-call dispatches drawn from a
  realistic workload distribution (45% structural, 35% semi-structural, 20%
  novel) and reports per-session hit rate, cold→warm progression, per-category
  breakdown, and the headline `200 dispatches → X misses` framing that maps
  directly to homepage agent positioning.
  - New CLI flags: `--agent`, `--agent-sessions N` (default 50),
    `--agent-dispatches N` (default 200), `--agent-threshold F` (default 0.85).
  - Synthetic mode (TF-IDF, default) is regression-gated via the new
    `benchmark/baseline.json` `agent_workload` block — locked to 8 metrics
    including aggregate hit rate, cold/warm session rates, per-category
    breakdowns, and p50/p95 misses-per-session.
  - Real-MiniLM mode (`--agent --use-sulci`) produces the conservative
    headline number cited externally (~60-75% aggregate hit rate vs ~95%
    TF-IDF upper bound).
  - Real-Anthropic mode (`--agent --use-sulci --use-claude`) adds measured
    LLM latency + dollar cost saved — the launch-post anchor.
  - New Makefile target: `make benchmark-agent` runs the agent benchmark +
    verifies against baseline. Daily `make checkin` is unchanged — the
    verifier graceful-skips when `agent_summary.json` is absent.
  - Outputs `agent_summary.json` + `agent_per_session.csv` to
    `benchmark/results/`.

### Fixed

- **`Cache.stats()['saved_cost']` now populates for users of the raw
  `get()`/`set()` API (closes #88).** Previously `saved_cost` only
  incremented inside `cached_call()`, which left LangChain integrations —
  the most common high-traffic path, via `set_llm_cache(SulciCache(...))`
  — reporting $0 saved indefinitely. `Cache.__init__` now accepts
  `cost_per_call: float = 0.005`; `get()` contributes that amount to
  `saved_cost` on every hit. `cached_call()` continues to support a
  per-call override (signature changed: default now `None` instead of
  `0.005`, meaning "use the Cache's setting"); explicit overrides apply
  as a delta to preserve identical numeric output for the existing
  `cached_call(cost_per_call=X)` pattern.

  Behavior change worth noting: pre-v0.7.1, `Cache().get()` alone left
  `saved_cost` at $0; now it adds $0.005 per hit (using the default).
  Opt out with `Cache(cost_per_call=0)` to preserve old semantics.
  All other paths produce identical numeric output to v0.7.0.

- **No more `FutureWarning` from sentence-transformers v3+ (closes #89).**
  `sulci/embeddings/minilm.py` now uses `get_embedding_dimension()` when
  available, falling back to the deprecated `get_sentence_embedding_dimension()`
  on sentence-transformers v2.x. The warning was harmless but appeared on
  every smoke run since the v3 upgrade earlier this year, polluting the
  examples output.

### Added

- **`examples/agent_example_crewai.py`** — CrewAI + Sulci agent demo with
  multi-shot warm-up loop. Builds a 2-agent Crew (researcher + writer) with
  a sequential task chain, runs it 3 times against the same topic, and
  prints per-run + aggregate cache stats showing the same cold → warm → hot
  pattern as the LangGraph example. Real Anthropic via the CrewAI `LLM`
  wrapper when `ANTHROPIC_API_KEY` is set; deterministic `BaseLLM` mock
  when not.
  - Demonstrates the BaseLLM-subclass integration pattern (in contrast to
    LangGraph's `set_llm_cache()` global-cache pattern). The same recipe
    works for any non-LangChain agent framework: subclass the framework's
    LLM base, intercept `call()`, route through `cache.get()`/`cache.set()`.
  - Suppresses CrewAI's first-run telemetry opt-in dialog and OTEL upload
    by setting `CREWAI_TRACING_ENABLED=false`, `CREWAI_TELEMETRY_OPT_OUT=true`,
    `OTEL_SDK_DISABLED=true` at module load. Users who want CrewAI tracing
    can re-enable it in their own code.
  - Registered in `scripts/run_examples.py` (LLM-using examples block)
    so `make checkin` exercises it on every dev run.
- **`examples/agent_example_langgraph.py`** — LangGraph + Sulci agent demo
  with multi-shot warm-up loop. Builds a 2-node ReAct-style graph (planner →
  actor), runs the same research task 3 times, and prints per-run + aggregate
  cache stats. Demonstrates the agent-workload value prop concretely
  (planner/reflector inner-loop dedupe) that GA-era homepage messaging
  references. Real Anthropic via `ANTHROPIC_API_KEY` when set; falls back to
  a deterministic `BaseChatModel` mock so the demo runs in CI / sandboxes
  without a key.
  - Documents the install pattern: `pip install "sulci[sqlite,langchain]" langgraph langchain-anthropic`.
  - Uses `set_llm_cache(SulciCache(context_window=4, threshold=0.85, ...))`
    so every LLM call in the LangGraph automatically routes through Sulci —
    no LangGraph-specific code needed.
  - Subclasses `SulciCache` as `CountingSulciCache` to expose per-run
    hit/miss counters for live demo visibility. Production code should use
    `SulciCache` directly and read aggregate numbers from `cache.stats()`.
  - Filename namespaces the framework explicitly (`agent_example_langgraph.py`)
    so future `agent_example_crewai.py` and `agent_example_autogen.py` siblings
    can land without disturbing existing references.

---

## [0.7.0] — 2026-05-26 — Cache() auto-connects telemetry when an api_key is resolvable

> Minor release closing a long-running footgun across **all tiers**
> (OSS-Connect, Pro, Business): pre-0.7.0, constructing
> `Cache(backend="sulci", api_key=...)` registered the key for cache backend
> auth but did **not** enable the telemetry flush path. The four
> telemetry-backed dashboard panels — TrendChart, AuditEventsTable,
> DeploymentsTable, the Active-SDKs counter — stayed empty until the user
> separately called `sulci.connect()`. The aggregate stat cards (Cost Saved,
> Hit Rate, Requests, Quota) populated normally via the cache event pipeline,
> producing a half-broken dashboard that looked like a product bug.
>
> The footgun was identical for paid managed-tier customers: a $29/mo Pro
> tenant following the documented `Cache(backend="sulci", api_key=...)`
> quickstart would land on `ProOverview` with a blank trend chart and empty
> audit feed immediately after upgrade.
>
> v0.7.0 unifies the opt-in rule per §5.2 trust-boundary spec and
> sulci-oss ADR 0001 (mirror: sulci-platform ADR 0021):
>
>     If api_key is resolvable AND telemetry=True (default) → telemetry flows.
>
> One sentence. Three personas covered (paid managed, OSS-Connect, pure
> self-hosted). One canonical quickstart shape everywhere.

### Changed — Cache() auto-connects telemetry

- `Cache.__init__` now auto-calls `sulci.connect()` when **all three** hold:
  1. `self._telemetry` is `True` (the constructor default; explicit
     `telemetry=False` opt-out remains and wins),
  2. an api_key is resolvable from any of `{kwarg, SULCI_API_KEY env,
     module-level _api_key set by a prior connect()}`,
  3. `sulci._telemetry_enabled` is still `False` (no prior `connect()` has
     run — the user's earlier explicit `telemetry=False` choice survives).
- Auto-connect runs with `prompt=False` so it never blocks Cache
  construction on a 15-minute device-code timeout. Users who want the
  device-code flow continue to call `sulci.connect(prompt=True)` explicitly
  before constructing a Cache; the "already connected" short-circuit
  preserves that ordering.
- Auto-connect failures **never crash Cache construction**. The cache
  itself stays fully functional; only telemetry stays disabled, and a
  WARNING is logged on the `sulci` logger naming the exception type +
  message and pointing the operator at `sulci.connect()` to retry.

### Why this is not a breaking change

- The semantic contract — *passing an api_key is explicit opt-in to
  telemetry* — was already documented in §5.2 trust-boundary spec via the
  "or set SULCI_API_KEY" clause. v0.7.0 makes the kwarg path behave
  identically to the env-var path that was already an authorized opt-in
  signal.
- Users who want the prior "cache without telemetry" shape have a clean,
  documented opt-out: `Cache(backend="sulci", api_key=..., telemetry=False)`.
  That kwarg has existed since the cloud backend shipped (v0.6.x); v0.7.0
  doesn't change its semantics, only makes it more useful as an explicit
  opt-out anchor.
- Existing `sulci.connect()` callsites are unchanged. The short-circuit
  at `_telemetry_enabled is False` (rung 3 above) means any prior
  `connect()` invocation wins — including `connect(telemetry=False)`.

### Canonical one-line quickstart (new)

The pattern that now works for every tier and every backend:

```python
from sulci import Cache

cache = Cache(
    backend = "sulci",                # or "sqlite"/"chroma"/etc. for OSS-Connect
    api_key = "sk-sulci-...",         # or set SULCI_API_KEY env var
)

cache.get("hello")  # populates the entire dashboard
```

`sulci.connect()` remains as the canonical entry point for two advanced
flows: the OSS-Connect device-code browser onboarding
(`sulci.connect(prompt=True)`), and the "register key without constructing
a Cache yet" boot pattern (`sulci.connect()` at module load, Cache instances
spawn later in worker threads).

### Added — Tests

- `tests/test_connect.py::TestCacheAutoConnect` — 8 new tests covering:
  kwarg trigger, env-var trigger with local backend, no-api-key short-circuit,
  `telemetry=False` opt-out, prior-connect-with-telemetry-False respect,
  prior-connect-no-re-call, connect-failure-does-not-crash-Cache, and the
  default-backend-no-key regression guard.
- Total test count: 488 → 496 passed (8 new), 30 skipped (unchanged).

### Added — Documentation

- `docs/architecture/adrs/0001-cache-auto-connect-telemetry.md` —
  decision capture, mirroring sulci-platform ADR 0021. First ADR in the
  sulci-oss repo; `docs/architecture/adrs/README.md` introduces the registry
  with the same conventions used in sulci-platform (sequential, never
  renumbered, ISO date-stamped going forward).
- `README.md` § "Sulci Cloud — zero infrastructure option" rewritten to
  show the auto-connect quickstart as primary. Advanced `sulci.connect()`
  patterns moved into a clearly-flagged subsection. Key resolution order
  table updated to reflect three equivalent opt-in signals (kwarg, env,
  prior `connect()`).

### Coordinated release

`sulci-cache==0.7.0` ships alongside `sulci-gateway==0.7.0` in the
sulci-platform repo. The gateway version bump is purely operational —
no behavioral change on the gateway side, since the auto-connect logic
lives entirely in the SDK. Bumping both to the same `0.7.0` marker
keeps the canonical version cross-reference clean in docs and support
conversations.



---

## [0.6.5] — 2026-05-13 — Resolution-path logging in `connect()` + `~/.sulci/config` age-gate (90 days)

> Patch release closing the two sulci-oss follow-up issues filed during
> the v0.6.x close-out (#79, #80). Both target the same surface — the
> four-rung api_key resolution chain in `sulci.connect()` — and the same
> failure mode: silent fallback to a stale key with no signal to the
> caller. Together they make "which key is the SDK actually using?"
> answerable without raw SQL access to the gateway, and they prevent
> stale persisted keys from winning the resolution race in the first
> place.

### Added — Resolution-path logging (closes #79, PR #82)

- `sulci.connect()` emits an INFO-level log line on the `sulci` logger
  indicating which resolution rung supplied the key:
sulci.connect: using explicit api_key argument (prefix=sk-sulci-abcd)
sulci.connect: using SULCI_API_KEY env var (prefix=sk-sulci-efgh)
sulci.connect: using persisted ~/.sulci/config (prefix=sk-sulci-ijkl, mtime=2026-05-13T14:23:00+00:00)
sulci.connect: using device-code flow result (prefix=sk-sulci-mnop)

- For the config rung, the file's mtime is included — knowing *when*
  the config was written is the single most useful diagnostic for
  stale-key cases.
- Default Python logging level (WARNING) keeps INFO lines quiet. Opt in
  with `logging.getLogger("sulci").setLevel(logging.INFO)` when
  debugging "wrong key" or "telemetry not arriving" issues.
- Two new internal helpers in `sulci/__init__.py`:
  - `_key_prefix(api_key)` — first 16 chars, enough to identify in
    logs, not enough to use as a credential
  - `_config_file_mtime()` — ISO mtime of `~/.sulci/config`, or `None`
    on any failure mode (file missing, permission, etc.)
- The "no key resolved" case emits a DEBUG line (not INFO) — fine
  behavior when `prompt=False`, not worth INFO-level attention.

### Added — Config staleness guard (closes #80, PR #83)

- `sulci/config.py::update()` now auto-stamps a `written_at` UTC ISO-8601
  timestamp whenever `api_key` is among the fields being written. Other
  field writes (e.g. `machine_id`-only via `get_machine_id()`) do not
  stamp — those are not authentication events.
- `sulci/__init__.py::_read_key_from_config()` now refuses to use
  `~/.sulci/config` entries that are stale. Three reject paths, all
  returning `None` and emitting WARNING with remediation text:
  - **Missing `written_at`** — config predates v0.6.5, age can't be
    verified, treated as stale
  - **Older than 90 days** (`_CONFIG_MAX_AGE_DAYS`) — over threshold
  - **Unparseable `written_at`** — corrupt or future-incompatible
- WARNING example:
sulci.connect: ~/.sulci/config is 127 days old (written
2026-01-06T10:30:00+00:00; threshold 90 days) — treating as stale
and skipping. Re-run with sulci.connect(prompt=True) to refresh,
or pass api_key=... explicitly.

### Why this matters

During the v0.6.x close-out debugging session (2026-05-13), an operator
had three different keys across three locations: an explicit arg in a
test script, `SULCI_API_KEY` env var set to a key from a previous
account, and `~/.sulci/config` with yet another stale key. Any no-args
`sulci.connect()` call silently picked whichever key was first in the
fallback chain. Multiple hours of debugging time were spent diagnosing
this. The v0.6.5 changes make the same diagnostic visible in 30 seconds:

```python
import logging
logging.getLogger("sulci").setLevel(logging.INFO)

import sulci
sulci.connect()
# INFO sulci: using persisted ~/.sulci/config (prefix=sk-sulci-EHPt5T,
#                                              mtime=2026-02-15T14:23:00+00:00)
```

A user seeing a February mtime on a config they thought was current has
their answer in one log line. And under the v0.6.5 rules, a config that
old would have been *skipped* with a WARNING in the first place,
forcing a fresh resolution that picks the actually-current key.

### Backward compatibility

- No API surface changes. Same `sulci.connect(...)` signature, same
  resolution order, same key selection in the fresh-key case.
- **One upgrade-path behavior change**: existing `~/.sulci/config` files
  from v0.6.4 and earlier have no `written_at` field. On the first
  `sulci.connect()` call after upgrading to v0.6.5, those configs are
  treated as stale and skipped, with a WARNING. The user has two paths
  to resolve:
  1. Pass `api_key=` explicitly (immediate bypass)
  2. Re-run with `sulci.connect(prompt=True)` to refresh via device-code

  This forces one fresh re-auth after upgrade, giving us a known-fresh
  key with a known timestamp going forward. The benefit (stale keys
  silently winning resolution races stop happening) substantially
  outweighs the one-time friction.
- The `~/.sulci/config` write format gains a `written_at` field. Reads
  tolerate its absence (treat missing as stale per the rule above).

### Verification

- 6 new tests in `tests/test_connect.py::TestResolutionPathLogging`
  cover each rung's log line, the mtime field, the DEBUG path for
  no-key cases, and the WARNING-default quiet behavior.
- 5 new tests in `tests/test_connect.py::TestConfigAgeOut` cover fresh
  config used (1 day + 90-day boundary), stale config skipped (91 days),
  missing `written_at` treated as stale, and unparseable `written_at`
  treated as stale.
- 4 new tests in `tests/test_config.py::TestWrittenAtStamping` verify
  that `update()` stamps `written_at` only when `api_key` is among the
  fields, and that subsequent api_key writes refresh the timestamp.
- `tests/test_telemetry_lifecycle.py` (v0.6.4 atexit regression suite)
  unchanged — no behavior change to the telemetry pipeline.
- Full `make checkin`: **496 passed, 0 failed, 0 errors, 30 skipped**
  on the v0.6.5 prep branch.
- Benchmark verification: every stateless + context-aware metric
  exactly matches the pre-v0.4.0 baseline (Δ=+0.0000 across the board).

---

## [0.6.4] — 2026-05-12 — Drain telemetry buffer on process exit

> Patch release. Closes the "I ran the snippet, nothing appeared on
> dashboard.sulci.io" failure mode: short-lived processes (CLI commands,
> demo scripts, serverless invocations, test runs) now flush their
> telemetry buffer before exit instead of silently dropping it.

### Changed

- `_start_flush_thread()` now registers an `atexit` hook that drains
  the event buffer on process exit. Previously, the daemon flush
  thread was killed when the main thread exited, losing any events
  buffered since the last 30s tick — which for any process shorter
  than 30 seconds meant *all* events.

### Why this matters

Before v0.6.4: running the /why-connect demo snippet (or any script
shorter than 30 seconds) showed no telemetry on dashboard.sulci.io. The
script exited before the first 30-second flush tick fired, and the
daemon thread died with the buffer intact. Workaround required adding
`time.sleep(35)` to scripts — fine for development, broken for
serverless/CLI use cases where it's not feasible.

After v0.6.4: the same script flushes its buffer at process exit
(typically within a few hundred milliseconds), then exits cleanly.
Telemetry appears on the dashboard within ~1-2 seconds of script
completion.

### Verification

A new behavioral test `tests/test_telemetry_lifecycle.py::
TestAtexitFlush::test_flush_on_exit_drains_buffer` mocks the HTTP
layer, emits an event, runs `atexit._run_exitfuncs()`, and asserts
the POST to `/v1/telemetry` was made. Catches future regressions
where someone removes the atexit hook or changes the flush thread
to non-daemon (which would block process exit instead).

### Backward compatibility

- No API surface changes. Same `sulci.connect(...)` signature.
- Long-running services (web servers, batch jobs, the benchmark
  suite) are unaffected — the existing 30-second flush loop continues
  to run identically. The atexit hook only fires once, at process
  exit, and is a no-op if telemetry is disabled.
- The atexit handler is wrapped in try/except to preserve the
  "telemetry never raises" contract. A stalled gateway at exit will
  delay process termination by httpx's default timeout but won't
  crash the process.

---

## [0.6.3] — 2026-05-12 — Promote `httpx` to mandatory dependency (closes B1)

> Patch release. Closes a packaging gap that was open since the first
> release: `pip install sulci` alone now ships what it needs to call
> `sulci.connect()` and use `Cache(backend="sulci")`. Previously `httpx`
> lived only in the `[cloud]` extra, so bare-install users hit silent
> telemetry failures and loud `ModuleNotFoundError` on the cloud backend.

### Changed

- `httpx>=0.27.0` is now a mandatory dependency (was `[cloud]`-extra-only).
- The `[cloud]` extra is preserved as a back-compat no-op so existing
  `pip install "sulci[cloud]"` install commands keep working without
  warnings.

### Why

`pip install sulci` on a fresh environment historically produced three
asymmetric outcomes depending on which code path the user hit first:

| User action                                | Pre-fix behavior |
|---|---|
| `Cache(backend="sqlite")`                  | Loud `ImportError` with helpful "install `sulci[sqlite]`" message (via the existing `sulci/embeddings/minilm.py` try/except wrapping). |
| `Cache(backend="sulci", api_key=…)`        | Loud but unhelpful `ModuleNotFoundError: No module named 'httpx'` from the top-of-module import in `sulci/backends/cloud.py`. |
| `sulci.connect(api_key=…)`                 | **Silent** failure: the telemetry path is contractually "never raise"; the missing-httpx ImportError is swallowed by the flush thread. User sees no error, but their deployment never appears on `dashboard.sulci.io`. |

The third outcome is the worst — it looks like everything works but the
user gets zero signal that anything's wrong. Promoting `httpx` to
mandatory deps eliminates all three failure modes at once.

Cost: ~150KB on disk, no transitive bloat. The heavier embedding stack
(`sentence-transformers` ~500MB with torch) remains in per-backend
extras because it's only needed for local-embedding backends.

### Verification

The B1 closure is verified by `tests/integration/flows/flow_2_e2e.py`
running on `pip install sulci` (no extras) in a clean venv: the
SulciCloudBackend construction succeeds and the wire-shape assertions
hold.

### Backward compatibility

- `pip install "sulci[cloud]"` keeps working (no warning, no change in
  resolved package set — pip dedupes the redundant httpx declaration).
- All other `pip install sulci[…]` extras keep working.
- No SDK-side API changes.

---

## [0.6.2] — 2026-05-11 — GDPR-adjacent fix: `cache.clear()` and `cache.delete_user()` now actually delete (sulci-oss #103 SDK companion)

> Patch release. Closes the SDK half of the cross-repo `cache.clear()` /
> `cache.delete_user()` bug — companion to **sulci-platform #103** which
> shipped the gateway-side DELETE routes the same day. With both halves
> live, customer requests for cache wipe or user-data deletion now
> actually delete data on the server instead of being silently swallowed.

### What was broken

Since v0.3.0 (~14 months) the SDK's `SulciCloudBackend.delete_user(id)` and
`.clear()` sent HTTP DELETEs to two paths that **never existed** on the
gateway:

  - `DELETE /v1/user/{user_id}` — gateway returns 404
  - `DELETE /v1/cache`          — gateway returns 404

Each method wrapped the call in `try: ... except Exception: pass`, so the
404 was silently swallowed and the method returned `None` — looking like
success to the calling app. **The customer-visible effect was a
success-shaped no-op for a GDPR-relevant operation:** a user requesting
deletion of their cached responses got `None` back (which the app could
not distinguish from success), but the data stayed in the gateway's
managed Qdrant.

The bug was masked from discovery for ~14 months by sulci-oss #62 (a
separate payload-contract mismatch that 422'd every cloud cache call
before any of these DELETEs would have mattered). Once #62 closed in
v0.6.0 (2026-05-11) and cloud transport actually started reaching the
gateway, the DELETE silent-404 became load-bearing for customer GDPR
compliance.

### Fixed

- **`sulci/backends/cloud.py:delete_user(user_id)`** now sends
  `DELETE /v1/cache/user/{user_id}` (the gateway's canonical route per
  sulci-platform #103). Path was `/v1/user/{user_id}` pre-fix.

- **`sulci/backends/cloud.py:clear()`** now sends `DELETE /v1/cache/clear`
  (canonical route per sulci-platform #103). Path was `/v1/cache` pre-fix.

- **Failures no longer silent.** Both methods previously wrapped the
  HTTP call in `try: ... except Exception: pass`, swallowing every
  failure mode including the 404. Both now use
  `except Exception as e: log.warning(...)` instead — the contract is
  still that neither method raises (preserves the v0.6.x non-crashing
  guarantee), but customers and operators can now see deletion failures
  in standard application logs. `log` is `logging.getLogger(__name__)`,
  so the warnings inherit whatever logging config the calling app
  already uses.

- **Vendored gateway contract extended.** Two new pydantic models
  added to `tests/test_cloud_backend.py::_GatewayContractModels`:
  `CacheClearResponse` and `CacheDeleteUserResponse`. These pin the
  gateway's DELETE-route response shapes so future drift in either
  the SDK or the gateway fails CI loudly rather than silently
  mis-deserializing in production — the same pattern v0.6.0 introduced
  for the GET/SET request contracts (closes the same class of bug at
  one layer up).

### Tests

- **`tests/test_cloud_backend.py::TestDeleteAndClear`** rewritten (4
  tests → 8 tests). New tests:
  - `test_delete_user_posts_to_canonical_path` and
    `test_clear_posts_to_canonical_path` — assert the corrected URLs
    (`/v1/cache/user/{id}` and `/v1/cache/clear`).
  - `test_delete_user_logs_warning_on_failure` and
    `test_clear_logs_warning_on_failure` — assert `log.warning` is
    called on `httpx` errors, replacing the pre-v0.6.2 silent-swallow
    behavior.
  - `test_delete_user_does_not_raise_on_failure` and
    `test_clear_does_not_raise_on_failure` — preserved as
    contract-pin tests: even with warnings, the methods still never
    crash the customer's app.
  - `test_delete_user_response_round_trips_through_CacheDeleteUserResponse`
    and `test_clear_response_round_trips_through_CacheClearResponse` —
    vendored-contract round-trip checks for the gateway's DELETE
    response shapes.

- **`tests/test_cloud_backend.py::TestCanonicalGatewayPaths::test_no_legacy_paths_in_source`**
  extended: now asserts cloud.py contains zero references to the
  pre-v0.6.2 broken DELETE paths (`"/v1/user/"` and the bare `"/v1/cache"`
  as a complete URL string). This catches future regressions where a
  new method is added but a `/cache/` prefix is forgotten.

### Notes for upgraders

- This is a **patch release** — no API surface changed. The contract for
  `delete_user(user_id) -> None` and `clear() -> None` is preserved.
  Drop-in upgrade from 0.6.1 → 0.6.2 with no code changes needed.
- If you were relying on the pre-fix silent-failure behavior for any
  reason (e.g., calling `delete_user` from a path where you didn't want
  log lines), the new `log.warning` may appear in your logs when the
  gateway is unreachable or returns an error. To suppress, configure
  your logging to silence the `sulci.backends.cloud` logger.
- If you were running pre-v0.6.0 (and therefore pre-v0.6.2 by extension),
  none of this matters — sulci-oss #62 prevented every cloud cache call
  from succeeding anyway. The GDPR risk only materialized for customers
  who upgraded to v0.6.0 + v0.6.1 between 2026-05-11 and now. The window
  is narrow.

---

## [0.6.1] — 2026-05-11 — Fix cloud-only install path (sulci-oss #60)

> Patch release. Closes a v0.6.0 install-path gotcha discovered during the
> v0.6.0 release smoke test: `pip install "sulci[cloud]==0.6.0"` followed by
> `Cache(backend="sulci", api_key=...)` crashed at construction with
> `ImportError: sentence-transformers not found`. The cloud extra correctly
> declares only `httpx>=0.27` (no sentence-transformers, since the cloud
> transport doesn't do local embedding), but `Cache.__init__` eagerly
> loaded a local `MiniLMEmbedder` regardless of which backend was selected.
>
> After this patch, `pip install "sulci[cloud]"` is enough to use
> `Cache(backend="sulci", ...)` end-to-end. Self-hosted backends are
> unaffected — they continue to load their embedder eagerly at construction
> time, identical to v0.6.0 behavior.

### Fixed

- `Cache.__init__` defers the local `Embedder` load when the backend is a
  cloud transport. Construction order is now: (1) load backend, (2) detect
  remote transport via `hasattr(backend, "remote_get")`, (3) skip embedder
  load when the flag is set. `self._embedder` stays `None` on the cloud
  path; every read of it (in `_context_vec`, the self-hosted branches of
  `Cache.get` / `Cache.set`, and the `cached_call` hit-record path) is
  already gated by `self._is_remote_transport` or sits inside the
  self-hosted `else:` branch. Closes sulci-oss
  [#60](https://github.com/sulci-io/sulci-oss/issues/60).

- `cached_call` hit-record session-tracking path now skips when
  `_is_remote_transport` is True. Mirrors the same-pattern guard
  (`raw_vec is not None`) already present in `Cache.set`. Without this
  guard, a cache hit on a session-aware cloud `Cache` (`backend="sulci"` +
  `context_window > 0`) would crash on `None.embed(query)`. Surfaced by
  code-path audit during the #60 fix; no customer report (the combination
  is unusual since cloud-tier session tracking happens on the gateway side).

- Friendlier error when constructing a cloud `Cache` without `httpx`
  installed: `_load_backend("sulci", ...)` now catches the
  `ModuleNotFoundError` from `import httpx` inside
  `sulci/backends/cloud.py` and re-raises with guidance to
  `pip install "sulci[cloud]"`. Pre-v0.6.1 surfaced as a bare
  `ModuleNotFoundError: No module named 'httpx'` — accurate but unhelpful.

### Tests

- New `TestCloudTransportNoLocalEmbedder` class in `tests/test_core.py` —
  4 tests covering: (a) constructing `Cache` with a fake remote-transport
  backend (object with `remote_get` + `remote_set` methods) leaves
  `self._embedder` as `None`; (b) `Cache.get` round-trips through
  `remote_get` without touching the embedder; (c) `Cache.set` round-trips
  through `remote_set` without touching the embedder; (d) `cached_call`
  hit-record session path skips embed call on remote transport. Uses the
  same offline `_FakeBackend` injection pattern as `TestInstanceInjection`
  (added in v0.6.0 PR #64) — no live gateway, no sentence-transformers
  import required.

### Behavior preserved (regression guard)

- Self-hosted `Cache(backend="sqlite", ...)`, `Cache(backend="chroma", ...)`,
  etc. still load `MiniLMEmbedder` eagerly at construction time — no
  behavior change. The pre-existing test suite (469 tests, all green on
  v0.6.0) continues to pass identically — the conditional only fires
  when the backend exposes the `remote_get` / `remote_set` duck-type
  protocol that v0.6.0 introduced.

---

## [0.6.0] — 2026-05-11 — Cloud transport finally works (umbrella #63)

> Brings the cloud backend end-to-end alive for the first time since v0.3.0.
> Three coordinated PRs under sulci-oss umbrella
> [#63](https://github.com/sulci-io/sulci-oss/issues/63):
> [#64](https://github.com/sulci-io/sulci-oss/pull/64) (Embedder + Backend
> instance injection in `Cache.__init__`),
> [#65](https://github.com/sulci-io/sulci-oss/pull/65) (cloud transport
> short-circuits local embedding, sends canonical payload),
> sulci-platform [#106](https://github.com/sulci-io/sulci-platform/pull/106)
> (canonical-architecture-v3 diagram aligned with embedder layer reality).
>
> **Customer-visible:** `Cache(backend="sulci")` now returns real cache hits
> instead of the silent `(None, 0.0)` it had been returning for ~14 months.
> Pre-v0.6.0 the SDK sent a payload the gateway 422-rejected (post-v0.5.7;
> 404'd pre-v0.5.7) and the SDK's outer `except Exception:` swallowed the
> error. v0.6.0 sends the canonical wire payload, the gateway accepts it,
> and the platform-side library does the embedding via `EmbedServiceEmbedder`.
> Self-hosted backends (chroma, qdrant, faiss, redis, sqlite, milvus) are
> completely unaffected.

### Added

- `Cache.__init__` now accepts pre-constructed `Embedder` and `Backend`
  protocol instances on the existing `embedding_model=` and `backend=`
  parameters. Previously these accepted only string identifiers
  (`"minilm"`, `"qdrant"`, etc.); injecting custom or platform-managed
  instances required subclassing `Cache` and overriding
  `_load_embedder` / `_load_backend` — the workaround the platform
  ships today as `LibraryBackedCache(sulci.Cache)`. After this change,
  `Cache(embedding_model=my_embedder, backend=my_backend)` works
  natively, and the subclass workaround can retire. Closes sulci-oss
  #34 sub-issues C1c (Embedder instance injection) and C1d (Backend
  instance injection).

- `SulciCloudBackend` gained `remote_get(query, threshold, ...)` and
  `remote_set(query, response, ...)` methods that send the canonical
  `{query, threshold, user_id, session_id}` and
  `{query, response, user_id, session_id, ttl_seconds}` wire payloads
  matching the gateway's `CacheGetRequest` / `CacheSetRequest` pydantic
  models exactly. `Cache.get` / `Cache.set` detect a cloud transport
  (via `hasattr(backend, "remote_get")` set once at `__init__` as
  `self._is_remote_transport`) and route through these methods directly,
  skipping `self._embedder.embed()` entirely — the gateway-side library
  does the embedding via its injected `EmbedServiceEmbedder`. Closes
  sulci-oss #62 (`SulciCloudBackend` violates ADR 0008).

### Changed

- Type signatures loosened on two `Cache.__init__` parameters:
  `backend: str` → `backend: Union[str, Backend]` and
  `embedding_model: str` → `embedding_model: Union[str, Embedder]`.
  Fully backward-compatible — every existing string-based caller works
  unchanged. The `_load_embedder` / `_load_backend` dispatchers gained
  an `isinstance(x, str)` short-circuit at the top; non-string inputs
  are returned as-is.

- **Behavior change for `Cache(backend="sulci")` users.** Pre-v0.6.0,
  `Cache.get` embedded queries locally and sent the embedding to the
  gateway, which 422-rejected the payload (the SDK silently swallowed
  it and returned `(None, 0.0)` — Issue #62). After v0.6.0, the SDK
  sends the raw query string and the gateway-side library does the
  embedding. **Net effect: cloud-tier customers see actual cache hits
  for the first time since v0.3.0.** Self-hosted backends (chroma,
  qdrant, faiss, redis, sqlite, milvus) are unaffected — they continue
  to receive `embedding` and do local ANN search.

### Removed

- `SulciCloudBackend.search()`, `.store()`, and `.upsert()` methods
  have been removed. These previously implemented the `Backend`
  protocol's vector-search shape, but the cloud backend was never
  actually a backend in that sense — it's a transport for the entire
  `Cache.get/set` call. The methods sent malformed payloads (Issue #62)
  and have never worked end-to-end against the live gateway since
  v0.3.0; any caller using them directly was already getting silent
  failures. Use `remote_get(query, threshold, ...)` and
  `remote_set(query, response, ...)` instead (or call via `Cache.get` /
  `Cache.set`, which now route automatically).

### Tests

- New `TestInstanceInjection` class in `tests/test_core.py` (7 tests)
  covers: instance pass-through identity, mixed string+instance kwargs,
  string-path regression (mocked MiniLM to keep offline-runnable),
  end-to-end `embed()` / `search()` / `store()` round-trips proving
  injected instances are actually used by `Cache.get` / `Cache.set`.

- New `TestRemoteGet` (9 tests) and `TestRemoteSet` (3 tests) in
  `tests/test_cloud_backend.py` replace the pre-v0.6.0 `TestSearch` and
  `TestUpsert` classes — same error-swallow / payload-shape coverage,
  on the new method names + canonical payload contract.

- New `TestCloudTransportShortCircuit` class (8 tests) verifies
  `Cache.get` / `Cache.set` route through `remote_get` / `remote_set`
  when the backend is a cloud transport, never touching the local
  embedder. Uses fake transports + fake embedders so it runs offline.

- `TestCanonicalGatewayPaths` extended (7 tests, up from 4) with
  **payload-contract round-trip assertions** — vendored copies of the
  gateway's `CacheGetRequest` / `CacheSetRequest` pydantic models live
  at `_GatewayContractModels` in the test file, pinned to
  `sulci-platform:shared/models.py` via a sync-comment. Every SDK
  payload is now `model_validate()`-checked against these copies, so
  any future field drift fails CI loudly rather than silently 422'ing
  in production (the failure mode that kept Issue #62 alive since
  v0.3.0). This extends the v0.5.7 URL-pinning lesson — "tests must
  assert what the gateway expects, not what the SDK does" — from URLs
  to JSON payloads.

---

## [0.5.7] — 2026-05-10 — Fix cloud backend URL paths (sulci-oss P0)

Three-string fix to `sulci/backends/cloud.py` aligning the SDK's request
URLs with the gateway's canonical paths. Pre-0.5.7, the cloud backend
POSTed to `/v1/get` and `/v1/set`; the gateway has always exposed
`/v1/cache/get` and `/v1/cache/set`. Every request returned 404, the
SDK's outer `except Exception:` clause swallowed it, and `cache.get()`
returned `(None, 0.0)` — a silent dataplane failure across the entire
managed-cloud tier. Users saw "sulci doesn't seem to be caching anything"
with nothing in their logs to investigate.

### Fixed

- **`SulciCloudBackend.search()`** (`cloud.py:101`) now POSTs to
  `/v1/cache/get` rather than `/v1/get`.

- **`SulciCloudBackend.store()`** (`cloud.py:150`) now POSTs to
  `/v1/cache/set` rather than `/v1/set`.

- **`SulciCloudBackend.upsert()`** (`cloud.py:179`) now POSTs to
  `/v1/cache/set` rather than `/v1/set`. The delete path at `cloud.py:201`
  already used `/v1/cache` and was unaffected.

### Added

- **`TestCanonicalGatewayPaths`** in `tests/test_cloud_backend.py`. Four
  assertions pinning each URL-bearing method (`search`, `store`, `upsert`)
  to the gateway's canonical path, plus a static-source check that catches
  the regression case where a new method is added but the URL prefix is
  forgotten. The class header documents the gateway-side source of truth
  (`gateway/app/main.py` for the prefix, `gateway/app/routes/cache.py` for
  the route decorators) so the contract is auditable from the test file.

### Changed

- **`TestSearch.test_sends_correct_payload`** previously asserted
  `call_args[0][0] == "/v1/get"`, tautologically locking in the bug. Now
  asserts `"/v1/cache/get"`. The stale docstring on
  `TestUpsert.test_sends_correct_payload` was also corrected.

### Why this slipped past CI

The pre-0.5.7 unit test asserted the SDK was POSTing to `/v1/get` and was
passing because the SDK was, in fact, POSTing to `/v1/get`. The test
verified the wrong contract — what the SDK *did*, rather than what the
gateway *expected*. The new `TestCanonicalGatewayPaths` class encodes the
gateway-side contract as a comment block and asserts against it directly,
so a future drift on either side is caught at the test boundary rather
than via production telemetry.

### Compatibility

Strictly a bugfix. No public API changes, no payload shape changes, no
new dependencies. Any caller that was getting silent `(None, 0.0)` cache
misses against the live gateway will now actually hit the cache.

---

## [0.5.6] — 2026-05-08 — `plan` field on `CacheEvent` (sulci-oss #36)

Additive field on the v0.5.0 `CacheEvent` dataclass plus a matching
keyword argument on `Cache.get` / `Cache.set` / `Cache.cached_call`,
so callers who know a tenant's plan tier at emit time can attribute
it onto the event without monkey-patching the dataclass or doing a
join at consume time. Backward-compatible per ADR 0005's
"additive kwarg with default" rule — pre-0.5.6 callers see no
behavior change; emitted events default to `plan=None`.

### Added

- **`CacheEvent.plan: Optional[str] = None`** (#36). New field on the
  privacy-firewalled event surface, sitting alongside `tenant_id`.
  Carries the customer plan tier (`'free' | 'pro' | 'business' |
  'enterprise' | 'oss_connect'`) when the caller knows it. Defaults
  to `None` so users of the OSS library who don't have plan context
  don't have to thread anything through.

- **`plan: Optional[str] = None`** added as a keyword-only argument
  to `Cache.get`, `Cache.set`, and `Cache.cached_call`. When supplied,
  it is forwarded onto the emitted `CacheEvent.plan`. `cached_call`
  threads it through both its internal `.get()` and `.set()` calls so
  the miss-then-set path emits two events that both carry plan.

- **`"plan"` added to `_ALLOWED_FIELDS`** in `sulci/sinks/telemetry.py`
  so it survives the privacy firewall and reaches `TelemetrySink` /
  `RedisStreamSink` consumers. The allowlist's docstring now
  articulates the three-criteria rule for future additions: a candidate
  field must be (a) low-cardinality, (b) already known to the
  recipient via auth context, and (c) explicitly billing- or
  routing-relevant. `plan` satisfies all three.

### Why

The sulci-platform billing pipeline reads cache events from a Redis
stream and routes them by tenant + plan. Pre-0.5.6, `CacheEvent` had
no plan field, so the gateway emitted events with `plan` recoverable
only by joining each event back to Postgres at consume time. That
join was painful enough that two real-world E2E tests in the platform
(`test_09_billing_events_have_correct_tenant_and_plan` and
`test_j09_billing_events_carry_pro_plan`) had been failing for weeks
with `[None, None, None, None, None]`, eating a per-PR bypass-note
tax on every backend-touching change. Carrying plan on the event
closes that gap and lets the gate run clean.

### Tests

- `tests/test_core.py::TestCacheEventPlan` (6 tests). Recording-sink
  fixture verifies `plan` flows from `Cache.get` / `.set` / `.cached_call`
  onto the emitted `CacheEvent`, that the default-`None` path is
  unchanged for pre-0.5.6 callers, and that `plan` is keyword-only
  with default `None` on all three methods (pinning the API shape
  the same way `tenant_id` / `user_id` / `session_id` are pinned).

- `tests/test_sinks.py` — `TestAllowlist::test_allowlist_contents_are_stable`
  extended to include `"plan"`. Two new scrubbing tests verify
  `plan="pro"` and `plan=None` both round-trip through `_scrub`. The
  canonical `sample_event` fixture now sets `plan="pro"` so all
  existing scrub-loop tests cover the new field implicitly.

### Privacy review note

Adding any field to `_ALLOWED_FIELDS` is a privacy-relevant change.
`plan` was reviewed against the rule the docstring now documents:

| Criterion                                     | `plan` satisfies? |
| --------------------------------------------- | ----------------- |
| Low-cardinality (closed enum, ~5 values)      | yes               |
| Already known to recipient via auth context   | yes               |
| Explicitly billing- or routing-relevant       | yes               |

Adding `plan` doesn't expose anything the receiving service didn't
already know; it removes a join. The cardinality is bounded; there
is no PII or free-form content carried.

### Compatibility

- Existing callers (no `plan` kwarg): emit `plan=None`, identical to
  pre-0.5.6 behavior on the wire.
- Older sinks that don't know about the new field: `_scrub` is built
  on `dataclasses.asdict`, so missing the field on an old struct is
  impossible — the field exists on every `CacheEvent` instance from
  this version forward.
- Custom `EventSink` implementations: receive `event.plan` like any
  other field; no breaking change to the sink API.

---

## [0.5.5] — 2026-05-07 — telemetry honors `SULCI_GATEWAY` (PR-D)

One-line behavior fix that unblocks staging-gateway smoke tests for the
sulci-platform Connected-OSS dashboard tier (LAUNCH-PLAN row C2e). No
new public API surface; no changes for users running against the
default `https://api.sulci.io` gateway.

### Fixed

- **`SULCI_GATEWAY` now actually redirects telemetry POSTs** (#51).
  `_TELEMETRY_URL` is now derived from `_GATEWAY_BASE` instead of being
  a separate hardcoded literal. Prior to v0.5.5, setting
  `SULCI_GATEWAY=https://staging.example.com` redirected the v0.6.0
  device-code flow but silently did NOT redirect the v0.5.x telemetry
  pipeline — the `_post()` helper still went to `api.sulci.io`. The
  module comment claimed staging override was supported; the code
  contradicted it. Now they agree:

  ```python
  # before (v0.5.4)
  _TELEMETRY_URL = "https://api.sulci.io/v1/telemetry"   # hardcoded
  _GATEWAY_BASE  = os.environ.get("SULCI_GATEWAY", "https://api.sulci.io").rstrip("/")

  # after (v0.5.5)
  _GATEWAY_BASE  = os.environ.get("SULCI_GATEWAY", "https://api.sulci.io").rstrip("/")
  _TELEMETRY_URL = f"{_GATEWAY_BASE}/v1/telemetry"
  ```

  Backward-compatible: callers who don't set `SULCI_GATEWAY` see no
  change (still resolves to `https://api.sulci.io/v1/telemetry`).

### Tests

- New `tests/test_telemetry_gateway_override.py` (6 tests) covering
  default URL, env override, trailing-slash normalization, localhost
  for local-dev, and end-to-end verification that `_post()` actually
  POSTs to the resolved URL — closing the gap that let v0.5.4 ship
  with a comment that disagreed with the code.

### Out of scope (filed as follow-up)

- `sulci/backends/cloud.py` (the `Cache(backend="sulci")` HTTP backend)
  still hardcodes `CLOUD_URL = "https://api.sulci.io"` and only honors
  a programmatic `gateway_url=` kwarg, not `SULCI_GATEWAY`. This is a
  separate issue and a separate ergonomic gap; tracked as #TBD-2 for a
  future minor.

---

## [0.5.4] — 2026-05-04 — D7 enabler bundle (PR-C)

Five paper-cut fixes that land alongside the platform's D7 dashboard
`/oss-connect` page work. Each one removes friction a freshly OSS-Connect-
funneled user would otherwise hit in the first five minutes — startup
visibility, raw-API stats, examples idempotency, key-rejection clarity,
and PyPI-page metadata. No new public API surface; one observable
behavior change called out under **Changed**.

### Added

- **POST `event='startup'` from `_flush()`** (#41). When `sulci.connect()`
  is called the resulting startup event now reaches `/v1/telemetry`
  instead of being drained on the floor. One POST per flush cycle that
  contains any startup event (multiple buffered startups collapse to a
  single dashboard row — startup is a state, not a counter). Backend
  is sniffed from any non-startup event in the same batch; if the
  startup ships before any cache traffic it goes out with `backend=""`,
  which the gateway accepts and the fingerprint dedupes against later
  rows. Result: a fresh deployment appears on the dashboard before its
  first cache.get / cache.set, which is what D7's ConnectedOssOverview
  needs to render.

- **`pyproject.toml` — `authors` block + `Changelog` URL** (#25).
  `pip show sulci` and the PyPI sidebar now surface
  `Author: Kathiravan Sengodan` and a direct link to `CHANGELOG.md`.
  The other `[project.urls]` entries that were already in place
  (Homepage / Repository / Documentation / Bug Tracker) are unchanged.

### Changed

- **`Cache._stats["hits"]/["misses"]` increment inside `Cache.get()`,
  not inside `Cache.cached_call()`** (#42). Users who use the raw
  `.get()` / `.set()` API previously saw `stats() == {"hits": 0,
  "misses": 0, "total_queries": 0}` regardless of activity, because
  the counters only fired through `cached_call()`. They now reflect
  every `.get()` call. `cached_call()` no longer increments them
  itself — it goes through `.get()` like everyone else, so existing
  hit/miss counts from `cached_call()`-only callers are identical to
  before. `saved_cost` stays a `cached_call()`-only metric, since
  raw `.get()` doesn't know what an LLM call would have cost.

  **Behavior change to flag:** if you have assertions against
  `stats()` that assumed raw `.get()` was a no-op for stats, those
  assertions will need to be updated.

### Fixed

- **`examples/` are now idempotent across re-runs** (#19).
  `basic_usage.py`, `anthropic_example.py`, `context_aware.py`, and
  `context_aware_example.py` each now use a per-run
  `tempfile.mkdtemp(prefix="sulci_<demo>_")` for `db_path` instead of
  inheriting the SQLite backend's default `./sulci_db` (which polluted
  the repo working tree) or hardcoding `/tmp/sulci_ctx_demo*` (which
  carried state across runs). `async_example.py` and
  `llamaindex_example.py` already used this pattern; no change there.

- **`examples/` fail fast with a useful message when an API key is
  rejected** (#20). `anthropic_example.py` and `async_example.py` now
  catch `anthropic.AuthenticationError` and `openai.AuthenticationError`
  on the first real call, print a one-line "key rejected — verify at
  <provider URL>" message, and fall back to the mock LLM for the rest
  of the demo. Previously a stale or wrong key surfaced as a raw
  `HTTPStatusError` traceback mid-output. The integration examples
  (`langchain_example.py`, `llamaindex_example.py`) already cascade
  across providers and survive missing-SDK gracefully; extending the
  same rejection-path coverage to them is a sibling follow-up since
  their provider-detection structure differs.

### Tests

- **+5 unit tests in `tests/test_telemetry.py::TestFlushIntegration`**
  for the new startup-event branch:
  - `test_startup_only_buffer_emits_one_post` — replaces the legacy
    `test_startup_only_buffer_does_not_post`, which encoded the bug.
  - `test_startup_with_cache_get_emits_two_posts_sharing_fingerprint`
    — codifies the dashboard-join invariant (startup + cache.get from
    the same flush share a fingerprint).
  - `test_startup_sniffs_backend_from_non_startup_event` — backend
    propagates from cache.set in the same batch into the startup row.
  - `test_startup_payload_only_contains_wire_fields` — defense against
    leaking SDK-internal keys past the gateway's `extra='forbid'`.
  - `test_multiple_startup_events_collapse_to_one_post` — defensive
    against any future "connect-after-disconnect" flow that buffers
    multiple startups.
- **+4 unit tests in `tests/test_core.py::TestStats`** for the raw
  `.get()` / `.set()` stats path:
  - `test_raw_get_miss_increments_misses`
  - `test_raw_set_then_raw_get_increments_hits`
  - `test_no_double_counting_via_cached_call` — regression guard for
    the increment-moved-into-get refactor.
  - `test_saved_cost_only_from_cached_call` — invariant that raw API
    use must not contribute to the cost-savings metric.

### Closed issues

- sulci-oss #41 — POST startup events from `_flush()`
- sulci-oss #42 — `Cache.stats()` reports 0/0 for raw `.get()`/`.set()` users
- sulci-oss #20 — examples: fail fast on rejected API key with informative message
- sulci-oss #19 — examples: db_path pollution makes demos non-idempotent
- sulci-oss #25 — packaging: add `authors` block + `Changelog` URL to pyproject.toml

### Follow-ups not in scope here

- sulci-oss #48 (pre-commit smoke ~8 min on macOS) — not folded in;
  the fix has design surface of its own (force CPU on macOS vs prewarm
  cache vs mock embedder for smoke tests) that deserves a dedicated
  discussion.
- sulci-platform #55 (un-awaited AsyncMock in `test_oss_connect_authorize`)
  — platform-side fix; lands separately.
- Extending the #20 fail-fast pattern through `langchain_example.py` and
  `llamaindex_example.py`'s multi-provider cascade — kept out of this
  bundle to avoid touching the provider-detection structure.

---

## [0.5.3] — 2026-05-04

OSS-Connect device-code SDK client (D12). Ships **latent**: the code is in
place, but the surrounding pieces of the OSS-Connect funnel — the gateway
endpoints (sulci-platform `/v1/oss-connect/*`) and the dashboard
`/oss-connect` page — may not yet be deployed in your environment.

The default for the new `prompt` parameter is `False` for that reason.
**Setting `prompt=True` against an environment that hasn't announced
OSS-Connect availability is user error** — wait for the Sulci team's
release announcement that the full chain is live (gateway + dashboard)
before flipping it on. v0.6.0 will flip the default to `True` once the
full chain ships end-to-end.

> **Post-hoc note (added in `[Unreleased]` on 2026-07-06):** the flip
> promised in the paragraph above did **not** happen in v0.6.0 or any
> subsequent release. The full OSS-Connect chain _is_ live end-to-end
> (SDK + gateway D4/D4.5/D5 + dashboard `/oss-connect`, cutover
> 2026-05-08), but the default nonetheless remains `prompt=False`. See
> the `## [Unreleased]` → Docs entry at the top of this file for the
> rationale (non-interactive default is safe for LangChain / LlamaIndex /
> FastAPI / CI callers; v0.7.0's `Cache()` auto-connect covers the
> ergonomic path without a blocking browser prompt at import time).
> Passing `prompt=True` explicitly remains fully supported.


### Added

- **`sulci.oss_connect`** — RFC 8628 device-code flow client.
  - `run_device_code_flow(gateway_base, sdk_version, client_name)` — blocks
    until the user authorizes via browser, denies, or the 15-minute
    device_code expires. Polls `/v1/oss-connect/token` at the gateway-
    advertised interval. Honors RFC 8628 `slow_down` (interval += 5s).
  - Lazy-imported from `sulci/__init__.py` only on the no-key-found path,
    so `import sulci` cost is unchanged for users who never trigger it.
  - Module is named `oss_connect` (not `connect`) to avoid shadowing the
    public `sulci.connect()` function. See sulci-platform ADR 0014
    §"Naming" for the full chronology of why the platform's URL prefix
    moved from `cli` → `connect` → `oss-connect`.
- **`sulci.connect(prompt=False)`** — new keyword parameter. When `True`,
  if no api_key is found through args/env/config, runs the browser-based
  device-code flow. Default is `False` in v0.5.3; will flip to `True`
  in v0.6.0 once the full OSS-Connect chain ships end-to-end.
- **Four-step api_key resolution** in `sulci.connect()`:
  1. `api_key=` argument
  2. `SULCI_API_KEY` environment variable
  3. `~/.sulci/config` (persisted from a prior successful connect)
  4. Browser device-code flow (only if `prompt=True`)
  Step 3 is new in v0.5.3 — connect()'s previously documented
  resolution stopped at step 2.
- **`SULCI_GATEWAY` env var** — overrides the gateway base URL for the
  device-code flow (default `https://api.sulci.io`). Used for staging /
  local-dev environments. Resolved at module-import time so the same
  value is used by both telemetry and the new device-code flow.

### Changed

- **`sulci.connect()` signature** gains the `prompt: bool = False`
  parameter. Existing callers that pass `api_key=...` are unaffected.
- **`tests/test_connect.py`** — three tests in the new `TestDeviceCodeFlow`
  class that exercise the device-code-fires path now pass `prompt=True`
  explicitly. `test_connect_without_key_does_not_enable_telemetry` and
  `test_connect_does_not_start_thread_without_key` continue to call with
  `prompt=False` to assert the no-op behavior.

### Tests

- **+27 new tests** across two files:
  - `tests/test_oss_connect.py` (new) — 19 tests for the RFC 8628 client
    (httpx mocked; deterministic; covers `slow_down`, denied, expired,
    network-error retry, the `_safe_error_field` helper).
  - `tests/test_connect.py::TestDeviceCodeFlow` — 8 new tests for the
    integration in `connect()` (resolution order, persistence on success,
    `prompt=False` escape hatch, RuntimeError propagation, persist-failure
    non-blocking).
- **Test-gate fix** — `scripts/run_tests_per_file.py::DEFAULT_FILES`
  gains `tests/test_oss_connect.py` so `make checkin` covers the new
  module. Without this addition, the 19 tests would exist but never run
  in the gate (same shape of test-gate omission caught and fixed
  upstream in sulci-platform PR #50). Per-file runner total goes from
  312 to 331.

### Compatibility

- **Backward-compatible against v0.5.2.** Existing `sulci.connect(api_key=...)`,
  `sulci.connect()` with `SULCI_API_KEY` set, and `sulci.connect(telemetry=False)`
  all preserve their v0.5.2 semantics.
- **The new step 3 (`~/.sulci/config` resolution) is observable only when
  the user has previously called `sulci.connect()` and `~/.sulci/config`
  contains an `api_key` field.** v0.5.2 didn't write this field.
  Pre-existing v0.5.2 configs (which only have `machine_id`) read as
  step 3 returning `None`, identical to no config existing.

### Privacy

- **No new wire fields.** The device-code flow is a `POST` to
  `/v1/oss-connect/{device-code,token}` with `{sdk_version, client_name,
  device_code, grant_type}` — no telemetry, no metrics, no user content.
- **The raw `api_key` returned by the flow is persisted to
  `~/.sulci/config` (mode 0600)** — same path / mode the v0.5.2
  `machine_id` already uses. The file is never logged or transmitted.

### Latent feature explainer

In v0.5.3, calling `sulci.connect(prompt=True)` against an environment
where the gateway hasn't deployed `/v1/oss-connect/*` endpoints will:

  1. Hit a 404 on `POST /v1/oss-connect/device-code`
  2. Raise `RuntimeError: sulci.connect() failed: could not request device code (HTTPStatusError: ...)`
  3. Leave `sulci._api_key = None` and telemetry disabled

If the gateway endpoints are deployed BUT the dashboard `/oss-connect`
page isn't:

  1. The SDK gets a `device_code` and prints `Visit {URL} and enter code: WXYZ-2345`
  2. The user follows the URL → 404 from the dashboard
  3. SDK polls for 15 minutes, then raises `RuntimeError: sulci.connect() timed out`

Both failure modes are clearly diagnosable. The default `prompt=False` is
designed to prevent users from discovering them by accident.

### Closed issues

- sulci-oss #35 (improvement 3) — device-code flow client, originally
  bundled with v0.5.2's improvements 1+2 but split out per launch-plan
  Phase Wave 2 sequencing.

### Wave 2 status (updated from v0.5.2 preview)

- ✅ **D12 — sulci-oss device-code client** (this release; latent)
- ✅ **D4 / D4.5 / D5 — gateway endpoints** (sulci-platform PR #51, pending merge + deploy)
- 🔲 **D7 — dashboard `/oss-connect` page** (sulci-platform; not yet started)
- ⏳ **v0.6.0 — promotion to production-ready** (after D7 merges + e2e validated)

### Naming chronology

The flow's URL/file naming went through two rename rounds at design time
in sulci-platform: `cli` → `connect` → `oss-connect`. The end-state
naming (`oss-connect` for URLs, `oss_connect` for Python identifiers)
is what's in this v0.5.3 release. The intermediate names do not appear
anywhere in the shipped code. Full chronology is in
`sulci-platform/docs/architecture/adrs/0014-restore-oss-connect-device-code.md`.

---

## [0.5.2] — 2026-04-30 — Connected-OSS telemetry wave 1: fingerprint + `cache.set` aggregation + opt-in nudge

Connected-OSS telemetry wave 1: per-deployment fingerprinting, `cache.set` aggregation,
opt-in nudge. Pairs with sulci-platform's already-shipped `/v1/telemetry`,
`/v1/analytics/deployments`, and `oss_connect` plan (gateway-side D1/D2/D3/D6/D9).
Wave 2 (`sulci.connect()` device-code flow) follows in v0.6.0 once the gateway's
`/v1/cli/device-code` and `/v1/cli/token` endpoints land.

### Added

- **`sulci.config`** — persistent SDK config at `~/.sulci/config`.
  - `load()` / `save()` / `update()` / `get_machine_id()` helpers.
  - File written with mode `0600`; directory `0700`. Atomic write via tempfile + rename.
  - Silent fallback on corruption — a malformed file never blocks `import sulci` or `Cache(...)`.
  - `get_machine_id()` generates a fresh `uuid4` on first call and persists it; same machine returns the same id forever after. Used as one input to the deployment fingerprint.
- **`sulci.telemetry`** — helpers for the legacy `connect()` emit pipe (distinct from the v0.5.0 `sulci.sinks.telemetry.TelemetrySink`, which is the per-event `EventSink` implementation — see module docstring for the disambiguation).
  - `build_fingerprint(machine_id, backend, embedding_model, threshold, context_window)` — stable, anonymous, config-aware deployment hash. 24 hex chars (12-byte blake2b).
  - `WIRE_FIELDS` — the exact 9-field allowlist accepted by the gateway `TelemetryEvent` schema. Imported into `_post()` as a final safety strip against any future flush() drift.
  - `coerce_to_wire(payload)` — strips non-allowlisted keys.
  - `python_version_str()` — version helper for the wire payload.
- **`fingerprint` field in `/v1/telemetry` payloads.** Resolves the `analytics.py` comment at line 103: *"v0.5.1 sends None"*. Now sends a stable per-deployment hash so the dashboard's "Active deployments" tile dedupes correctly across restarts.
- **`cache.set` events** are now buffered and POSTed as a separate aggregated batch per flush. Convention (documented in `_flush()`): `hits = number of set() calls aggregated`, `misses = 0`, `avg_latency_ms = average set() latency`. The gateway's TelemetryEvent schema already accepts `event='cache.set'`.
- **Passive nudge in `Cache.stats()`** — after 100 raw `.get()` calls on a Cache instance, prints a single stderr line suggesting `sulci.connect()`. One-shot per process; suppressed by `SULCI_QUIET=1` or by `sulci.connect()` already being active.

### Changed

- **`Cache.set()`** now records the per-call latency and emits a `cache.set` telemetry event when the instance has telemetry enabled and `sulci.connect()` has been called. The structured `EventSink` path (added in v0.5.0) is unchanged.
- **`Cache.get()`** emit payload now also carries `embedding_model`, `threshold`, and `context_window` keys so `_flush()` can compute the deployment fingerprint without coupling to a specific event type. These keys never reach the wire — `_post()` strips them via the `WIRE_FIELDS` allowlist.
- **`_flush()` rewritten** to handle multiple event types in one drain: emits up to two HTTP POSTs per flush (one for `cache.get`, one for `cache.set`), each carrying the deployment fingerprint. Empty-bucket short-circuiting preserved.

### Fixed

- None. v0.5.2 is purely additive.

### Privacy

- **No new wire fields beyond `fingerprint`**, which is a one-way hash containing no recoverable PII. Deriving the originating `machine_id` from a fingerprint requires brute-forcing a 96-bit blake2b — computationally infeasible.
- **Five new tests in `test_telemetry.py::TestPrivacyInvariants`** assert that `query`, `response`, and `embedding` fields are never sent on the wire even when poisoned events are placed directly in the buffer. Defense-in-depth against future regressions.
- **`coerce_to_wire()` is invoked in `_post()`** as a final safety strip — even if a future `_flush()` change accidentally constructs a payload with an extra key, the gateway's `extra='forbid'` rejection (HTTP 422) won't drop entire batches.

### Tests

- **+56 new tests** across three new files:
  - `tests/test_config.py` — 20 tests (1 skipped on root)
  - `tests/test_telemetry.py` — 24 tests
  - `tests/test_nudge.py` — 13 tests (covers threshold, one-shot, suppression, return-value invariants)
- **0 regressions** in pre-existing `tests/test_connect.py` (28/28 unit tests; 4 Cache-integration tests require a real embedder and run in CI).

### Compatibility

- **Fully backward-compatible.** Existing `sulci.connect(api_key=...)` flow unchanged. All v0.5.x callers continue to work.
- The `fingerprint` field is `Optional[str]` on the gateway side; older SDK versions sending `None` (or omitting it entirely) continue to be accepted.
- Nudge defaults to ON. Set `SULCI_QUIET=1` to silence; set it in CI before running tests against this version if any test asserts on clean stderr.

### Known limitations (deferred to follow-up issues)

- `_emit("startup", {})` events emitted by `connect()` are drained by `_flush()` but never POSTed — the legacy emit pipe lacks a `startup` HTTP path. The gateway schema already accepts `event='startup'`. Documented in `_flush()`'s docstring.
- `Cache._stats["hits"]/["misses"]` only increment in `cached_call()`, not in raw `.get()`. The new `_query_count` field works around this for the nudge logic, but the underlying `stats()` inconsistency remains.

### Closed issues

- sulci-oss #35 — SDK fingerprint emission.

### Wave 2 preview (v0.6.0)

`sulci.connect()` device-code flow, `sulci/cli.py`, `~/.sulci/config` API-key persistence
end-to-end. Blocked on sulci-platform `/v1/cli/device-code` and `/v1/cli/token` endpoints
(D4/D5) and the dashboard `/cli` authorization page (D7).

---

## [0.5.1] — 2026-04-28

### Added

- `RedisBackend(key_prefix=...)` constructor kwarg.
  - Defaults to `"sulci:"` (matches v0.4.x behavior — no breaking change for existing callers).
  - Replaces three previously-hardcoded `"sulci:*"` literals in `_key()`, the SCAN match pattern in `search()`, and the keys-glob in `clear()`.
  - Production callers can now pick a custom prefix to coexist with other Redis-using processes on a shared daemon (e.g., `RedisBackend(key_prefix="acme:cache:")`).

### Changed

- **CI matrix** — Python 3.10 now tested in `tests.yml` and `publish.yml`. Previously: `[3.9, 3.11, 3.12]`. Now: `[3.9, 3.10, 3.11, 3.12]`. Aligns CI coverage with `pyproject.toml` classifiers (which already claimed 3.10 support).
- `LOCAL_SETUP.md` Python-version hint reflects the new matrix.

### Fixed

- **Test fixtures (`backend_instance` in `tests/compat/conftest.py`)**: now clear state on setup, not just teardown. Defends against state leaked by any test that crashed before reaching teardown. SQLite/Qdrant fixtures get fresh `tmp_path`/collection per call so the setup clear is a no-op for them; matters for Redis where the daemon is shared across tests.
- **Test fixtures (`event_sink` in `sulci/tests/compat/conftest.py`)**: `RedisStreamSink` writes to a persistent Redis stream key that the fixture had no teardown for. Two changes: factory now `DEL`s the test stream key on construction; fixture now has a teardown that `DEL`s the stream when the implementation has a Redis client.
- **Redis test namespacing**: All Redis-backed tests now use a session-scoped key prefix (`sulci:test:<8-char-uuid>:`) instead of the production-default `sulci:`. Tests SCAN/MATCH only their own session's keys; sulci-platform's runtime data on the same Redis daemon is now safe during `make checkin` execution. Two concurrent `make checkin` runs against the same Redis no longer interfere.

### Verified

`make checkin` runs cleanly with sulci-platform Docker Compose stack active. No platform state corruption; no test-result corruption from platform writes.

### Compatibility

- **Fully backward-compatible.** All v0.5.0 code continues to work unchanged.
- The new `RedisBackend(key_prefix=...)` kwarg is purely additive; the default value matches v0.5.0 behavior. Honors the ADR 0005 protocol-stability commitment via additive-extension.
- 390 existing tests pass; no test count change in v0.5.1 (no new test files, only fixture and CI infrastructure changes).

### Closed issues

- #28 — Fixture: clear-on-setup pattern for backend_instance
- #29 — Namespace conformance test runs to prevent cross-project Redis interference
- #30 — Decide on Python 3.10 in CI matrix (Option A — added to both matrices)

### Phase 3 readiness

All four v0.5.1 blockers needed for Phase 3 entry are now closed: three in this release plus sulci-platform#12 (Dependabot triage). See `sulci-platform/docs/roadmap/PHASE-3-WORKSTREAM-C.md` for the gating list.

---

## [0.5.0] — 2026-04-27

### Added

- `sulci.sessions` package — SessionStore protocol and implementations
  - `SessionStore` — public stable protocol
  - `InMemorySessionStore` — default, process-local (extracted from sulci/context.py)
  - `RedisSessionStore` — Redis Lists-backed for horizontal scaling
- `sulci.sinks` package — EventSink protocol and implementations
  - `EventSink` — public stable protocol
  - `CacheEvent` — dataclass representing a cache event
  - `NullSink` — default no-op sink
  - `TelemetrySink` — HTTPS POST with strict field allowlist (never emits query/response/vectors)
  - `RedisStreamSink` — writes scrubbed events to a Redis Stream
- `Cache(session_store=..., event_sink=...)` — two new constructor kwargs
  - Both default to `None`, which uses `InMemorySessionStore()` and `NullSink()` respectively
  - Enables horizontal-scale deployments (via `RedisSessionStore`) and observability/billing (via any EventSink)
- `SyncCache` — alias for `Cache` exported from the top-level `sulci` namespace
  - Naming symmetry with existing `AsyncCache`
  - `sulci.SyncCache is sulci.Cache` returns True
- Conformance suites: `sulci.tests.compat.test_session_store_conformance` + `test_event_sink_conformance`

### Changed

- `sulci/__init__.py` exports `SyncCache` and the new session/sink primitives
- `Cache.__init__` gains `session_store` and `event_sink` kwargs (both `None` by default).
  When `session_store` is injected, Cache uses an internal bridge
  (`_ProtocolAdaptedSessionStore`) to translate between the new
  `sulci.sessions.SessionStore` protocol and the legacy `ContextWindow` surface
  Cache uses internally.
- `sulci/context.py` is **unchanged** — the legacy `SessionStore` class
  (higher-level ContextWindow manager) remains the default when no
  `session_store` kwarg is passed. See ADR 0007.

### Compatibility

- Fully backward-compatible. All v0.4.x code continues to work unchanged.
- 335+ existing tests pass + ~50 new tests added (sessions, sinks, conformance, injection).
- `AsyncCache` behavior unchanged. No async-native refactor.
- Defaults preserve exact v0.4.x behavior if new kwargs are not supplied.
- `from sulci.context import SessionStore` returns the **legacy** higher-level
  manager class (unchanged), not `sulci.sessions.InMemorySessionStore`. The
  bundle originally proposed aliasing them; we kept them separate to preserve
  v0.4.x behavior for direct importers. See ADR 0007 for the full rationale.
  When `Cache(session_store=<sulci.sessions.SessionStore impl>)` is injected,
  Cache adapts via an internal bridge (`_ProtocolAdaptedSessionStore`) that
  rebuilds a transient `ContextWindow` per lookup.

### Privacy

- `TelemetrySink` and `RedisStreamSink` enforce a strict field allowlist (`_ALLOWED_FIELDS` frozenset).
- The `CacheEvent.metadata` dict is NEVER shipped externally.
- Query text, response text, and embedding vectors NEVER leave the process via shipped sinks.

### Related ADRs

- ADR 0004 — SessionStore and EventSink protocols
- ADR 0007 — Preserve the legacy `sulci.context.SessionStore` class (B1 adapter)

### Roadmap

- See `docs/roadmap/FUTURE-DESIGN-OPTIONS.md` — v0.5.0 is additive by design.
  True async-native Cache refactor is deferred as roadmap item R2.

---

## [0.4.0] — 2026-04-26

### Added

- **Public Backend protocol** (`sulci/backends/protocol.py`) — formalizes the
  shape every vector-cache backend must satisfy. `runtime_checkable` Protocol
  with `store()`, `search()`, `clear()` methods. New `tenant_id` keyword-only
  parameter for multi-tenant partition isolation. STABLE API per ADR 0005.
- **Public Embedder protocol** (`sulci/embeddings/protocol.py`) — formalizes
  the shape MiniLMEmbedder and OpenAIEmbedder already had: `dimension`
  property, `embed(text)`, `embed_batch(texts)`. L2-normalization required.
- **`tenant_id` partition isolation** — first-class kwarg on `Cache.get()`,
  `Cache.set()`, and `Cache.cached_call()`. Forwarded to backend's `store`/
  `search` calls. Tenant isolation is a hard boundary — entries from other
  tenants must not be returned even when similarity exceeds threshold.
- **Keyword-only enforcement** (`*,` separator) on `Cache.get()`, `set()`,
  `cached_call()` — locks down `tenant_id`, `user_id`, `session_id`, and
  `metadata` as keyword-only to prevent positional misuse.
- **`ENFORCES_TENANT_ISOLATION` class attribute** on every backend, declaring
  whether `search()` filters by tenant_id. QdrantBackend = True (uses payload
  Filter); other shipped backends accept tenant_id as a label only.
- **Conformance test suite** (`tests/compat/`) — parametrized tests verifying
  that any class claiming to implement Backend or Embedder protocol satisfies
  the contract. Three groups: TestStructural (signature checks, runs always),
  TestRoundTrip (behavioral, runs when backend is constructable),
  TestTenantIsolation (runs only on backends with ENFORCES_TENANT_ISOLATION).
- **Qdrant tenant isolation tests** (`tests/test_qdrant_tenant_isolation.py`)
  — 11 tests across 8 customer-support scenarios (HelpDesk AI / Acme /
  Globex / Initech) verifying isolation guarantees end-to-end against an
  embedded Qdrant. Test names framed as product scenarios so failures
  describe user-impacting breakage.
- **`docs/protocols.md`** — Backend and Embedder protocol reference for
  developers extending sulci with custom backends or embedders.
- **`docs/multi_tenancy_and_isolation.md`** — OSS-layer trust and partition
  model. Generic customer scenarios, what's enforced where, FAQ on hashing,
  rotation, GDPR, encryption-at-rest.
- **`examples/extending_sulci/custom_backend.py`** — InMemoryBackend
  reference implementation. ~150 lines, in-memory dict-based, satisfies the
  full Backend protocol with self-test.
- **Developer tooling** (`scripts/`):
  - `run_tests_per_file.py` — runs pytest test files in fresh subprocesses
    (avoids MPS deadlock on Apple Silicon)
  - `run_examples.py` — runs every example + smoke test with timeout
  - `verify_integration_examples.py` — 8-scenario LLM provider matrix for
    langchain/llamaindex examples
  - `verify_benchmark.py` — runs canonical benchmark and verifies headline
    numbers haven't drifted from `benchmark/baseline.json`
- **`benchmark/baseline.json`** — canonical TF-IDF benchmark numbers from
  pre-v040-baseline. Used by verify_benchmark.py for regression detection.

### Changed

- **`__version__`** is now derived dynamically from `pyproject.toml` via
  `importlib.metadata.version("sulci")`. Previously hardcoded in three
  places (pyproject.toml, \_SDK_VERSION, USER_AGENT) which had drifted.
- **`_SDK_VERSION`** still exists (telemetry payload field name unchanged
  on the wire) but now equals `__version__`. Marked as deprecated alias.
- **`SulciCloudBackend.USER_AGENT`** now `f"sulci/{__version__}"` (was
  hardcoded "sulci/0.3.0", drifted by two minor releases).
- **`SulciCloudBackend.store()`** added (was missing — `cloud.py` only had
  `upsert()` while `core.py` always called `self._backend.store()`. Latent
  AttributeError on `Cache(backend='sulci').set()` is now fixed).

### Fixed

- **qdrant-client 1.x compatibility**: `QdrantBackend.search()` migrated
  from `client.search()` (removed) to `client.query_points()` with
  `.points` iteration. `QdrantBackend.clear()` now deletes points (preserves
  collection schema) instead of `delete_collection()` which broke subsequent
  operations on qdrant-client 1.x.
- **Cross-tenant data leak in `tenant_id=None` read path**: stores wrote
  `tenant_id="global"` for None, but searches with `tenant_id=None` added
  no filter, so unscoped reads silently returned named-tenant entries.
  Fixed by always filtering to "global" when None is passed. Caught by
  `test_named_tenant_entry_does_not_match_global_search`.
- **`examples/anthropic_example.py`** previously hardcoded `backend="chroma"`
  and documented `pip install "sulci[chroma]" anthropic` install line, but
  the README's quickstart recommends `sulci[sqlite]`. Mismatch caused
  ImportError on first run for users following the README. Switched to
  `backend="sqlite"` (functionally equivalent for this demo) and added
  graceful mock-LLM fallback when `ANTHROPIC_API_KEY` is unset.
- **`benchmark/.gitignore`** had a typo (`iresults/*.json`) that left
  benchmark output untracked-but-visible in `git status`. Fixed.

### CI

- `qdrant-client` added to `.github/workflows/tests.yml` install step.
- New CI steps: "Test Qdrant tenant isolation" and "Conformance suite" run
  early in the matrix to fail-fast on isolation regressions.

### Makefile

- New targets: `test-per-file`, `test-per-file-fast`, `examples`,
  `verify-integration-examples`, `benchmark-verify`, `checkin`. The
  `checkin` target chains smoke + tests + examples + benchmark-verify
  as a comprehensive pre-PR check (~7 min wall-clock).

### Notes

- `tenant_id` is honored ungated when passed (no `personalized` flag
  required). `user_id` continues to be gated by `personalized=True` for
  backwards compatibility with v0.3.x users; this asymmetry will be
  reconciled in v0.5.0+.
- After a version bump, run `pip install -e . --no-deps` in editable
  installs to refresh `importlib.metadata`'s cached dist-info.
- Built-in TF-IDF benchmark numbers verified byte-stable across the
  v0.3.x line and pre-v040-baseline (CI runs #26 through #36).
- Verified end-to-end via `make checkin`: 290 pytest tests pass, 12/12
  examples pass (including real OpenAI + Anthropic API calls), all 17
  benchmark metrics within tolerance vs baseline.

---

## [0.3.7] — 2026-04-11

### Added

- `sulci.AsyncCache` — non-blocking async wrapper around `sulci.Cache`.
  Delegates all cache operations to a thread pool via `asyncio.to_thread()`
  so the event loop is never blocked during embedding or vector search.
  Required for FastAPI, LangChain async chains, LlamaIndex async agents,
  and any asyncio-based application.
- `sulci/async_cache.py` — `AsyncCache` implementation
  - Async methods: `aget()`, `aset()`, `acached_call()`, `aget_context()`,
    `aclear_context()`, `acontext_summary()`, `astats()`, `aclear()`
  - Sync passthrough: `get()`, `set()`, `cached_call()`, `stats()`, `clear()`,
    `get_context()`, `clear_context()`, `context_summary()`
  - All constructor parameters identical to `sulci.Cache`
- `sulci/__init__.py` — `AsyncCache` exported, `_SDK_VERSION` bumped to `0.3.7`
- `smoke_test_async.py` — end-to-end async smoke test (24 checks)
- `examples/async_example.py` — AsyncCache demo with FastAPI pattern shown
  Supports OpenAI, Anthropic, or built-in mock LLM fallback

### Tests

- `tests/test_async_cache.py` — 25 tests (212 total, 205 passed, 7 skipped)
  - `TestConstruction` (4) — constructor passthrough, repr, invalid backend
  - `TestAget` (5) — hit, miss, session_id, user_id, 3-tuple return
  - `TestAset` (3) — stores entry, advances context window, session_id
  - `TestAcachedCall` (4) — hit, miss, dict shape, cost_per_call
  - `TestContextMethods` (4) — aget_context, aclear_context, acontext_summary,
    session isolation
  - `TestStats` (3) — astats dict shape, aclear resets stats, repr
  - `TestSyncPassthrough` (2) — sync get/set/stats still work on AsyncCache

### Makefile

- `make smoke-async` — AsyncCache smoke test only
- `make test-async` — `tests/test_async_cache.py` only
- `make smoke` updated — includes `smoke_test_async.py`
- `make test-all` updated — includes `tests/test_async_cache.py`

### Notes

- Zero breaking changes — `sulci.Cache` is unchanged
- Pattern: `asyncio.to_thread()` — idiomatic Python 3.9+, same approach
  used by LangChain `BaseCache.alookup()` and `SulciCacheLLM.acomplete()`
- Future v2: native async backends for Qdrant (`AsyncQdrantClient`) and
  Redis (`redis.asyncio`) when throughput demands justify the rewrite

---

## [0.3.6] — 2026-04-10

### Changed

- Version bump to re-release v0.3.5 content to PyPI — the v0.3.5 wheel was
  published from an earlier tag before examples and doc updates were committed.
  No code changes — library behaviour is identical to v0.3.5.

### Includes (carried from v0.3.5)

- `examples/langchain_example.py` — LangChain stateless + context-aware demo
- `examples/llamaindex_example.py` — LlamaIndex Settings.llm demo
- `LOCAL_SETUP.md` — Step 12, smoke-llamaindex, v0.3.5 references
- `README.md` — examples section, Project Structure updated

---

## [0.3.5] — 2026-04-09

### Added

- Native LlamaIndex LLM wrapper `SulciCacheLLM` — first correct LLM-level
  semantic cache for LlamaIndex. Wraps any `LLM` subclass (OpenAI, Anthropic,
  Ollama, HuggingFaceLLM, etc.). `complete()` and `chat()` are cached;
  streaming passes through uncached; async methods use `run_in_executor`.
- `sulci/integrations/llamaindex.py` — `SulciCacheLLM(LLM)` implementation
- `sulci/integrations/__init__.py` — updated with LlamaIndex entry
- `pyproject.toml` — `llamaindex = ["llama-index-core>=0.10.0"]` extra
- `smoke_test_llamaindex.py` at repo root

### Tests

- `tests/test_integrations_llamaindex.py` — 29 tests (TestConstruction,
  TestComplete, TestChat, TestStreaming, TestAsync, TestStats)

### Examples

- `examples/langchain_example.py` — two demos in one file:
  - Demo 1: stateless `set_llm_cache(SulciCache(...))` — semantic hit/miss
    across 4 rounds showing real API latency vs <10ms cache hits
  - Demo 2: context-aware `ContextAwareSulciCache` subclass using `llm_string`
    as `session_id` — two isolated user sessions (alice/bob), 58% hit rate
  - Supports OpenAI, Anthropic, or built-in mock LLM fallback
  - API key detection logged at startup (`✓ found` / `✗ not set`)

- `examples/llamaindex_example.py` — four rounds:
  - Round 1: fresh questions per session (all misses)
  - Round 2: paraphrases in same sessions (93-96% similarity hits, <7ms)
  - Round 3: context-aware follow-ups in a single topic session
  - Round 4: clearly unrelated question (clean miss)
  - `Settings.llm = SulciCacheLLM(...)` — idiomatic LlamaIndex pattern
  - Supports OpenAI, Anthropic, or built-in mock LLM fallback
  - API key detection logged at startup

### Notes

- GPTCache's claimed LlamaIndex integration was a broken global OpenAI API
  patch. SulciCacheLLM uses the idiomatic `LLM` subclass pattern and works
  with any LlamaIndex-compatible model.

---

## [0.3.4] — 2026-04-08

### Fixed

- `SulciCache`: `namespace_by_llm=True` now logs a warning and is silently
  disabled when `backend="sulci"`. Sulci Cloud handles tenant isolation
  server-side; `db_path`-based partitioning was creating phantom
  `SulciCloudBackend` instances with no effect.

### Added

- `SulciCloudBackend`: new `gateway_url` parameter (default: `https://api.sulci.io`).
  Enterprise VPC customers can point to a self-hosted gateway:
  `Cache(backend="sulci", api_key="...", gateway_url="https://cache.acme.internal")`
- `Cache`: `gateway_url` threaded through `_load_backend()` when `backend="sulci"`.
- `SulciCache` (LangChain): `gateway_url` documented in `**kwargs` table.

### Tests

- `test_cloud_backend.py`: 3 new tests — default gateway URL, custom gateway URL,
  trailing slash stripping
- `test_integrations_langchain.py`: 3 new tests — `TestNamespaceByLLMCloudWarning`

---

## [0.3.3] — 2026-04-08

### Added

**LangChain integration — context-aware semantic cache adapter**

- `sulci/integrations/__init__.py` — new `integrations` sub-package
- `sulci/integrations/langchain.py` — `SulciCache(BaseCache)` for LangChain
  - Positioned as the **context-aware semantic cache** — distinct from stateless
    semantic caches (GPTCache, RedisSemanticCache) already in langchain-community
  - `lookup(prompt, llm_string)` — semantic match via `sulci.Cache.get()`,
    returns `list[Generation]` on hit, `None` on miss
  - `update(prompt, llm_string, return_val)` — stores first `Generation.text`
  - `clear()` — evicts data and resets namespace dict via `finally` block
    (guarantees `_ns_caches` is always cleared even if a data-clear raises)
  - `namespace_by_llm=True` (default) — separate cache partition per LLM config;
    uses MD5-hashed `db_path` suffix for local backends
  - `alookup`, `aupdate`, `aclear` — async overrides via `run_in_executor`
  - Silent failure throughout — cache errors never raise to the caller's app
  - `stats()` — passthrough to `sulci.Cache.stats()`
  - Lazy import of `langchain-core` — raises `ImportError` with install hint
    if not installed; core `sulci` package never depends on LangChain
  - `langchain_core.globals` used (not `langchain.globals`) — only `langchain-core`
    required, not the full `langchain` package

**LangChain integration — tests**

- `tests/test_integrations_langchain.py` — 24 tests, zero LLM API keys required
  - `TestContract` (9) — lookup/update/clear/exact-hit/semantic-miss/list-return
  - `TestNamespacing` (4) — model isolation, shared mode, clear resets dict
  - `TestSilentFailure` (3) — db errors in lookup/update/clear never raise
  - `TestAsync` (4) — alookup/aupdate/aclear/concurrent reads
  - `TestStats` (3) — dict shape, required keys, repr format
  - `TestGlobalRegistration` (1) — `set_llm_cache` / `get_llm_cache` round-trip

**LangChain integration — smoke test**

- `smoke_test_langchain.py` — standalone smoke test at repo root
  - Runs automatically via `setup.sh` after core smoke test
  - Skips gracefully (exit 0) if `langchain-core` is not installed
  - Covers: create → store → exact hit → unrelated miss → stats

**Developer tooling**

- `setup.sh` — updated to install `.[langchain]` extra and run both smoke tests
  sequentially; `Next steps` section updated to list actual `make` targets
- `Makefile` — new targets:
  - `make smoke` — runs `smoke_test.py` + `smoke_test_langchain.py`
  - `make smoke-core` — core smoke test only
  - `make smoke-langchain` — LangChain smoke test only
  - `make test` — core pytest suite
  - `make test-integrations` — LangChain + LlamaIndex integration tests
  - `make test-all` — full suite
  - `make test-cov` — full suite with coverage report
  - `make verify` — `smoke` + `test-all` (pre-commit full check)

**LangChain community PR artifact**

- `langchain_community_pr/sulci_cache_addition.py` — ready-to-paste addition
  for `langchain_community/cache.py` PR to `langchain-ai/langchain`

### Changed

- `pyproject.toml` — version bumped to `0.3.3`
- `pyproject.toml` — added `langchain = ["langchain-core>=0.1.0"]` optional extra
- `pyproject.toml` — added `pytest-asyncio==0.21.1` to `dev` deps
  (pinned — 0.23.x has a package collection bug)
- `pyproject.toml` — added `asyncio_mode = "auto"` to `[tool.pytest.ini_options]`
- `pyproject.toml` — added `"context-aware-semantic-cache"` keyword for PyPI search
- `sulci/__init__.py` — `_SDK_VERSION` bumped from `"0.3.0"` to `"0.3.3"`
  (was already out of sync with pyproject.toml since 0.3.1)

### Fixed (discovered during integration test development)

- `sulci/integrations/langchain.py` `clear()` — moved `_ns_caches.clear()` into
  a `finally` block so namespace dict is always reset even if a backend `clear()`
  raises an exception
- `tests/test_integrations_langchain.py` — assertion order in
  `test_clear_removes_all_partitions` corrected: `len(_ns_caches) == 0` must be
  checked _before_ any `lookup()` call, since `lookup()` calls `_cache_for()`
  which recreates namespace entries for any `llm_string` it encounters
- `tests/test_integrations_langchain.py` — `test_concurrent_lookups_no_crash`
  revised to check no exceptions are raised (not that all 20 concurrent SQLite
  reads return non-None — a single connection under high concurrency may return
  miss on some reads, which is acceptable behaviour)
- `tests/test_integrations_langchain.py` — `TestGlobalRegistration` import changed
  from `langchain.globals` to `langchain_core.globals` — only `langchain-core` is
  required, not the full `langchain` package

### Backward compatibility

- All existing code using local backends (`sqlite`, `chroma`, `faiss`, etc.)
  is completely unaffected — zero breaking changes
- `context_window=0` (default) remains stateless and identical to prior versions
- New `integrations` sub-package is purely additive — not imported unless
  explicitly requested by the caller

### Test count after this release

```
test_core.py                       27 tests
test_context.py                    35 tests
test_backends.py                    9 tests  (skipped if backend dep not installed)
test_connect.py                    32 tests
test_cloud_backend.py              25 tests
test_integrations_langchain.py     24 tests  ← new
────────────────────────────────────────────
Total                             152 tests
```

---

## [0.3.2] — 2026-03-27

### Patent & Legal

- Updated NOTICE file with US Patent Application No. 64/018,452
- Added Patent Pending badge and notice to README
- Updated PyPI description to include Patent Pending

### No code changes — library behaviour is unchanged

---

## [0.3.1] — 2026-03-27

### License

- Changed from MIT License to Apache License 2.0
- Added NOTICE file as required by Apache 2.0
- Updated pyproject.toml classifier to Apache Software License
- Added SPDX identifiers to all Python source files
- Rationale: Apache 2.0 includes patent retaliation clause and explicit
  patent grant; aligns with pending patent application IDF-SULCI-2026-001

### No code changes — library behaviour is unchanged

---

## [0.3.0] — 2026-03-25

### Added

- **Sulci Cloud backend** — `Cache(backend="sulci", api_key="sk-sulci-...")` routes
  cache operations to `api.sulci.io` via HTTPS. Zero infrastructure for the user —
  one parameter change from any self-hosted backend.
- `sulci/backends/cloud.py` — `SulciCloudBackend` via httpx
  - `search()` returns `(None, 0.0)` on timeout or any error — never crashes caller
  - `upsert()` failure is silent — fire and forget
  - `delete_user()` and `clear()` also fail silently
- `sulci.connect(api_key, telemetry=True)` — opt-in gateway to Sulci Cloud
  - Stores API key at module level for all `Cache(backend="sulci")` instances
  - Enables optional usage telemetry — flushed to `api.sulci.io` every 60 seconds
  - Strictly opt-in: `_telemetry_enabled = False` until `connect()` is called
- `Cache` gains two new constructor parameters:
  - `api_key` — API key for `backend="sulci"` (resolution: arg > env > `connect()`)
  - `telemetry` — per-instance opt-out (default `True`)
- `SULCI_API_KEY` environment variable — zero-code alternative to `api_key=`
- `sulci[cloud]` install extra — `pip install "sulci[cloud]"`
- `tests/test_connect.py` — 32 tests covering `sulci.connect()` and telemetry
- `tests/test_cloud_backend.py` — 25 tests covering `SulciCloudBackend` and wiring

### Changed

- Version bumped to `0.3.0`
- `README.md` updated with Sulci Cloud section and `sulci.connect()` docs
- `LOCAL_SETUP.md` updated with Week 2 and Week 3 setup instructions
- `pyproject.toml` — added `cloud = ["httpx>=0.27.0"]` extra

### Backward compatibility

- All existing code using local backends (`sqlite`, `chroma`, `faiss`, etc.) is
  completely unaffected — zero breaking changes
- `connect()` and `api_key=` are purely additive
- Default backend behaviour unchanged

---

## [0.2.5] — 2026-03-17

### Repository & Housekeeping

- Transferred repository from `id4git/sulci` to `sulci-io/sulci-oss` under new GitHub org
- Renamed repo from `sulci` to `sulci-oss` (PyPI package name `sulci-cache` and import `from sulci` unchanged)
- Added `LICENSE` (MIT) and `NOTICE` files to repo root with clear OSS/enterprise demarcation
- Updated `pyproject.toml` repository URLs to reflect new org and repo name

### Docs

- Added `LOCAL_SETUP.md` — full local development guide: venv setup, install, test runs, smoke test, troubleshooting
- Corrected test counts across `README.md` and `LOCAL_SETUP.md`:
  - `test_core.py`: 27 tests (was 26)
  - `test_context.py`: 35 tests (was 27)
  - `test_backends.py`: 9 tests (was unknown)
  - Total: 71 tests (was 53)
- Updated project structure tree in both docs to match actual repo layout (7 directories, 29 files)
- Removed inline changelog table from `README.md` — full history lives in `CHANGELOG.md`
- Fixed `pyproject.toml` comment to correctly distinguish repo root (`sulci-oss/`) from package directory (`sulci/`)

### No code changes — library behaviour is identical to 0.2.4

---

## [0.2.4] — 2026-03-16

- Release v0.2.4 — Developer Edition baseline — pre-enterprise transition

---

## [0.2.3] — 2026-03-16

- Release v0.2.3 — correct test counts, updated docs

---

## [0.2.2] — 2026-03-15

- Packaging fix: re-publish of 0.2.1 (PyPI file conflict resolution)

---

## [0.2.1] — 2026-03-11

- Context-aware benchmark suite: `--context` flag
- 25 session pools, brute-force cosine scan
- Results: +20.8pp resolution accuracy

---

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

- Initial release — 6 backends, MiniLM, TTL, personalization, stats
