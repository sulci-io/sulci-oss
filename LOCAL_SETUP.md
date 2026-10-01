# Sulci Cache — Local Setup Guide

Everything you need to clone the repo, install dependencies, run tests, and verify a working local environment from scratch.

---

> **Fresh-machine run-through (2026-09-25, M2 MacBook Air, macOS, Python 3.12, v0.9.1;
> test and check-in counts re-measured on v0.9.2, 2026-10-01):**
> Steps 1–8.5 and 11–13 were followed literally on a clean `~/code` and
> corrected where they broke; every command in those steps has been run as now
> written. The reference sections from *Troubleshooting* onward were checked
> against the repo on 2026-09-30. In Steps 9–10, every keyless snippet and test
> command was run on 2026-10-01; the parts that need a live gateway and a key
> (`connect()` with a real key, `prompt=True`, the end-to-end staging smoke)
> are optional checks for account holders and have **not** been re-run.

## Current state — re-measured 2026-09-30

| Fact | Value | Re-measure |
|---|---|---|
| Version | **0.9.2** (tag `v0.9.2`, on PyPI) | `grep '^version' pyproject.toml` |
| Public methods on `Cache` | **8** | `python3 scripts/check_api_surface.py --show` |
| Default backend | `"chroma"` | ditto — **not** sqlite |
| Default `ttl_seconds` | `86400` — entries **do** expire after 24h | ditto |
| Default `db_path` | `"./sulci_db"` | ditto |
| Backends | 6 free + 1 managed | |

`python3 scripts/check_api_surface.py --show` measures the whole surface by AST
(no install needed); without `--show` it exits 1 if
[`docs/API-SURFACE.md`](docs/API-SURFACE.md) has drifted from the code. **If a document anywhere disagrees with that
command's output, the output wins.** Four of these defaults were documented
wrongly for months — two of them behaviourally, so a reader believed the default
backend was SQLite and that entries never expire.

---

## Shell and tooling gotchas

These are not hypothetical; each has cost real debugging time.

- **zsh needs quoted globs.** `pip install "sulci[sqlite]"` fails without the
  quotes, because zsh treats `[...]` as a glob. Same reason
  `grep -r --include=*.jsx` fails with "no matches found" — write
  `--include='*.jsx'`.
- **zsh passes `# comments` to the command unless told otherwise.** Many code
  blocks here end lines with `# explanation`. Interactive zsh does not treat `#`
  as a comment by default, so pasting `which python   # should show …` hands
  `#`, `should`, `show` to `which` as arguments (`gh` fails with
  `accepts at most 1 arg(s), received 9`). Fix once:
  `echo 'setopt interactivecomments' >> ~/.zshrc`, then open a new tab.
- **`find` may be aliased to `fd`.** If `find src -type f` fails with a `--type`
  error, that is the alias. Use `\find`, and likewise `\cat` / `\ls` if `bat` /
  `eza` are aliased.
- **Bare `pip` is not on `PATH` under macOS Command Line Tools Python.** Use
  `python3 -m pip`.
- **After a version bump, an editable install still reports the old version.**
  `importlib.metadata` reads cached metadata; run `pip install -e . --no-deps` to
  refresh it. Otherwise `sulci.__version__` lies to you and you will chase it.
- **If your prompt's version segments suddenly go blank, that is `$PATH`** —
  clobbered by an `export PATH="…"` that assigned rather than prepended, dropping
  Homebrew. Open a new tab rather than patching the current one.
- **macOS Apple Silicon test flakiness** is an MPS deadlock, not a real failure.
  Run `scripts/run_tests_per_file.py`, which uses a fresh subprocess per file.
- **`pytest: command not found`** → `python -m pytest tests/ -v`.

---

## Requirements

- Python **3.9, 3.10, 3.11, or 3.12** — all four are tested in CI; nothing
  newer is. **Use 3.12 for local development.** The `mcp` and `litellm` extras
  need 3.10+, so 3.9 cannot run the whole suite.
- `git`

**macOS: your default `python3` is probably too new.** Homebrew's `python3`
tracks the latest release (3.14 as of 2026-09), and a stock Mac has no `python`
command at all. Check, and install 3.12 alongside it if needed:

```bash
python3 --version              # 3.13 or newer → install 3.12 below
brew install python@3.12       # provides the python3.12 command
python3.12 --version           # should print Python 3.12.x
```

**direnv users:** the repo ships an `.envrc` that activates `.venv`. Until
Step 2 creates the venv, entering the directory prints
`.envrc:1: .venv/bin/activate: No such file or directory` (or, on a machine
that has never allowed it, `.envrc is blocked`). Both are expected — carry on
and run `direnv allow` at the end of Step 2. If you don't use direnv, ignore
the file.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/sulci-io/sulci-oss.git
cd sulci-oss
```

All active development is on `main`. Feature branches are short-lived and merge
to `main` via PR.

---

## Step 2 — Create and Activate a Virtual Environment

Always use a virtual environment. Never install Sulci dependencies into your system Python.

Create the venv with an explicit 3.12 interpreter. On macOS, `python` does
not exist outside a venv, and plain `python3` may be 3.13+ (see Requirements).

```bash
# create — name the interpreter explicitly
python3.12 -m venv .venv

# activate — macOS / Linux
source .venv/bin/activate

# activate — Windows
.venv\Scripts\activate

# confirm you're inside the venv
which python        # should show .venv/bin/python
python --version    # should be 3.12 (3.9–3.11 also work; 3.13+ is untested)
```

Inside an activated venv, `python` and `pip` point at the venv, so the bare
`python` / `pip` commands in the rest of this guide work as written.

**direnv users:** run `direnv allow` now. From here on the venv activates
automatically whenever you `cd` into the repo.

**Every new terminal:** all commands in this guide run from the repo root with
the venv active. In a fresh tab, first run `cd sulci-oss` (direnv then
activates the venv) or `cd sulci-oss && source .venv/bin/activate`. If you see
`command not found: pip` or `python`, you skipped this.

---

## Step 3 — Install the Library

Install in editable mode (`-e`) so any changes you make to `sulci-oss/` source code are reflected immediately without reinstalling.

**New developers: run these two commands and move on to Step 4.**

```bash
pip install --upgrade pip        # the venv's bundled pip is often stale
pip install -e ".[sqlite,redis,qdrant,langchain,llamaindex,mcp,litellm,proxy,dev]" pytest-timeout
```

This is the full dev setup, matching what CI installs for the test suite.
**Don't trim this list.** Some tests fail rather than skip when an extra is
missing: without `mcp` or `litellm`, `pytest tests/` stops at collection with
two errors; without `redis`, three `TestRedisStreamSink` tests fail. The
`redis` extra is needed even though those tests mock the Redis server.
`pytest-timeout` backs the `@pytest.mark.timeout` markers in the tests.

The install pulls in PyTorch via `sentence-transformers`, so expect a few
minutes; the resulting `.venv` is about 1.4 GB (measured on an M2 Mac,
Python 3.12, 2026-09-25). Step 4's import check needs at least the `langchain` and
`llamaindex` extras.

> **zsh users:** always wrap extras in quotes — `".[sqlite]"` not `.[sqlite]`.
> Without quotes, zsh treats the brackets as a glob pattern and throws `no matches found`.

### Optional extras (reference — not needed to get started)

Add any of these later with `pip install -e ".[name]"`. Combine several in one
set of brackets, e.g. `".[sqlite,chroma,faiss]"`.

| Extra | Adds |
|---|---|
| `sqlite` | local SQLite backend — zero infra, fully offline |
| `langchain` | LangChain adapter (`langchain-core` only, not full langchain) |
| `llamaindex` | LlamaIndex wrapper |
| `chroma` / `faiss` / `qdrant` / `redis` / `milvus` | other vector backends |
| `openai` | OpenAI embeddings |
| `mcp` / `litellm` / `proxy` | v0.9.0 integration surfaces — **Python 3.10+ only** |
| `dev` | pytest, coverage, build/twine |

`AsyncCache` is part of the base install and works with any backend — no extra
required. `httpx` is a core dependency (since v0.6.3), so it is always installed.

---

## Step 4 — Verify the Install

```bash
python -c "
from sulci import Cache, ContextWindow, SessionStore, connect
from sulci.backends.cloud import SulciCloudBackend
from sulci.integrations.langchain import SulciCache
from sulci.integrations.llamaindex import SulciCacheLLM
from sulci import AsyncCache
print('Import OK')
"
```

Expected output:

```
Import OK
```

If you see a `ModuleNotFoundError` on a backend (e.g. `chromadb`, `faiss`), that backend's
extra is not installed. Install it with `pip install -e ".[backend_name]"`.

If you see `ModuleNotFoundError: langchain_core`, install the langchain extra:

```bash
pip install -e ".[langchain]"
```

If you see `ModuleNotFoundError: llama_index`, install the llamaindex extra:

```bash
pip install -e ".[llamaindex]"
```

---

## Step 5 — Run the Tests

Always use `python -m pytest` rather than bare `pytest` to avoid PATH issues.

```bash
python -m pytest tests/ -v -rs
```

`-rs` prints a summary of *why* each test skipped at the end. Don't pipe the
run through `| tail`: `tail` shows nothing until pytest exits, which looks
exactly like a hang. To keep a log and still watch progress, use
`python -m pytest tests/ -v -rs 2>&1 | tee /tmp/sulci-tests.log`.

### Expected result

v0.9.2, M2 Mac, Python 3.12, offline, no Redis or Qdrant server running.
The suite collects **739** tests.

```
702 passed, 37 skipped, 0 failed     # Step 3 install + Step 8.5 agent packages (measured 2026-10-01)
697 passed, 42 skipped, 0 failed     # Step 3 install only (derived: see below)
```

Skips are expected. Re-measure the current count with
`python -m pytest tests/ --collect-only -q | tail -1`. The Step 8.5 agent
packages pull in `chromadb`, which turns 5 Chroma skips into passes; without
them, expect the second line. The 37 skips in the measured run break down as:

| Skips | Reason shown by `-rs` | To run them |
|---|---|---|
| 14 | `redis-server not running on localhost:6379` / `RedisBackend: no local construction available` | Start a local Redis (see *Redis-dependent tests* at the end of this guide) |
| 7 | `faiss-cpu not installed` / `FAISSBackend: no local construction available` | `pip install -e ".[faiss]"` |
| 7 | `pymilvus not installed` / `MilvusBackend: no local construction available` | `pip install -e ".[milvus]"` (runs locally, no server) |
| 5 | `OpenAIEmbedder: no local construction available` | Set `OPENAI_API_KEY` — makes real, billed API calls |
| 4 | `SQLiteBackend` / `ChromaBackend does not enforce tenant isolation` | Never — by design, the contract doesn't apply |

**Any failure or collection error on a fresh setup is a real problem** — most
often a missing extra (see the warning in Step 3).

### Runtime and the network

The suite loads the embedding model (`all-MiniLM-L6-v2`) many times, and by
default each load checks Hugging Face for updates. Runtime therefore depends
on your connection, not your CPU:

| Mode | Full suite, M2 MacBook Air |
|---|---|
| Offline (`HF_HUB_OFFLINE=1`) | **~70–85 s** (67.8 s at v0.9.1, 83.1 s at v0.9.2) |
| Online, ordinary connection | ~7 min |
| Online, in-flight Wi-Fi | 72 min at ~3% CPU — looks like a hang, isn't |

The **first** run must be online, to download the model (~90 MB, cached in
`~/.cache/huggingface`). After that, always run offline:

```bash
HF_HUB_OFFLINE=1 python -m pytest tests/ -v -rs
```

Consider adding `export HF_HUB_OFFLINE=1` to your shell profile once the model
is cached; unset it only when you need a model you haven't downloaded yet.

**For a fully offline run, also set `LITELLM_LOCAL_MODEL_COST_MAP=True`.**
Importing `litellm` (the `litellm` extra) otherwise downloads its model price
list from `raw.githubusercontent.com` during test collection; with this set it
uses the copy bundled in the package.

```bash
export HF_HUB_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True
```

If one test sits for several minutes at near-0% CPU **with**
`HF_HUB_OFFLINE=1` set, see the Apple Silicon (MPS) note in the gotchas
section and use the per-file runner.

### Targeted test runs

```bash
# core cache logic only
python -m pytest tests/test_core.py -v

# context and session tests only
python -m pytest tests/test_context.py -v

# backend tests only
python -m pytest tests/test_backends.py -v

# telemetry + sulci.connect() tests only
python -m pytest tests/test_connect.py -v

# v0.5.2 — config / telemetry / nudge tests only
python -m pytest tests/test_config.py tests/test_telemetry.py tests/test_nudge.py -v

# v0.5.3 — OSS-Connect device-code client tests
python -m pytest tests/test_oss_connect.py -v

# SulciCloudBackend + Cache wiring tests only
python -m pytest tests/test_cloud_backend.py -v

# LangChain integration tests only
python -m pytest tests/test_integrations_langchain.py -v

# LlamaIndex integration tests only
python -m pytest tests/test_integrations_llamaindex.py -v

# single backend by keyword
python -m pytest tests/test_backends.py -v -k sqlite
python -m pytest tests/test_backends.py -v -k chroma

# one specific test by name
python -m pytest tests/test_core.py::TestBasicOperations::test_semantic_hit -v

# stop at first failure
python -m pytest tests/ -v -x

# with line-level coverage report
python -m pytest tests/ -v --cov=sulci --cov-report=term-missing
```

### Make targets

```bash
make test               # core pytest suite (excludes integrations)
make test-integrations  # LangChain + LlamaIndex integration tests
make test-async         # AsyncCache tests only
make test-all           # full suite
make test-cov           # full suite with coverage report
make verify             # smoke + test-all (run before committing)
```

---

## Step 6 — Run the Examples

Every example below except `anthropic_example.py` runs fully offline with no
API keys set (mock LLM), about 7–11 s each — verified
2026-09-25 on an M2 with `HF_HUB_OFFLINE=1`. Each exits 0 and ends with a
stats summary (hit rate, cost saved, LLM calls made).

### No API key required

```bash
# stateless cache demo
python examples/basic_usage.py

# context-aware demo — 4 walkthroughs, fully offline
python examples/context_aware.py

# additional context-aware patterns
python examples/context_aware_example.py
```

### Requires `ANTHROPIC_API_KEY`

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/anthropic_example.py    # Anthropic Claude + context-aware
```

### LangChain and LlamaIndex integration examples

These work with OpenAI, Anthropic, or a built-in mock LLM — API key is optional.

```bash
# set one or both keys (optional — mock LLM used if neither is set)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

python examples/langchain_example.py    # LangChain: stateless + context-aware demo
python examples/llamaindex_example.py   # LlamaIndex: Settings.llm = SulciCacheLLM
python examples/async_example.py        # AsyncCache demo — FastAPI pattern shown
```

Each example prints which LLM is active at startup:

```
── API key detection ──────────────────────────────
  OPENAI_API_KEY    : ✓ found
  ANTHROPIC_API_KEY : ✗ not set
  → Using: OpenAI gpt-4o-mini
```

Priority: OpenAI → Anthropic → mock. To force Anthropic: `unset OPENAI_API_KEY`.

---

## Step 7 — Run the Benchmark

```bash
# regression check — TF-IDF path vs benchmark/baseline.json (~19 s on an M2, no network)
python scripts/verify_benchmark.py        # or: make benchmark-verify

# fast exploratory runs — TF-IDF, ~10 s each; not the shipped engine, do not cite.
# Note the --out: see the warning below.
python3 benchmark/run.py --no-sweep --queries 1000 --out /tmp/sulci-bench-q1000
python3 benchmark/run.py --no-sweep --queries 1000 --context --out /tmp/sulci-bench-q1000

# THE CANONICAL RUN — shipped engine (MiniLM), stateless + context (~10 min)
# (the sqlite extra it needs is already installed if you followed Step 3)
python3 benchmark/run.py --use-sulci --fresh --no-sweep --context
```

⚠️ **Give non-default runs their own `--out`.** Without it, a `--queries 1000`
run writes to `benchmark/results/tfidf/`, the same directory the regression
check uses at the default 5000 queries. The next `verify_benchmark.py` then
refuses to overwrite it (`VARIANT COLLISION … --queries: on disk '1000'`) and
exits with code 2 — `make checkin` included. If that has already happened,
`rm -rf benchmark/results/tfidf` (nothing in it is tracked by git) and re-run.

⚠️ **`--use-sulci` is opt-in and everything above it measures a built-in
TF-IDF engine that ships in no product.** It is kept because a 4-second
regression check on every check-in is worth having, not because it measures
the product. Anything you would show someone needs `--use-sulci`.

Results are written to `benchmark/results/tfidf/` or `benchmark/results/minilm/`
— the default output directory is suffixed with the engine, so a fast TF-IDF
run can no longer overwrite a MiniLM run's `summary.json`. An explicit `--out`
is used verbatim. The `.gitignore` in that directory excludes `*.json` and
`*.csv` so result files are never committed.


⚠️ `--seed`'s default is `42` **in effect, not in argparse**. `argparse` supplies
`None`; `run.py:146` reseeds only when the value is not `None`, leaving the
module RNG at the `random.seed(42)` from `:86`. Running without `--seed` is
therefore deterministic and reproduces `baseline.json` exactly —
`scripts/verify_benchmark.py` confirms 17 metrics at Δ=0.0000.

📌 The four committed draws behind every published figure are at
`benchmark/results/minilm/seed-{1,2,3,42}`.

### All benchmark flags

| Flag                    | Default             | Description                                       |
| ----------------------- | ------------------- | ------------------------------------------------- |
| `--context`             | off                 | Enable context-aware benchmark pass               |
| `--no-sweep`            | off                 | Skip threshold sweep (much faster)                |
| `--queries N`           | 5000                | Number of test queries                            |
| `--threshold F`         | 0.85                | Similarity threshold for stateless pass           |
| `--context-threshold F` | 0.58                | Similarity threshold for context pass             |
| `--context-window N`    | 4                   | Turns per session window                          |
| `--use-sulci`           | off                 | Use real MiniLM embeddings (vs TF-IDF simulation) |
| `--out DIR`             | `benchmark/results` | Output directory for result files                 |
| `--seed N`              | 42                  | Corpus RNG seed. `--seed 1 2 3 42` is what every published figure uses. |
| `--agent`               | off                 | Agent-workload pass: 50 sessions x 200 dispatches |
| `--fresh`               | off                 | Delete existing benchmark DBs first. Without it the cache is warm from the previous run. |
| `--allow-overwrite`     | off                 | Replace a results directory that holds a run at a different calibration (e.g. other `--queries`). |

---

## Step 8 — Smoke Tests (Quick End-to-End Sanity Check)

Smoke test scripts live at the repo root. Run individually or together via
`make smoke` to confirm the full stack is working end-to-end.

Measured 2026-09-25 on an M2 with `HF_HUB_OFFLINE=1`: `make smoke` runs all
four scripts in **~33 s** and exits 0. **Read the output, not just the exit
code** — the LangChain and LlamaIndex scripts also exit 0 when they *skip*
because a package is missing. A full run prints four section headers (Core,
LangChain, LlamaIndex, AsyncCache), no `✗`, and no "skip" lines:
`grep -niE 'skip|✗' <log>` should print nothing.

```bash
# All smoke tests in sequence (recommended)
make smoke

# Or individually
python smoke_test.py               # core — no API key needed
python smoke_test_langchain.py     # LangChain integration — no API key needed
python smoke_test_llamaindex.py    # LlamaIndex integration — no API key needed
python smoke_test_async.py         # AsyncCache — no API key needed
```

`smoke_test.py` covers: stateless cache, semantic hit, stats, and context-aware mode.

`smoke_test_langchain.py` covers: `SulciCache` lookup/update/miss/stats via
`langchain_core.globals`. Skips gracefully (exit 0) if `langchain-core` is not installed.

`smoke_test_llamaindex.py` covers: `SulciCacheLLM` wrapping a mock LLM, complete/chat
hit/miss, streaming pass-through, and stats. Skips gracefully if `llama-index-core`
is not installed.

`smoke_test_async.py` covers: `AsyncCache` construction, `aset`, `aget` hit/miss,
`acached_call`, context methods, `astats`, `aclear`, and sync passthrough. No API key
required — runs entirely offline.

### Make targets

```bash
make smoke              # all smoke tests in sequence
make smoke-core         # core smoke test only (smoke_test.py)
make smoke-langchain    # LangChain smoke test only (smoke_test_langchain.py)
make smoke-llamaindex   # LlamaIndex smoke test only (smoke_test_llamaindex.py)
make smoke-async        # AsyncCache smoke test only (smoke_test_async.py)
make smoke-fast         # all smoke tests with SENTENCE_TRANSFORMERS_DEVICE=cpu
                        # (Apple Silicon: sidesteps MPS if `make smoke` stalls)
```

---

## Step 8.5 — Pre-PR Verification with Runner Scripts

When you're about to open a PR, the `scripts/` directory provides three
runners that wrap the most common pre-commit verification flows. Each one
runs sequentially across many files in fresh subprocesses, captures
pass/fail per file, and prints a summary table at the end. Failure logs
are saved to `/tmp/sulci-*-runner/` for later inspection.

### Why these scripts exist

Several integration tests construct multiple `MiniLMEmbedder` instances
in one process. On Apple Silicon (MPS), this occasionally deadlocks at
`embeddings.cpu()` under memory pressure. The runner scripts launch each
test file (or example) in its own Python subprocess, giving each a clean
MiniLM cold-start and avoiding the deadlock. Trade-off: each subprocess
pays its own ~30-40s warmup, so wall-clock is longer than a single
`pytest tests/` invocation. CI doesn't need this — it runs on Linux
without MPS — but local development on M-series Macs benefits.

### Make targets

```bash
make test-per-file              # all test files in fresh subprocesses (~10-15 min)
make test-per-file-fast         # skip the slowest 4 files (~3-5 min, faster iteration)
make examples                   # all examples + smoke tests with timeout (~10-15 min)
make verify-integration-examples  # full 4-scenario LLM-provider matrix for langchain
                                  # + llamaindex (~10-15 min, requires both API keys,
                                  # ~$0.10-0.20 in real LLM calls per run)
make benchmark-verify           # run TF-IDF benchmark, verify against baseline.json (~19 s on an M2)
make checkin                    # pre-PR check: smoke + test-per-file + examples + benchmark-verify
                                #   + check-ci-coverage + check-agent-draws
make checkin-fast               # same, but smoke-fast (CPU) — the Makefile's recommendation on macOS
                                #   (rationale: docs/architecture/adrs/0002-smoke-fast-cpu-mode.md)
```

**`make checkin` needs two more packages than Step 3 installs.** The examples
runner includes `examples/agent_example_langgraph.py` and
`examples/agent_example_crewai.py`, which need frameworks that are in no sulci
extra. Without them, each exits 1 in 0.1 s and the check-in fails
(`TOTAL: 14/16 passed`, make exit 2). Install them once — **both lines, in
this order**:

```bash
pip install langgraph langchain-anthropic crewai
pip install "mcp>=2.0.0"
```

The second line is not optional. `crewai` pins `mcp~=1.28.1` and silently
downgrades the `mcp` 2.x that sulci's `mcp` extra requires, which brings back
the `tests/test_integrations_mcp.py` collection error from Step 5. `pip check`
does **not** catch that (it ignores extras). Re-installing `mcp>=2.0.0` makes
pip print `crewai 1.15.22 requires mcp~=1.28.1 … incompatible`, and `pip check`
repeats it — **both are expected**: the crewai example only needs crewai's core
and runs fine against mcp 2.x (verified 2026-09-27: both agent examples exit 0,
and `test_integrations_mcp.py` passes 22/22). Confirm with:

```bash
python -m pytest tests/test_integrations_mcp.py -q
```

**On an M-series Mac, use `make checkin-fast`.** Before either target, start a
local Redis if you want the Redis-backed paths covered — see the
*Redis-dependent tests* note at the end of this guide. Without one, those
tests skip and the run still passes. Set `HF_HUB_OFFLINE=1` (Step 5) to keep
the runtime independent of your connection.

### When to use which

| If you changed... | Run |
|---|---|
| `sulci/` source code | `make test-per-file` |
| `examples/*.py` or `smoke_test*.py` | `make examples` |
| `examples/langchain_example.py` or `examples/llamaindex_example.py` | `make verify-integration-examples` |
| `benchmark/` files or anything that touches headline numbers | `make benchmark-verify` |
| Anything before opening a PR | `make checkin` (`make checkin-fast` on macOS) |

### Direct script invocation

The runners support direct invocation if you want to override defaults
like timeout or filter to specific files:

```bash
python scripts/run_tests_per_file.py --help
python scripts/run_examples.py --help
python scripts/verify_integration_examples.py --help
```

For example, to run only the four fast test files with a tight
60-second timeout:

```bash
python scripts/run_tests_per_file.py \
    --files tests/test_backends.py tests/test_cloud_backend.py \
            tests/test_connect.py tests/compat/ \
    --timeout 60
```

### What `make checkin` produces

A successful run prints a per-file test summary, then the examples summary,
then a final `✓ checkin verification complete` banner (`checkin-fast` prints
`✓ checkin-fast verification complete (CPU smoke mode)`). Measured 2026-10-01
(v0.9.2, M2, `make checkin-fast`, `HF_HUB_OFFLINE=1`, full Step 3 install plus
the agent packages and mcp re-pin above, no Redis), make exit 0, about 5 minutes
end to end:

```
TOTAL: 727 passed, 0 failed, 0 errors, 55 skipped     (per-file tests)
TOTAL: 16/16 passed                                    (examples)
```

The test counts differ slightly from Step 5 because crewai pulls in `chromadb`,
which un-skips the Chroma backend tests. If anything fails,
the failure log path is printed in the per-file summary table so you
can `cat` the relevant log rather than re-running with extra flags.

### Adding new runner scripts

If you add a new dev-tooling script:

1. Make it a `#!/usr/bin/env python3` script in `scripts/` with `chmod +x`
2. Use `argparse` with a `--help` that explains preconditions and exit codes
3. Use `subprocess.run(timeout=...)` rather than the GNU `timeout` command
   (macOS doesn't ship `timeout` by default; `subprocess.run` is portable)
4. Save per-target logs to `/tmp/<runner-name>/`
5. Add a Makefile target so contributors don't have to remember the
   invocation
6. Update `scripts/README.md` and this section

---

## Step 9 — Test sulci.connect() Locally

`sulci.connect()` is the opt-in telemetry gate. The default state
is **silent** — nothing is sent until you explicitly call `connect()`.

**You don't need a Sulci account or API key to develop or test sulci-oss.**
Every test of the connected-OSS surface (`connect()`, telemetry, the device-code
flow, `SulciCloudBackend`) runs against mocks: `tests/conftest.py` strips
`SULCI_API_KEY` from the environment, each test patches the network and the
`~/.sulci/config` lookup, and CI injects no Sulci key. The keyless snippets
below are safe the same way.

> **Known gap (2026-10-01):** a background telemetry flush thread started by one
> test keeps running for the rest of the session, and can fire during a later
> test that has telemetry enabled but hasn't yet patched `httpx.post`. That
> sends a real POST to `https://api.sulci.io/v1/telemetry` carrying a fake test
> key. In guarded full-suite runs it appeared once, with the flush interval
> forced down to 50 ms; at the normal 30 s interval the window is tiny, but not
> zero. Until the suite blocks real HTTP and stops the thread centrally, the only
> local guarantee is running the tests with no network connection (telemetry
> failures are swallowed, so results are unaffected). Don't set `SULCI_GATEWAY`
> for the test run as a workaround: two tests assert the production default URL
> and fail.

The **live** checks in this step (`connect()` with a real key, `prompt=True`, the
staging smoke) are optional. They're for account holders who have minted a key,
by signing up or through the device-code flow, and want to exercise the engine
against a real gateway.

The repo has exactly **one** automated live check: PROBE 2 in
`tests/integration/flows/flow_2_routemismatch.py`. `test_flow_contracts.py` runs
that script as part of a normal `pytest tests/`, and PROBE 2 activates only when
both `SULCI_LIVE_GATEWAY` and `SULCI_LIVE_KEY` are set. It has no default URL: it
calls whatever you set. If you enable it, use staging and a staging key, and set
the variables for that one run only, not in your shell profile:

```bash
SULCI_LIVE_GATEWAY=https://staging.api.sulci.io SULCI_LIVE_KEY=sk-sulci-<staging-key> \
  python tests/integration/flows/flow_2_routemismatch.py
```

(The six flow scripts run as subprocesses; they were checked separately on
2026-10-01 and make no network calls unless PROBE 2 is enabled.)

> ⚠️ **Never test against production.** `https://api.sulci.io` is the
> production gateway, and it is the SDK's default. Before running anything in
> Steps 9–10 that calls `connect()` with a key, uses `prompt=True`, or uses
> `backend="sulci"` for real, point the SDK elsewhere:
>
> ```bash
> export SULCI_GATEWAY=https://staging.api.sulci.io   # shared staging
> # or
> export SULCI_GATEWAY=http://localhost:8000          # local docker gateway (sulci-platform)
> ```
>
> One variable redirects everything: telemetry, the device-code flow, and
> `SulciCloudBackend`. It is read **when `sulci` is imported**, so set it first,
> in the shell, not inside an already-running Python session. Use a key issued by
> that gateway, never a production key. To confirm where traffic will go:
> `python -c "import sulci; print(sulci._TELEMETRY_URL)"`.
>
> The default-state, nudge and gateway-override checks below, all of Step 10's
> snippets, and every `pytest` command make **no** network calls, so they are
> safe without it.

### Verify default state

```python
import sulci

# Before connect() — everything is off
print(sulci._telemetry_enabled)    # False
print(sulci._api_key)              # None
print(sulci._event_buffer)         # []
```

### Test connect() with a real key

Run these with `SULCI_GATEWAY` set to staging or a local gateway (see the
warning above), using a key from that gateway. A successful `connect()` writes
the key to `~/.sulci/config`.

```python
import sulci

# Option 1 — explicit key
sulci.connect(api_key="sk-sulci-...")
print(sulci._telemetry_enabled)    # True
print(sulci._api_key)              # sk-sulci-...

# Option 2 — from environment variable
# export SULCI_API_KEY=sk-sulci-...
sulci.connect()
print(sulci._api_key)              # sk-sulci-...

# Option 3 — connect but disable telemetry reporting
sulci.connect(api_key="sk-sulci-...", telemetry=False)
print(sulci._telemetry_enabled)    # False (key stored, no reporting)
```

### Disable telemetry per Cache instance

```python
# Even after connect(), an individual Cache can opt out
cache = sulci.Cache(backend="sqlite", telemetry=False)
print(cache._telemetry)            # False
```

### v0.5.2 — what `connect()` does at the wire level

When you call `sulci.connect(api_key="sk-sulci-...")` for the first time,
Sulci writes a small file at `~/.sulci/config` (mode 0600) containing a
freshly-generated `machine_id` (uuid4):

```bash
$ ls -la ~/.sulci/
drwx------  2 you  staff  64 May  3 14:23 .
-rw-------  1 you  staff  60 May  3 14:23 config

$ cat ~/.sulci/config
{
  "machine_id": "a1b2c3d4e5f6..."
}
```

This `machine_id` is anonymous — it's a fresh UUID, never derived from
hostname, MAC address, or filesystem path. It's used as one input to the
**deployment fingerprint** sent on each telemetry POST:

```python
fingerprint = blake2b(
    machine_id || backend || embedding_model || threshold || context_window,
    digest_size=12,
).hexdigest()   # 24 hex chars
```

Switching backend or embedding_model produces a new fingerprint, which
the dashboard's `/v1/analytics/deployments` view treats as a new
deployment. Same machine + same config → stable fingerprint across
restarts.

### v0.5.2 — passive nudge in `cache.stats()`

After 100 cached queries on a `Cache` instance that hasn't been
connected (`sulci.connect()` not called), `cache.stats()` will emit a
single one-line nudge to stderr suggesting `sulci.connect()`. One-shot
per process. Three ways to silence it:

```bash
# Globally, in your shell
export SULCI_QUIET=1

# Or: just call connect() — already-connected silences the nudge
python -c "import sulci; sulci.connect(api_key='sk-sulci-...')"

# Or per Cache instance — telemetry=False also disables the nudge path
cache = sulci.Cache(backend="sqlite", telemetry=False)
```

Verify the nudge fires (and only once):

```python
import sulci
import sulci.core as core
core._NUDGE_SHOWN = False           # reset for this demo

cache = sulci.Cache(backend="sqlite")
cache._query_count = 100             # simulate 100 queries
cache.stats()                        # → prints nudge to stderr
cache.stats()                        # → silent (one-shot)
```

### v0.5.3 — OSS-Connect device-code flow (latent)

In v0.5.3 `sulci.connect()` gains a `prompt: bool = False` parameter.
When set to `True`, and no api_key is found through the
arg/env/`~/.sulci/config` resolution chain, the SDK runs the RFC 8628
device-code flow against the gateway:

```python
import sulci

# v0.5.3 default — completely safe everywhere:
sulci.connect()
# → falls through args → env → config; if none yield a key, returns silently
#   (no telemetry enabled, no network call attempted)

# To opt into the browser-based onboarding flow (SULCI_GATEWAY set to staging/local!):
sulci.connect(prompt=True)
# → if no key found through the first three steps:
#     [sulci] Visit https://dashboard.sulci.io/oss-connect and enter code: WXYZ-2345
#     (production shown; the gateway you target decides the real URL)
#     [sulci] Waiting for authorization (Ctrl+C to cancel)...
#   On success: SDK gets the api_key, writes to ~/.sulci/config (mode 0600)
#   On user-deny / 15-min timeout: raises RuntimeError
```

> **The device-code chain is live in production** (gateway
> `/v1/oss-connect/{device-code,authorize,token}` plus the dashboard
> `/oss-connect` page, since the 2026-05-08 cutover), so end users can run
> `prompt=True` safely. **Developers still test it against staging or a local
> gateway, never production** (see the warning at the top of this step).
>
> **Check that your target gateway has deployed the OSS-Connect routes first.**
> Where it hasn't, `prompt=True` on a missing key either 404s immediately or
> blocks for 15 minutes waiting for an authorization that cannot happen.
>
> **The `prompt` default stays `False` permanently.** v0.6.0 was once going
> to flip it; that was decided against on 2026-07-06 — see the reasoning in
> `sulci.connect`'s docstring. Do not expect the flip, and do not re-file it.

For local dev against a docker-compose gateway, override the gateway URL:

```bash
export SULCI_GATEWAY=http://localhost:8000
python -c "import sulci; sulci.connect(prompt=True)"
```

> **Note for v0.5.0 - v0.5.4:** this env var only redirects the device-code
> flow shown above; telemetry POSTs from `connect()` + `Cache.get()` stay
> pinned to `api.sulci.io` regardless. v0.5.5 fixed that — see the next
> sub-section.

### v0.5.5 — staging-gateway redirect for telemetry

In v0.5.5 a single `SULCI_GATEWAY` value redirects both the device-code flow
*and* the telemetry pipeline. This is what makes a published `pip install
sulci` wheel usable against a non-prod gateway without code changes — e.g.
pointing it at the Railway staging gateway before DNS cutover.

**Quick verification** that the env var is reaching `_TELEMETRY_URL`:

```bash
SULCI_GATEWAY=https://staging.example.com python -c "
import sulci
print('_GATEWAY_BASE: ', sulci._GATEWAY_BASE)
print('_TELEMETRY_URL:', sulci._TELEMETRY_URL)
"
# v0.5.5 prints:
#   _GATEWAY_BASE:  https://staging.example.com
#   _TELEMETRY_URL: https://staging.example.com/v1/telemetry
#
# v0.5.4 and earlier print _TELEMETRY_URL still pointing at api.sulci.io
# regardless of the env var — that's the bug 0.5.5 fixed.
```

**End-to-end staging smoke** (Connected-OSS dashboard tier, mirrors
sulci-platform LAUNCH-PLAN row C2e):

```bash
# In a clean venv so dev imports don't shadow the published wheel
python -m venv ~/c2e_venv && source ~/c2e_venv/bin/activate
pip install "sulci>=0.5.5"   # 0.5.6+ also works; pin if you specifically need 0.5.5 behavior

export SULCI_GATEWAY=https://staging.api.sulci.io     # never api.sulci.io
export SULCI_API_KEY=sk-sulci-<staging-oss-connect-test-key>   # plan='oss_connect'

python - <<'PY'
import sulci, time
print("telemetry destination:", sulci._TELEMETRY_URL)

from sulci import Cache
sulci.connect()                       # picks up SULCI_API_KEY from env
c = Cache(backend="sqlite")
c.set("How do I deploy to AWS?", "Use the AWS CLI...")
resp, sim, ctx = c.get("How do I deploy to AWS?")
print(f"hit: sim={sim:.3f}")

# Background flush thread runs every 30s. One full cycle is enough
# for both the startup event (from connect) and the cache.get event
# to land on the gateway.
time.sleep(35)
PY

# Verify on the gateway side — fingerprint should appear
curl -H "X-Sulci-Key: $SULCI_API_KEY" \
  "$SULCI_GATEWAY/v1/analytics/deployments" | python -m json.tool
```

The fingerprint that lands here is what powers the `ConnectedOssOverview`
"Active SDKs" stat card and the `DeploymentsTable` row on the customer
dashboard, in the dashboard deployment that matches the gateway you targeted.

### Run only the gateway-override tests

```bash
python -m pytest tests/test_telemetry_gateway_override.py -v
# 6 tests — default URL, env override, trailing-slash normalization,
# localhost-for-local-dev, _post() honoring resolved URL (with + without env)
```

### Key resolution order (v0.5.3)

When `backend="sulci"` is used, the API key is resolved in this order
(first match wins):

```
1. Explicit api_key= argument to Cache() or sulci.connect()
2. SULCI_API_KEY environment variable
3. ~/.sulci/config (persisted from a prior successful sulci.connect() call)
4. Browser-based OSS-Connect device-code flow — only if prompt=True
```

Step 3 is new in v0.5.3 — your first successful
`sulci.connect(api_key="sk-sulci-...")` persists the key, and subsequent
`sulci.connect()` calls with no arguments pick it up automatically.

Step 4 is the new device-code flow described above.

### v0.6.5 — resolution-path observability + config staleness guard

In v0.6.5 the four-rung key resolution chain became debuggable instead of
silent, and stale `~/.sulci/config` entries are now rejected at read time
rather than 401'ing later against the gateway. Two changes, both opt-in /
upgrade-path-aware so nothing breaks for callers who don't engage them.

**1. INFO-level log line on every `sulci.connect()`** identifying which
rung supplied the key. Default Python logging level is WARNING, so these
lines stay quiet unless explicitly enabled:

```python
import logging
logging.getLogger("sulci").setLevel(logging.INFO)

import sulci
sulci.connect()
# Typical output (one of four shapes, depending on which rung won):
#   INFO sulci: using explicit api_key argument (prefix=sk-sulci-abcd)
#   INFO sulci: using SULCI_API_KEY env var (prefix=sk-sulci-efgh)
#   INFO sulci: using persisted ~/.sulci/config (prefix=sk-sulci-ijkl,
#               mtime=2026-02-15T14:23:00+00:00)
#   INFO sulci: using device-code flow result (prefix=sk-sulci-mnop)
```

The config-rung line includes the file's mtime — the single most useful
diagnostic for "I thought I refreshed but telemetry is still hitting the
old account." The "no key resolved" case emits a DEBUG line (not INFO);
fine behavior when `prompt=False`, not worth INFO attention.

**2. `~/.sulci/config` staleness guard.** `sulci/config.py::update()` now
auto-stamps `written_at` (UTC ISO-8601) on any write that touches
`api_key`. `_read_key_from_config()` refuses to use entries that are:

- missing `written_at` (config predates v0.6.5; age can't be verified → treated as stale)
- older than 90 days (`_CONFIG_MAX_AGE_DAYS`)
- have an unparseable `written_at` field

All three reject paths return `None` and emit a `WARNING` with remediation
text pointing at `api_key=` explicit pass or `sulci.connect(prompt=True)`.

**Upgrade-path consequence.** Existing `~/.sulci/config` files from v0.6.4
and earlier have no `written_at` field. On the first `sulci.connect()`
call after upgrading to v0.6.5, those configs are skipped with a WARNING
and the resolution chain falls through. Resolution paths to clear the
warning:

```bash
# Path A — pass api_key explicitly once, then sulci.connect() rewrites
# the config with a written_at stamp on success:
python -c "import sulci; sulci.connect(api_key='sk-sulci-...')"

# Path B — re-run the device-code flow (writes a fresh config):
python -c "import sulci; sulci.connect(prompt=True)"
```

After either path the config has a fresh `written_at` and subsequent
`sulci.connect()` calls pick it up silently as before.

### Run only the resolution-path + staleness-guard tests

```bash
python -m pytest tests/ -v -k "ResolutionPathLogging or ConfigAgeOut or WrittenAtStamping"
# 15 tests added in v0.6.5 across TestResolutionPathLogging,
# TestConfigAgeOut, TestWrittenAtStamping (in test_connect.py + test_config.py)
```

### Run only the connect tests

```bash
python -m pytest tests/test_connect.py -v
# Expected: 59 passed (verified 2026-10-01; mocked, no network)

# Run a specific class
python -m pytest tests/test_connect.py::TestDefaultState -v
python -m pytest tests/test_connect.py::TestConnect -v
python -m pytest tests/test_connect.py::TestEmit -v
python -m pytest tests/test_connect.py::TestFlush -v
python -m pytest tests/test_connect.py::TestCacheIntegration -v
python -m pytest tests/test_connect.py::TestThreadSafety -v

# v0.5.5 — gateway-URL override coverage (separate file because it
# requires sys.modules purging + reimport to re-evaluate module-level
# constants; mixing this fixture pattern into test_connect.py would
# couple unrelated tests).
python -m pytest tests/test_telemetry_gateway_override.py -v
```

---

## Step 10 — Test SulciCloudBackend Locally

`SulciCloudBackend` is the cloud backend driver. It routes cache operations
over httpx to, in order: the `gateway_url=` argument, else `SULCI_GATEWAY`,
else the production default `https://api.sulci.io`. **For any live test, set
`SULCI_GATEWAY` to staging or a local gateway first** (see Step 9). The
snippets below are safe as they stand: constructing the backend opens no
connection, and the wiring checks use a mock.

### Verify the import and basic construction

```python
from sulci.backends.cloud import SulciCloudBackend

# Confirm ValueError on missing key
try:
    b = SulciCloudBackend(api_key=None)
except ValueError as e:
    print(f"ValueError ok: {e}")

# Confirm repr
b = SulciCloudBackend(api_key="sk-sulci-testkey1234567")
print(b)
# SulciCloudBackend(url='https://api.sulci.io', key_prefix='sk-sulci-testkey', timeout=5.0)
# (url shows whatever SULCI_GATEWAY resolves to; nothing is sent)
```

### Verify Cache constructor wiring

```python
from unittest.mock import patch
from sulci import Cache

# Explicit key
with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
    MockBackend.return_value = MockBackend
    cache = Cache(backend="sulci", api_key="sk-sulci-testkey1234567")
    print(f"Cache with sulci backend: {cache}")

# Via env var
import os
os.environ["SULCI_API_KEY"] = "sk-sulci-testkey1234567"
with patch("sulci.backends.cloud.SulciCloudBackend") as MockBackend:
    MockBackend.return_value = MockBackend
    cache = Cache(backend="sulci")
    print("Env var resolution ok")
del os.environ["SULCI_API_KEY"]
```

### Run only the cloud backend tests

```bash
python -m pytest tests/test_cloud_backend.py -v
# Expected: 55 passed (verified 2026-10-01; all mocked, no network)

# Run a specific class
python -m pytest tests/test_cloud_backend.py::TestConstruction -v
python -m pytest tests/test_cloud_backend.py::TestRemoteGet -v
python -m pytest tests/test_cloud_backend.py::TestRemoteSet -v
python -m pytest tests/test_cloud_backend.py::TestDeleteAndClear -v
python -m pytest tests/test_cloud_backend.py::TestCacheWiring -v
python -m pytest tests/test_cloud_backend.py::TestCanonicalGatewayPaths -v
python -m pytest tests/test_cloud_backend.py::TestCloudTransportShortCircuit -v
```

---

## Step 11 — Test LangChain Integration Locally

`SulciCache(BaseCache)` is the LangChain cache adapter added in v0.3.3.

### Verify the import

```bash
python -c "from sulci.integrations.langchain import SulciCache; print('✅ Import OK')"
```

### Run the integration tests

```bash
python -m pytest tests/test_integrations_langchain.py -v
# Expected: 27 passed  (verified 2026-09-25, ~18 s with HF_HUB_OFFLINE=1)
```

### Run the LangChain smoke test

```bash
python smoke_test_langchain.py
# or: make smoke-langchain
```

---

## Step 12 — Test LlamaIndex Integration Locally

`SulciCacheLLM(LLM)` is the native LlamaIndex LLM wrapper added in v0.3.6.

### Verify the import

```bash
python -c "from sulci.integrations.llamaindex import SulciCacheLLM; print('✅ Import OK')"
```

### Run the integration tests

```bash
python -m pytest tests/test_integrations_llamaindex.py -v
# Expected: 29 passed  (verified 2026-09-25, ~12 s with HF_HUB_OFFLINE=1)
```

### Run the LlamaIndex smoke test

```bash
python smoke_test_llamaindex.py
# or: make smoke-llamaindex
```

---

## Step 13 — Test AsyncCache Locally

`AsyncCache` is the non-blocking async wrapper added in v0.3.7.

### Verify the import

```bash
python -c "from sulci import AsyncCache; print('✅ Import OK')"
```

### Run the integration tests

```bash
python -m pytest tests/test_async_cache.py -v
# Expected: 40 passed  (verified 2026-09-25, ~14 s with HF_HUB_OFFLINE=1)
```

### Run the AsyncCache smoke test

```bash
python smoke_test_async.py
# or: make smoke-async
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `.envrc:1: .venv/bin/activate: No such file or directory` on `cd` | direnv runs the repo's `.envrc` before Step 2 created the venv | Expected — do Step 2, then `direnv allow` |
| `command not found: pip` / `python` | venv not active (new terminal tab, or outside the repo) | `cd` into the repo (direnv activates it) or `source .venv/bin/activate` |
| `zsh: no matches found: .[sqlite]` | zsh glob expansion | Quote extras: `".[sqlite]"` |
| Pasted command gets extra arguments (`accepts at most 1 arg(s), received 9`) | zsh passing trailing `# comments` as arguments | `echo 'setopt interactivecomments' >> ~/.zshrc`, new tab |
| `pytest: command not found` | pytest not on `PATH` | `python -m pytest` |
| `ModuleNotFoundError: sulci` | Not installed | Run the Step 3 install |
| `ModuleNotFoundError` for `chromadb` / `langchain_core` / `llama_index` / other backend | Extra not installed | Re-run the Step 3 install, or `pip install -e ".[<extra>]"` |
| `Interrupted: 2 errors during collection` — `ImportError: mcp>=2.0.0 is required` (or litellm) | `mcp` / `litellm` extra missing, **or** `mcp` downgraded to 1.x by `crewai` | Re-run the Step 3 install, then `pip install "mcp>=2.0.0"`. `pip check` will not detect this |
| 3 `TestRedisStreamSink` tests fail: `redis package not installed` | `redis` extra missing (the tests mock the server but still import the package) | Step 3 install includes `redis` |
| `TestRedisBackend::test_contract_local` takes ~5 s instead of skipping instantly | `localhost` resolves to a non-loopback address on your machine, so the connection times out rather than being refused | Check `grep localhost /etc/hosts`; it should map to `127.0.0.1` / `::1` only |
| Test run looks hung — one test for minutes at ~0% CPU | Each embedding-model load checks Hugging Face; slow network | `export HF_HUB_OFFLINE=1` once the model is cached (Step 5) |
| `VARIANT COLLISION … --queries: on disk '1000'`, `verify_benchmark.py` exit 2 | A non-default benchmark run wrote to `benchmark/results/tfidf/` | `rm -rf benchmark/results/tfidf` (untracked); give exploratory runs their own `--out` (Step 7) |
| `make checkin`: `agent_example_langgraph.py` / `agent_example_crewai.py` `FAIL exit=1 0.1s` | Agent frameworks not installed | Step 8.5: install them, then re-pin `mcp>=2.0.0` |
| `ValueError: not enough values to unpack` | v0.1 unpacking style | `cache.get()` returns a **3-tuple** — `response, sim, ctx_depth = cache.get(...)` |
| MiniLM takes 2–3 s on first call | Model cold load | Normal. Warm the model at app startup, not per request |
| `RuntimeWarning: sulci: upgraded the SQLite cache schema … discarded N entries written before 0.9.2` | First open of a pre-0.9.2 SQLite cache under 0.9.2. The schema is migrated in place and every old row is dropped, because its scope can't be determined | Expected, once per database. The cache starts cold and refills on misses |
| `sqlite3.OperationalError: ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint` | A 0.9.1 (or older) checkout writing to a SQLite cache that 0.9.2 has already migrated. Downgrading a migrated database isn't supported | Point the older version at a fresh `db_path`, or delete that cache directory |
| `git push` returns 403 | GitHub credentials missing or expired | `gh auth login`, then `gh auth setup-git`. Don't put a token in the remote URL — it is stored in plain text in `.git/config` |
| `_telemetry_enabled` is True unexpectedly | `connect()` called elsewhere | Check app code and test fixtures for `sulci.connect()` — telemetry is opt-in only |

---

## API Key Notes

The core library and all tests run **without any API key**. The only things that
require a key:

| File                                                | Key needed                                                        |
| --------------------------------------------------- | ----------------------------------------------------------------- |
| `examples/anthropic_example.py`                     | `ANTHROPIC_API_KEY`                                               |
| `examples/langchain_example.py`                     | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (optional — mock fallback)|
| `examples/llamaindex_example.py`                    | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (optional — mock fallback)|
| `examples/async_example.py`                         | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (optional — mock fallback)|
| `examples/agent_example_langgraph.py`, `agent_example_crewai.py` | `ANTHROPIC_API_KEY` (optional — mock fallback)       |
| `examples/litellm_example.py`                       | `OPENAI_API_KEY` (optional — mock fallback)                       |
| `examples/proxy_example.py`                         | `OPENAI_API_KEY` **required**, plus a running `sulci-proxy` on :8787 — so it is not in `make examples` |
| `sulci/embeddings/openai.py`                        | `OPENAI_API_KEY`                                                  |
| `sulci.connect()` / `Cache(backend="sulci")`        | `SULCI_API_KEY` (Sulci Cloud — optional)                          |
| All other code                                      | None                                                              |

The default embedding model (`minilm`) runs locally via `sentence-transformers`.
Sulci itself makes no network calls unless you configure `embedding_model="openai"`
or use `backend="sulci"` / `sulci.connect()`. **The model loader does:** the first
load downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face, and every later load
checks it for updates unless `HF_HUB_OFFLINE=1` is set (see Step 5).

> **`SULCI_API_KEY`** is the environment variable for the Sulci Cloud managed backend.
> Get a free key at [sulci.io/signup](https://sulci.io/signup). Setting this variable
> is optional — the library works fully offline without it. That is a
> **production** key: for development and testing, use a key issued by the
> staging or local gateway, with `SULCI_GATEWAY` set (Step 9).

---

## What a Clean Run Looks Like

Tail of a clean run (2026-10-01, v0.9.2, M2, Step 3 install plus the Step 8.5
agent packages, offline, no Redis or Qdrant server):

```
$ HF_HUB_OFFLINE=1 python -m pytest tests/ -v -rs
...
tests/test_telemetry_lifecycle.py::TestAtexitFlush::test_flush_thread_is_daemon PASSED
...
=========== 702 passed, 37 skipped, 5 warnings in 83.12s ===========
```

(Step 3 install alone: expect `697 passed, 42 skipped` — 5 Chroma tests skip
without `chromadb`.)

- **The 5 warnings are expected.** They are `UserWarning`s from tests that
  deliberately leave `context_threshold` unset (see `docs/context-threshold.md`).
- **Skips are expected** for backends whose packages you didn't install, tests
  that need a live server, and four permanent by-design skips. Step 5 breaks
  them down; `-rs` lists the reason for each.
- **Failures and collection errors are not expected.** Not every missing extra
  produces a skip — some fail or error instead (Step 3, Troubleshooting).

---

## Project Structure (Reference)

Generated from the repo on 2026-09-30 (v0.9.1); unchanged in 0.9.2. Per-file test counts are
deliberately left out — they go stale with every release. For the current total,
run `python -m pytest tests/ --collect-only -q | tail -1` (739 at v0.9.2).

```
.
├── .envrc                      ← direnv: activates .venv (tracked in git)
├── .github/workflows/          ← CI: tests.yml (3 OS × Python 3.9–3.12), publish.yml, benchmark.yml
├── CHANGELOG.md  CONTRIBUTING.md  LICENSE  NOTICE  README.md  SECURITY.md
├── LOCAL_SETUP.md              ← this guide
├── Makefile                    ← smoke, test-*, checkin / checkin-fast, benchmark-verify, …
├── pyproject.toml              ← name="sulci", version, extras, console scripts
├── setup.py
├── setup.sh                    ← quick start only: installs a SUBSET of the Step 3 extras
│                                 and uses plain `python3` — follow Steps 2–3 for development
├── smoke_test.py  smoke_test_async.py  smoke_test_langchain.py  smoke_test_llamaindex.py
├── benchmark/
│   ├── README.md               ← methodology and results
│   ├── baseline.json           ← pinned TF-IDF regression baseline
│   ├── run.py                  ← benchmark CLI (Step 7)
│   └── results/                ← tfidf/ (scratch) and minilm/ (committed published draws)
├── docs/
│   ├── API-SURFACE.md          ← public API, checked by scripts/check_api_surface.py
│   ├── context-threshold.md    ← why context_threshold has no default
│   ├── protocols.md  multi_tenancy_and_isolation.md  OSS_BOUNDARY_POLICY.md
│   └── architecture/           ← ADRs (e.g. 0002 smoke-fast CPU mode)
├── examples/                   ← see Step 6 and API Key Notes
│   ├── basic_usage.py  context_aware.py  context_aware_example.py
│   ├── anthropic_example.py  langchain_example.py  llamaindex_example.py  async_example.py
│   ├── agent_example_langgraph.py  agent_example_crewai.py
│   ├── mcp_example.py  litellm_example.py  proxy_example.py  gh_aw_sulci_mcp.md
│   └── extending_sulci/        ← reference implementations of the protocols
├── scripts/                    ← see scripts/README.md
│   ├── run_tests_per_file.py  run_examples.py  verify_integration_examples.py
│   ├── verify_benchmark.py  check_agent_draws.py  check_api_surface.py
│   └── check_ci_test_coverage.py  check_release_ready.py  check_tag_version.py
├── sulci/
│   ├── __init__.py             ← exports Cache, AsyncCache, ContextWindow, SessionStore, connect()
│   ├── core.py                 ← Cache engine (context-aware)
│   ├── async_cache.py          ← AsyncCache wrapper
│   ├── context.py              ← ContextWindow + in-process session handling
│   ├── config.py               ← ~/.sulci/config persistence
│   ├── telemetry.py  oss_connect.py   ← opt-in telemetry, device-code flow (Step 9)
│   ├── backends/               ← chroma, faiss, milvus, qdrant, redis, sqlite (free)
│   │                             + cloud (backend="sulci", managed); protocol.py
│   ├── embeddings/             ← minilm (default, local), openai; protocol.py
│   ├── integrations/           ← langchain, llamaindex, litellm, mcp_server
│   ├── proxy/                  ← sulci-proxy (OpenAI-compatible caching proxy)
│   ├── sessions/               ← SessionStore: memory, redis; protocol.py
│   ├── sinks/                  ← EventSink: null, telemetry, redis_stream; protocol.py
│   └── tests/compat/           ← SessionStore + EventSink conformance suites
└── tests/                      ← test_*.py per feature, plus compat/ (backend + embedder
                                  conformance) and integration/flows/
```

Console scripts installed by the package: `sulci-mcp` (MCP server) and
`sulci-proxy` (caching proxy).

> **Redis-dependent tests:** the per-file runner exercises `RedisBackend`,
> `RedisSessionStore`, and `RedisStreamSink` against a real Redis daemon. Make sure
> one is running on `localhost:6379` before `make checkin`. Two ways:
>
> ```bash
> # Option A — Docker (no install needed)
> docker run -d --rm -p 6379:6379 --name sulci-test-redis redis:7-alpine
>
> # Option B — Homebrew (macOS)
> brew install redis && brew services start redis
> ```
>
> Without Redis up, the relevant tests skip via the fixture (it probes once,
> with a short timeout), so the suite still completes — but you lose coverage of
> the Redis-backed sessions and sinks paths.

---

## Related Docs

- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — adding a backend, pre-publish review, releasing
- [`CHANGELOG.md`](./CHANGELOG.md) — version history
- [`benchmark/README.md`](./benchmark/README.md) — benchmark methodology and results
- [`docs/API-SURFACE.md`](./docs/API-SURFACE.md) — public API reference
- [`scripts/README.md`](./scripts/README.md) — the verification and runner scripts
- [PyPI: sulci](https://pypi.org/project/sulci/)
- [GitHub: sulci-io/sulci-oss](https://github.com/sulci-io/sulci-oss)

---

## Branch Reference

- **`main`** is the current release line (v0.9.2 as of 2026-10-01 — check
  `pyproject.toml`). All work merges here via PR.
- Work happens on short-lived branches. Prefixes in use: `feat/` or `feature/`,
  `fix/`, `docs/`, `chore/`, `test/`, and `release/vX.Y.Z` for releases.
- Merged branches are not always deleted, so `git branch -r` lists many that are
  merged or stale; don't read anything into them.
- Release steps (version bump, CHANGELOG, merge-commit rule) are in
  [`CONTRIBUTING.md`](./CONTRIBUTING.md#releasing).
