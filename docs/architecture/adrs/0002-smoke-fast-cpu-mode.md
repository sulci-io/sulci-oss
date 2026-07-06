# ADR 0002 — `make smoke-fast` forces CPU inference on macOS

- **Status:** Accepted
- **Date:** 2026-07-06
- **Shipped:** `sulci-oss` PR #TBD (this ADR + Makefile change)
- **Closes:** sulci-oss #48 (pre-commit smoke ~8 min on macOS)

---

## Context

`make smoke` runs four scripts back-to-back:

1. `smoke_test.py`
2. `smoke_test_langchain.py`
3. `smoke_test_llamaindex.py`
4. `smoke_test_async.py`

Each one instantiates `Cache(backend="sqlite", ...)` at least once, which
loads `SentenceTransformer("all-MiniLM-L6-v2")` via
`sulci/embeddings/minilm.py`. On macOS Apple Silicon (M1/M2/M3), the default
device selection in recent `sentence-transformers` and `torch` releases is
MPS (Metal Performance Shaders). Each load allocates ~200 MB of MPS memory
and runs a JIT warm-up pass on the first embedding call. Four back-to-back
Python processes each pay the full warm-up cost — MPS does not share
compiled kernels across process boundaries — so the wall time on M2/M3
laptops accumulates to roughly **8 minutes** just for `make smoke`, before
`test-per-file` even starts. `make checkin` (which runs `smoke` +
`test-per-file` + `examples` + `benchmark-verify`) becomes uncomfortably
slow as a pre-PR gate.

Linux CI runners and Intel-mac laptops are unaffected — they use
CPU inference by default and each `SentenceTransformer` load completes in
under a second.

## Decision

Add an **opt-in** `make smoke-fast` target (and matching `make checkin-fast`)
that sets the following environment variables before invoking each smoke
script:

```makefile
SMOKE_FAST_ENV = \
	CUDA_VISIBLE_DEVICES="" \
	SENTENCE_TRANSFORMERS_DEVICE=cpu \
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	TOKENIZERS_PARALLELISM=false \
	SULCI_SMOKE_FAST=1
```

- `SENTENCE_TRANSFORMERS_DEVICE=cpu` — primary lever. Bypasses MPS entirely.
- `CUDA_VISIBLE_DEVICES=""` — belt-and-suspenders for anyone with a Linux
  box that has CUDA and wants smoke to stay predictable.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — silences MPS warnings if any dependency
  probes MPS availability before honoring the env var.
- `TOKENIZERS_PARALLELISM=false` — silences the HuggingFace tokenizers fork
  warning that fires under `subprocess`-based test runners.
- `SULCI_SMOKE_FAST=1` — a marker env var for any future in-repo code that
  wants to know smoke mode is active. Currently unused by the library
  itself.

CPU inference on MiniLM-L6-v2 is fast enough for smoke coverage: each smoke
script completes in **under 10 seconds** on the same M2/M3 hardware, for a
total wall time of roughly **30 seconds** vs the prior ~8 minutes.

The default `make smoke` and `make checkin` are **not** modified. The MPS
warm-up cost is a real signal on CI (the tests.yml matrix includes macos
runners), and the default target continues to exercise whatever accelerator
is present. `checkin`'s success message points macOS contributors at
`checkin-fast` explicitly.

### Considered alternatives

- **Prewarm cache with a shared model** — would need a long-lived process
  or IPC coordinator across four independent scripts. Adds surface area
  for a problem the CPU switch solves in one env var.
- **Mock embedder for smoke tests** — smoke tests would no longer verify
  that the real MiniLM path loads and produces vectors. That's the exact
  regression class smoke is meant to catch (see the v0.4.0 dimension-attr
  hasattr guard for MiniLM). Rejected.
- **Change the default** to CPU-forced smoke — silently changes what CI
  exercises on the macos-latest matrix rows. Bad idea for a released
  library. Rejected.
- **Make `SULCI_SMOKE_FAST=1` shadow the env-var set from inside the smoke
  scripts** — pushes complexity into every smoke script (four files) for
  zero user-facing benefit vs setting env vars once in the Makefile.
  Rejected.

## Consequences

- macOS contributors have an explicit fast path: `make smoke-fast` or
  `make checkin-fast`. Documented in the Makefile's own help output.
- The default target's behavior is preserved for CI and for anyone who
  wants the MPS path exercised.
- No SDK code changes. No behavior change for users installing
  `sulci` from PyPI.
- If a future embedding model added to `MODELS` in
  `sulci/embeddings/minilm.py` has a different MPS-compatibility profile,
  this ADR's guidance still holds — CPU is the safe smoke default on
  laptops.
- Any pre-commit hook or contributor tooling that currently calls
  `make smoke` can switch to `make smoke-fast` when running under macOS
  by detecting `uname -s` or `platform.system()`.

## References

- sulci-oss issue #48 — original bug report
- sulci-oss #43 — the earlier libomp workaround (Makefile
  `export KMP_DUPLICATE_LIB_OK := TRUE` at top of file)
- `sulci/embeddings/minilm.py::MiniLMEmbedder`
- HuggingFace `sentence-transformers` device-selection docs
  (`SENTENCE_TRANSFORMERS_DEVICE` supported since 3.0.0)
