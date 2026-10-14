# scripts/

Developer tooling for sulci-oss. These scripts are not part of the published
package — they exist to help contributors run pre-commit / pre-release
verification more reliably than ad-hoc shell loops.

## What's here

| Script | Purpose | Typical wall-clock | API cost |
|---|---|---|---|
| `run_tests_per_file.py` | Run pytest test files one at a time, in fresh subprocesses | 10-15 min full sweep on M-series Mac | none |
| `run_examples.py` | Run every example + smoke test with timeout, capture pass/fail | 10-15 min full sweep on M-series Mac | none with mock fallback; small $ if API keys are set |
| `verify_integration_examples.py` | Verify langchain & llamaindex examples select the right LLM provider across 4 credential scenarios | 10-15 min for full 8-run matrix | $0.10-0.20 |
| `verify_benchmark.py` | TF-IDF **regression** check — runs `benchmark/run.py --no-sweep --context` without `--use-sulci` and fails if any non-latency metric drifts from `benchmark/baseline.json`. TF-IDF is not the shipped engine; its figures are not publishable (the ones this row used to quote were retired 2026-08-12) | ~19s on an M2 | none |
| `check_agent_draws.py` | The committed MiniLM agent draws (`benchmark/results/minilm/seed-*`) still aggregate to the published register figures (`make check-agent-draws`) | ~0s | none |
| `check_api_surface.py` | `docs/API-SURFACE.md` agrees with the code, by AST (`make check-api`; `--show` prints the surface) | ~1s | none |
| `check_ci_test_coverage.py` | Every test file is run by some CI step (`make check-ci-coverage`) | ~0s | none |
| `check_release_ready.py` | Release pre-flight: version restatements agree, no retired figure outside a retraction, `sulci[all]` installs on the Python floor (`make release-check`) | seconds | none |
| `check_tag_version.py` | The tag about to be pushed matches `pyproject.toml` (`make check-tag TAG=vX.Y.Z`; also CI's release-guard) | ~0s | none |
| `_make_issues.py`, `_make_more_issues.py`, `_make_web_issue.py`, `_update_contributing_v2.py` | One-off generators from the v0.4.0 era, kept for lineage; not part of any check | — | none |

## Why per-file pytest invocations?

The `run_tests_per_file.py` script doesn't run `pytest tests/` as one command —
it runs each file in its own subprocess. This is deliberate: several
integration tests construct multiple `MiniLMEmbedder` instances in one process,
which on Apple Silicon (MPS) occasionally deadlocks at `embeddings.cpu()`
under memory pressure. Running each test file in a fresh Python process gives
each one a clean MiniLM cold-start and avoids the deadlock.

Trade-off: every subprocess pays its own MiniLM cold-load cost (~30-40s on
M-series Macs, much less on CPU-only Linux runners). Wall-clock is longer
than a single pytest invocation. CI doesn't need this — it uses a single
pytest invocation on a clean Linux runner where MPS is irrelevant.

## When to use which

Picking by what you changed:

| If you changed... | Run |
|---|---|
| `sulci/` source code | `make test-per-file` |
| `examples/*.py` or `smoke_test*.py` | `make examples` |
| `examples/langchain_example.py` or `examples/llamaindex_example.py` (and you have API keys) | `make verify-integration-examples` |
| `benchmark/` files or anything that affects benchmark numbers | `make benchmark-verify` |
| Anything before opening a PR | `make checkin` (`make checkin-fast` on macOS) |
| Cutting a release | `make release-check` and `make check-tag TAG=vX.Y.Z` |

## Output and logs

The three runners (`run_tests_per_file.py`, `run_examples.py`,
`verify_integration_examples.py`) save per-target log files to `/tmp/sulci-*-runner/`. On
failure, the summary table prints the path of the failing log so you can
inspect what actually went wrong rather than re-running with `-v`.

Override the log directory with `--log-dir <path>` on any script.

## Adding new scripts

If you add a new runner here:

1. Make it a `#!/usr/bin/env python3` script with `chmod +x`
2. Use `argparse` with a `--help` that explains preconditions and exit codes
3. Use `subprocess.run(timeout=...)` rather than the GNU `timeout` command
   (macOS doesn't ship `timeout` by default; `subprocess.run` is portable)
4. Save per-target logs to `/tmp/<runner-name>/` so failure diagnosis
   doesn't require re-running the whole sweep
5. Add a Makefile target so other contributors don't have to remember
   the exact invocation
6. Update this README with a new row in the "What's here" table
