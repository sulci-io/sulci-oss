#!/usr/bin/env python3
"""
scripts/_make_issues.py
========================
Generate /tmp/sulci_oss_issues.md with the 7 sulci-oss follow-up issues
captured during v0.4.0 release work. Run once after cloning to regenerate
if needed; not part of the published surface.

The output file uses a delimiter-driven format that can be split into
individual `gh issue create` calls. See the trailing main block for
how to do that.
"""
from pathlib import Path

ISSUES = [
    {
        "title": "tests: cache MiniLMEmbedder at session scope to cut suite wall-clock",
        "labels": ["tests", "performance", "contributor-friendly"],
        "body": """\
## Problem

Running the full test suite via `pytest tests/` takes ~26 minutes on Apple
Silicon (M-series MPS). Most tests construct a fresh `Cache()` instance,
each triggering a fresh `MiniLMEmbedder.__init__` -> `SentenceTransformer.__init__`
-> ~10s MiniLM cold-load to MPS device.

With ~150+ such tests, the cumulative cost dominates the suite runtime.
This is a CI-infrastructure issue, not a correctness issue. CI on Linux
(without MPS) is much faster, but local-dev iteration on M-series Macs is
painful.

## Proposed fix

Add a session-scoped pytest fixture in `tests/conftest.py` that constructs
one `MiniLMEmbedder` per pytest session and shares it across all tests.
Tests inject the shared embedder via dependency injection instead of
constructing their own.

## Estimated impact

- Suite wall-time: ~26 min -> under 2 min on M2 Air
- Memory pressure: ~150 model copies -> 1
- Side benefit: resolves the langchain/llamaindex MPS deadlock issue

## Effort

Touches every test that constructs `Cache()`. ~30-50 test files to audit,
mechanical edit. Best done as a single dedicated PR, not folded into other work.

## Context

Discovered during v0.4.0 release verification. Worked around with
`scripts/run_tests_per_file.py` (each test file gets a fresh subprocess);
the workaround is good enough for CI but the fundamental issue should be fixed.
""",
    },
    {
        "title": "tests: langchain/llamaindex tests deadlock under MPS pressure when run together",
        "labels": ["tests", "bug", "macos", "contributor-friendly"],
        "body": """\
## Problem

When `pytest tests/` is run as a single invocation on Apple Silicon (MPS),
two tests intermittently deadlock at `embeddings.cpu()`:

- `tests/test_integrations_langchain.py::TestNamespacing::test_clear_removes_all_partitions`
- `tests/test_integrations_langchain.py::TestAsync::test_aclear`

Stack trace points at `sentence_transformers/SentenceTransformer.py:1148`
where embeddings are moved back from MPS to CPU.

Root cause: these tests construct multiple `MiniLMEmbedder` instances in
a single Python process via the langchain `_cache_for(llm_string)` namespace
pattern. Multiple BertModels coexisting on MPS hit memory pressure that
manifests as a deadlock when moving tensors back to CPU.

CI (Linux, no MPS) does not reproduce this.

## Workaround in place

`scripts/run_tests_per_file.py` runs each test file in a fresh subprocess.
File-by-file isolation gives each test file its own clean MPS state, and
the deadlock disappears. `make test-per-file` uses this script.

## Real fix

Same root cause as the MiniLMEmbedder cold-start issue (see related issue
on session-scoped fixture). Fixing that by sharing one MiniLMEmbedder
across the session would also resolve this deadlock.

## Priority

Low. Workaround works; fix is contingent on the session-fixture work.

## Context

Surfaced during v0.4.0 release verification.
""",
    },
    {
        "title": "examples: db_path pollution across runs makes demos non-idempotent",
        "labels": ["examples", "dx", "contributor-friendly"],
        "body": """\
## Problem

`examples/anthropic_example.py` (and possibly others) construct
`Cache(backend=\"sqlite\")` with the default db_path. The cache file is not
cleared between runs, so:

1. First run: cache misses correctly, real LLM calls happen, responses cached
2. Second run: cache hits the entries from the first run, so even the
   "fresh question" round shows hits with stale responses

Discovered during v0.4.0 verification: a run with `ANTHROPIC_API_KEY` set
inadvertently picked up `[Mock]` responses from a previous no-key run,
making demo behavior confusing.

## Proposed fix

Three options, in order of impact:

(a) Examples use `tempfile.TemporaryDirectory()` for db_path. Each run is
    a fresh sandbox. Demos are idempotent.
(b) Examples call `cache.clear()` near startup before doing demo work.
    Less invasive, makes intent explicit.
(c) Document the persistent behavior clearly in each example's docstring.
    Lowest effort, but doesn't fix the surprising behavior.

Recommend (a) for example/demo files.

## Scope

7 example files in `examples/` to audit. Plus the new
`examples/extending_sulci/custom_backend.py` (in-memory, unaffected).

## Effort

~10-20 lines per example file. Could be one focused PR.
""",
    },
    {
        "title": "examples: fail fast on rejected API key with informative message",
        "labels": ["examples", "dx", "ergonomics"],
        "body": """\
## Problem

When an LLM-using example is run with an `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` value that the provider rejects (typo'd key, revoked
key, wrong-tier key), the example crashes deep inside the SDK with a
70-line stack trace ending in `AuthenticationError: Error code: 401`.

The traceback is technically informative if you know what to look for, but
unfriendly for the typical case of "I pasted the wrong env var."

## Proposed enhancement

Wrap the first LLM call in each example with a `try/except` that catches
provider auth errors and prints a clean message:

    try:
        response = call_llm(query)
    except (openai.AuthenticationError, anthropic.AuthenticationError) as e:
        print(f"\\nERROR: API key was rejected by the provider.")
        print(f"  Provider message: {e.message}")
        print(f"  Common causes: typo in key, revoked key, wrong tier.")
        sys.exit(1)

## Scope

Affects 4 examples that make real provider calls:
- examples/anthropic_example.py
- examples/langchain_example.py
- examples/llamaindex_example.py
- examples/async_example.py

## Priority

Low. Current behavior is technically defensible (raises a clear exception
from the SDK). Enhancement is ergonomic.

## Context

Discovered during v0.4.0 LLM-provider verification matrix.
""",
    },
    {
        "title": "docs: clarify TF-IDF vs MiniLM engine in published benchmark numbers",
        "labels": ["documentation", "marketing"],
        "body": """\
## Problem

The README's published headline numbers (85.88% hit rate, +20.8pp delta,
$21.47 saved per 5000 queries) come from the built-in TF-IDF engine
(`benchmark/run.py` without `--use-sulci`). The website (sulci.io
Architecture and Sulci docs tabs) quotes the same numbers without engine
context.

A user who installs sulci, runs `python3 benchmark/run.py --use-sulci`
(real production setup with MiniLM), will see different numbers:

- Stateless hit rate: ~94% (MiniLM is more sensitive than TF-IDF)
- Context-aware delta: ~+0pp (stateless is already so good there's no
  headroom for context to fill)

This is not dishonest - `benchmark/README.md` clearly distinguishes Mode 1
(synthetic TF-IDF) from Mode 2 (real MiniLM). But the OSS root README and
the website don't carry that nuance.

## Proposed fixes

1. README.md: under the "Benchmark results" callout, add a footnote
   explaining the engine and pointing at `benchmark/README.md` for details.

2. sulci-web (separate issue, separate repo): same footnote on sulci.io
   marketing pages.

## Why it matters

A user expecting +20.8pp improvement and seeing flat numbers in their own
MiniLM benchmark would feel misled. Honest framing prevents that.

## Effort

OSS README: ~3 lines added. sulci-web: separate issue.
""",
    },
    {
        "title": "benchmark: --use-sulci mode leaves SQLite db directories in benchmark/results/",
        "labels": ["benchmark", "low-priority"],
        "body": """\
## Problem

Running `python3 benchmark/run.py --use-sulci [...]` creates SQLite-backed
Cache instances that persist their data inside `benchmark/results/`:

- benchmark/results/ctx_bench_context_db/
- benchmark/results/ctx_bench_stateless_db/
- benchmark/results/sulci_bench_db/

These accumulate across runs (each run adds rows), are not auto-cleaned,
and used to pollute `git status` output. Now properly gitignored as of
v0.4.0 (pattern `results/*_db/`).

Disk space is small per run (~few MB), but it grows over many runs.

## Proposed fix

`benchmark/run.py` should either:
- Use `tempfile.TemporaryDirectory()` for sulci-mode runs, OR
- Clean its own output directories at end of run

## Priority

Low. Gitignore is the main mitigation; disk usage is small enough not to
be a real concern.

## Context

Surfaced during v0.4.0 benchmark verification.
""",
    },
    {
        "title": "api: reconsider `personalized` flag - `user_id` divergence with `tenant_id` for v0.5.0+",
        "labels": ["api-design", "breaking-change-candidate"],
        "body": """\
## Problem

v0.4.0 introduces `tenant_id` partition isolation as a first-class
ungated kwarg on `Cache.get/set/cached_call`. When passed, it is
honored unconditionally.

But `user_id` (which has been on the API since v0.1) continues to be
gated by the constructor flag `personalized=True`:

    # In Cache.get/set/cached_call body:
    user_id = user_id if self.personalized else None

If a user constructs `Cache(backend="sqlite")` (default
`personalized=False`) and calls
`cache.set(query, response, user_id="alice")`, `user_id` is silently
dropped at the Cache layer before reaching the backend. The entry
stores under `user_id="global"`.

This was a backwards-compatibility decision for v0.4.0. The asymmetry
between `tenant_id` (always honored) and `user_id` (gated) is
confusing for new users.

## Proposed reconciliation (for v0.5.0)

Three options to evaluate:

(a) Drop the `personalized` gate entirely. `user_id` is always honored
    when passed. Symmetric with `tenant_id`. Possibly mildly breaking
    for v0.3.x users; in practice, callers who pass `user_id` already
    expect it to be honored, so the breakage surface is small.

(b) Keep status quo and document the divergence. Users who want
    `user_id` honored set `personalized=True` at construction. The
    asymmetry stays but is no longer surprising.

(c) Repurpose the flag to gate something else (TBD via separate
    planning). `user_id` partitioning becomes ungated like `tenant_id`;
    `personalized` controls a different concern.

## Decision needed

Pick (a), (b), or (c) for v0.5.0. The (c) direction is being evaluated
as part of separate tier-strategy planning and is intentionally not
specified in this issue.

## Context

Surfaced during v0.4.0 phase 1.7b ("Plumb tenant_id through Cache
public API") when we decided how to wire tenant_id. We chose ungated
for tenant_id; this issue captures the follow-up question about
user_id parity.
""",
    },
]


def write_combined():
    """Write all 7 issues to one file with delimiters for inspection."""
    out = Path("/tmp/sulci_oss_issues.md")
    blocks = []
    for i, issue in enumerate(ISSUES, 1):
        block = (
            f"{'='*60}\n"
            f"ISSUE {i} / {len(ISSUES)}\n"
            f"{'='*60}\n"
            f"TITLE: {issue['title']}\n"
            f"LABELS: {', '.join(issue['labels'])}\n"
            f"{'='*60}\n\n"
            f"{issue['body']}"
        )
        blocks.append(block)
    out.write_text("\n".join(blocks))
    print(f"wrote {out} ({len(out.read_text().splitlines())} lines, "
          f"{len(ISSUES)} issues)")


def write_per_issue():
    """Write each issue to its own file, ready for `gh issue create -F`."""
    for i, issue in enumerate(ISSUES, 1):
        body_path = Path(f"/tmp/sulci_oss_issue_{i}_body.md")
        body_path.write_text(issue["body"])
        title_path = Path(f"/tmp/sulci_oss_issue_{i}_title.txt")
        title_path.write_text(issue["title"])
        labels_path = Path(f"/tmp/sulci_oss_issue_{i}_labels.txt")
        labels_path.write_text(",".join(issue["labels"]))
    print(f"wrote {len(ISSUES)} per-issue files to /tmp/sulci_oss_issue_*")


def emit_gh_script():
    """Print a bash script that creates all 7 issues via `gh`."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for i in range(1, len(ISSUES) + 1):
        lines.append(
            f'gh issue create '
            f'--title "$(cat /tmp/sulci_oss_issue_{i}_title.txt)" '
            f'--body-file /tmp/sulci_oss_issue_{i}_body.md '
            f'--label "$(cat /tmp/sulci_oss_issue_{i}_labels.txt)"'
        )
    script = "\n".join(lines) + "\n"
    Path("/tmp/create_sulci_oss_issues.sh").write_text(script)
    Path("/tmp/create_sulci_oss_issues.sh").chmod(0o755)
    print("wrote /tmp/create_sulci_oss_issues.sh (run after writing per-issue files)")


if __name__ == "__main__":
    write_combined()
    write_per_issue()
    emit_gh_script()
