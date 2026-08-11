---
# GitHub Agentic Workflow — Sulci as a cache MCP server.
#
# Compile with:  gh aw compile
# Then commit the generated .lock.yml alongside this file.
#
# WHAT THIS DEMONSTRATES
# ──────────────────────
# gh-aw runs the agent inside a container behind a Squid egress proxy with a
# domain allowlist. A hosted cache would need `network.allowed` entries and a
# trust conversation. Sulci with backend=sqlite makes ZERO network calls, so
# there is nothing to allowlist — the store is a file.
#
# Persistence across runs comes from gh-aw's own `cache-memory` tool, which
# mounts /tmp/gh-aw/cache-memory/ backed by actions/cache. Point Sulci's
# db_path at it and the cache survives between runs for free.
#
# ⚠️ TTL MISMATCH, ON PURPOSE. sulci's ttl_seconds defaults to 86400 (24h);
#    the surrounding actions/cache retains for 7 days. Entries therefore
#    expire six days before the cache does. That is usually what you want in
#    CI — set --ttl-seconds explicitly if it is not.
#
# ⚠️ TENANT ISOLATION IS NOT ENFORCED ON SQLITE. Only the qdrant backend sets
#    ENFORCES_TENANT_ISOLATION = True. `--tenant-id` below is accepted and
#    IGNORED by the sqlite backend, and sulci will warn about it on startup.
#    It is present so the intent is recorded and so this workflow is correct
#    the day sqlite enforces — NOT because it protects you today. If you need
#    real per-SHA isolation right now, use backend=qdrant, or give each SHA
#    its own db_path under the cache-memory mount (see the commented line).
#
# ⚠️ AND THE CORRECTNESS PROBLEM THAT NO CONFIGURATION SOLVES. A cached answer
#    is keyed on the question. The repository is not in the question. A cached
#    "the auth module looks fine" from three commits ago is a confidently
#    wrong review of new code. For anything whose answer depends on the tree,
#    scope by commit SHA — and read the sqlite caveat above before assuming
#    that scoping is doing anything.

on:
  workflow_dispatch:
  schedule:
    - cron: "0 6 * * 1-5"

permissions:
  contents: read
  issues: write

engine: copilot

network:
  # Deliberately empty for the cache: backend=sqlite reaches nothing.
  # Add domains here only for what the AGENT itself needs.
  allowed: []

tools:
  # Persists /tmp/gh-aw/cache-memory/ across runs via actions/cache.
  # 7-day retention, 10GB/repo, LRU eviction, branch-scoped.
  cache-memory:
    allowed-extensions: [".db", ".sqlite", ".sqlite3", ".json"]

mcp-servers:
  sulci:
    command: "sulci-mcp"
    args:
      - "--backend=sqlite"
      - "--db-path=/tmp/gh-aw/cache-memory/sulci_db"
      # Per-SHA store. Uncomment INSTEAD of --tenant-id for isolation that
      # actually holds on sqlite — separate files cannot bleed into each other.
      # - "--db-path=/tmp/gh-aw/cache-memory/sulci_${{ github.sha }}"
      - "--tenant-id=${{ github.repository }}"
      - "--context-window=4"
      - "--transport=stdio"
    env:
      # stdio transport carries JSON-RPC on stdout; anything else printed
      # there corrupts the stream. sulci-mcp sets this itself, belt and braces.
      SULCI_QUIET: "1"
    allowed:
      - "cache_lookup"
      - "cache_store"
      - "cache_stats"

safe-outputs:
  create-issue:
    max: 1

timeout_minutes: 15
---

# Daily repository triage, with a cache

You are triaging this repository's open issues.

## Use the cache

Before doing expensive analysis on any question, call `cache_lookup` with the
question as `query` and today's workflow run id as `session_id`. If it returns
`cache_hit: true`, use the cached `response` instead of recomputing.

After you produce an analysis you expect to be asked for again, call
`cache_store` with the same `query` and your answer as `response`.

Call `cache_stats` at the end and include the hit rate in your summary, so the
value of the cache is measured rather than assumed.

## Do not cache these

Do not `cache_store` anything whose correctness depends on the current state of
the code — file contents, test results, review verdicts, dependency versions.
Those change between runs and a cached answer will be confidently wrong. Cache
only questions whose answers are stable: project conventions, architectural
explanations, "how do I run X", label taxonomies.

## Then

Summarise what changed since the last run and open a single issue with your
findings, including the cache hit rate from `cache_stats`.
