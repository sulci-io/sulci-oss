# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
examples/litellm_example.py
────────────────────────────
Sulci as the cache layer inside LiteLLM.

    pip install "sulci[litellm,sqlite]"
    OPENAI_API_KEY=sk-... python examples/litellm_example.py

Runs WITHOUT a key too: it falls back to a mock LLM, exactly like
examples/anthropic_example.py and agent_example_crewai.py. The cache
behaviour being demonstrated is identical either way -- only the miss-path
latency is real when a key is present.

LiteLLM has no `custom` cache type — the injection point is replacing the
inner implementation after constructing a Cache, which install() does.
"""
from __future__ import annotations

import os
import tempfile
import time

import litellm

from sulci.integrations.litellm import install

QUESTION = "In two sentences, what is semantic caching?"

# Per-run tempdir — see examples/basic_usage.py and issue #19.
_DB_PATH = os.path.join(tempfile.mkdtemp(prefix="sulci_litellm_"), "cache")


def main() -> None:
    # LiteLLM's OWN mock_response kwarg, not a monkeypatch of
    # litellm.completion. This matters: the cache is consulted INSIDE the
    # litellm.completion wrapper, so replacing that function removes the
    # entire caching path -- the first version of this fallback did exactly
    # that and printed hits=0, misses=0, total_queries=0 while still exiting
    # 0. mock_response goes through the full wrapper, so the cache is
    # exercised identically to a real call.
    mock_kwargs = {}
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠  OPENAI_API_KEY not set — using litellm's mock_response "
              "(set a key for real miss-path timings)\n")
        mock_kwargs["mock_response"] = (
            "Semantic caching stores LLM responses indexed by meaning rather "
            "than exact text, so near-duplicate questions reuse one answer."
        )

    # namespace_by_model=False: sqlite does not enforce tenant isolation, so
    # leaving it on would only produce a warning and no protection.
    adapter = install(
        backend="sqlite",
        db_path=_DB_PATH,
        context_window=4,
        namespace_by_model=False,
    )

    for label in ("first", "second"):
        t0 = time.perf_counter()
        resp = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": QUESTION}],
            metadata={"sulci_session_id": "demo-1"},
            **mock_kwargs,
        )
        ms = (time.perf_counter() - t0) * 1000
        text = resp["choices"][0]["message"]["content"][:60]
        print(f"{label:<7} {ms:8.1f} ms  {text}...")

    stats = adapter.stats()
    print("stats:", stats)

    # Assert rather than assume. Without this, a fallback that bypasses the
    # cache entirely still exits 0 and `make examples` reports PASS.
    if stats["total_queries"] != 2 or stats["hits"] != 1:
        raise SystemExit(
            f"\nFAILED: expected 1 hit / 1 miss, got {stats}.\n"
            "The cache was not consulted -- litellm.completion is the layer "
            "that consults it, so anything replacing that function removes "
            "the caching path."
        )
    print("\nOK: miss then hit, cache consulted on both calls.")


if __name__ == "__main__":
    main()
