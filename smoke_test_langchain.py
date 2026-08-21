# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
smoke_test_langchain.py
───────────────────────
Smoke test for the Sulci LangChain integration.

Runs automatically as part of setup.sh, and on demand via:
    python smoke_test_langchain.py
    make smoke-langchain

No LLM API key, no Docker, no external services required.
Uses the local SQLite backend only.

If langchain-core is not installed, prints a clear message and exits 0
(non-blocking) so the rest of setup.sh continues unaffected.

Install the integration extra to enable this test:
    pip install "sulci[sqlite,langchain]"
"""

import sys
import tempfile
import os


def _check_deps() -> bool:
    """Return True if all deps are present, False + print hint if not."""
    try:
        import langchain_core       # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError as e:
        print(f"  skipped — {e}")
        print("  To enable: pip install \"sulci[sqlite,langchain]\"")
        return False
    return True


def main() -> None:
    print("→ Running LangChain integration smoke test")

    if not _check_deps():
        print()
        return  # exit 0 — non-blocking

    from langchain_core.globals import get_llm_cache, set_llm_cache
    from langchain_core.outputs import Generation
    from sulci.integrations.langchain import SulciCache

    QUERY     = "What is semantic caching?"
    RESP      = "Semantic caching stores LLM responses by meaning."
    UNRELATED = "What is the boiling point of water at high altitude?"

    with tempfile.TemporaryDirectory() as tmp:
        sc = SulciCache(
            backend          = "sqlite",
            db_path          = os.path.join(tmp, "lc_smoke"),
            namespace_by_llm = False,
        )
        set_llm_cache(sc)

        # store — simulates LangChain caching an LLM response
        sc.update(QUERY, "", [Generation(text=RESP)])

        # exact match → hit
        result = sc.lookup(QUERY, "")
        assert result is not None, "FAIL: lookup returned None on exact match"
        assert result[0].text == RESP, f"FAIL: text mismatch — got {result[0].text!r}"
        print(f"  LangChain hit:  sim=1.000  ✅  {result[0].text[:45]}...")

        # unrelated query → miss
        miss = sc.lookup(UNRELATED, "")
        assert miss is None, f"FAIL: expected miss, got {miss!r}"
        print("  LangChain miss: confirmed  ✅")

        # stats
        stats = sc.stats()
        for key in ("hits", "misses", "hit_rate", "total_queries", "saved_cost"):
            assert key in stats, f"FAIL: stats() missing key '{key}'"
        print(f"  Stats: {get_llm_cache()}")

        set_llm_cache(None)

    print("  LangChain smoke test passed.  ✅\n")


if __name__ == "__main__":
    main()
