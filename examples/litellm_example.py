# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
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


def _install_mock_llm() -> None:
    """Register a mock provider so the example runs with no credentials.

    Every other example in this directory degrades to a mock rather than
    raising (see anthropic_example.py:37, agent_example_crewai.py). The
    first version of this file did not, and exited with a 60-line litellm
    traceback -- which reads like the ADAPTER is broken when in fact only
    the key is missing.
    """
    import hashlib

    def _mock_completion(model, messages, **kwargs):
        from litellm.types.utils import ModelResponse

        prompt = messages[-1]["content"] if messages else ""
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:6]
        time.sleep(0.4)  # stand in for network latency on the miss path
        return ModelResponse(
            **{
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": (
                                f"[Mock {digest}] Semantic caching stores LLM "
                                "responses indexed by meaning rather than by "
                                "exact text, so near-duplicate questions reuse "
                                "one answer."
                            ),
                        },
                    }
                ],
                "model": model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
            }
        )

    litellm.completion = _mock_completion


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠  OPENAI_API_KEY not set — using mock LLM "
              "(set it for real miss-path timings)\n")
        _install_mock_llm()

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
        )
        ms = (time.perf_counter() - t0) * 1000
        text = resp["choices"][0]["message"]["content"][:60]
        print(f"{label:<7} {ms:8.1f} ms  {text}...")

    print("stats:", adapter.stats())


if __name__ == "__main__":
    main()
