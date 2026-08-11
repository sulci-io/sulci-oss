# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kathiravan Sengodan
"""
examples/litellm_example.py
────────────────────────────
Sulci as the cache layer inside LiteLLM.

    pip install "sulci[litellm,sqlite]"
    OPENAI_API_KEY=sk-... python examples/litellm_example.py

LiteLLM has no `custom` cache type — the injection point is replacing the
inner implementation after constructing a Cache, which install() does.
"""
from __future__ import annotations

import time

import litellm

from sulci.integrations.litellm import install

QUESTION = "In two sentences, what is semantic caching?"


def main() -> None:
    # namespace_by_model=False: sqlite does not enforce tenant isolation, so
    # leaving it on would only produce a warning and no protection.
    adapter = install(
        backend="sqlite",
        db_path="./sulci_db_litellm",
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
