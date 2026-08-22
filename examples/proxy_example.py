# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
examples/proxy_example.py
──────────────────────────
The zero-code-change door: point an SDK at sulci-proxy and every call is
cached without importing sulci anywhere.

Run the proxy in one terminal:

    pip install "sulci[proxy,sqlite]"
    sulci-proxy --backend sqlite --db-path ./sulci_db --port 8787

Then this script in another:

    OPENAI_API_KEY=sk-... python examples/proxy_example.py

Expected: the first call is a miss and reaches OpenAI; the second is served
from cache with x-sulci-cache: hit and usage.total_tokens == 0.

⚠️  START THE PROXY WITH A FRESH --db-path, e.g.

        sulci-proxy --backend sqlite --db-path "$(mktemp -d)/cache" --port 8787

    `--db-path ./sulci_db` is sulci's DEFAULT path, shared with anything else
    in this repo that omits db_path. A previous run can leave a semantically
    similar entry there, and then the "first" call below comes back as a HIT
    and the demo proves nothing — silently. This script now asserts the first
    call is a miss rather than trusting the labels.

⚠️  A CACHE HIT IS NOT AUTHENTICATED. The proxy forwards your Authorization
    header upstream only on a MISS; a hit is served straight from the store
    without any credential check. That is inherent to any caching proxy, and
    it means an invalid or expired key still returns 200 on a cached
    question. Do not expose sulci-proxy beyond localhost or a trusted
    network segment.
"""
from __future__ import annotations

import os
import time

import httpx

PROXY = os.environ.get("SULCI_PROXY", "http://127.0.0.1:8787")
KEY = os.environ.get("OPENAI_API_KEY", "")
QUESTION = "In two sentences, what is semantic caching?"


def ask(client: httpx.Client, session: str) -> tuple:
    t0 = time.perf_counter()
    r = client.post(
        f"{PROXY}/v1/chat/completions",
        headers={
            "authorization": f"Bearer {KEY}",
            "content-type": "application/json",
            # Scope the lookup. NOTE: on the sqlite backend tenant_id is
            # accepted and IGNORED — only qdrant enforces isolation. See
            # sulci/integrations/_scope.py.
            "x-sulci-tenant-id": "examples/proxy_example",
            "x-sulci-session-id": session,
        },
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": QUESTION}]},
        timeout=120.0,
    )
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.headers.get("x-sulci-cache"), ms, r.json()


def main() -> None:
    if not KEY:
        raise SystemExit("Set OPENAI_API_KEY.")
    with httpx.Client() as c:
        results = []
        for label in ("first", "second"):
            state, ms, body = ask(c, session="demo-1")
            tokens = body.get("usage", {}).get("total_tokens")
            print(f"{label:<7} {state:<6} {ms:8.1f} ms  tokens={tokens}")
            results.append(state)
        print("stats:", c.get(f"{PROXY}/stats").json())

        # Assert rather than assume. A "first" that comes back as a hit means
        # the store was already warm, and every timing below it is then
        # meaningless as a demonstration.
        if results[0] != "miss":
            raise SystemExit(
                f"\nFAILED: the first call was '{results[0]}', not 'miss'.\n"
                "The proxy's --db-path already contained a similar entry, so\n"
                "this run demonstrated nothing about the miss path. Restart\n"
                'sulci-proxy with a fresh store:\n\n'
                '    sulci-proxy --backend sqlite --db-path "$(mktemp -d)/cache"\n'
            )
        if results[1] != "hit":
            raise SystemExit(f"\nFAILED: the second call was '{results[1]}', not 'hit'.")
        print("\nOK: miss then hit, as expected.")


if __name__ == "__main__":
    main()
