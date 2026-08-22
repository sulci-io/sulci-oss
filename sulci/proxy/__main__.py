# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""Console-script entry point for ``sulci-proxy``."""
from __future__ import annotations

import argparse
import os
from typing import Optional


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sulci-proxy",
        description=(
            "OpenAI/Anthropic-compatible caching proxy. Point OPENAI_BASE_URL "
            "or ANTHROPIC_BASE_URL at it and every call is cached."
        ),
    )
    p.add_argument("--host", default=os.environ.get("SULCI_PROXY_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("SULCI_PROXY_PORT", "8787")))
    p.add_argument("--backend", default=os.environ.get("SULCI_BACKEND", "sqlite"))
    p.add_argument("--db-path", default=os.environ.get("SULCI_DB_PATH", "./sulci_db"))
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--context-window", type=int, default=4)
    p.add_argument("--ttl-seconds", type=int, default=None)
    p.add_argument("--openai-upstream", default=None)
    p.add_argument("--anthropic-upstream", default=None)
    p.add_argument(
        "--share-across-models",
        action="store_true",
        help="Serve a response cached for one model to another. Cheaper, "
             "occasionally wrong.",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    args = _parser().parse_args(argv)
    import uvicorn
    from sulci.proxy.app import build_app

    kwargs: dict = {
        "backend": args.backend,
        "db_path": args.db_path,
        "threshold": args.threshold,
        "context_window": args.context_window,
    }
    if args.ttl_seconds is not None:
        kwargs["ttl_seconds"] = args.ttl_seconds

    app = build_app(
        openai_upstream=args.openai_upstream,
        anthropic_upstream=args.anthropic_upstream,
        share_across_models=args.share_across_models,
        **kwargs,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
