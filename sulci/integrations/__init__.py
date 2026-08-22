# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci.integrations
──────────────────
Optional framework adapters for Sulci Cache.

Each sub-module guards its imports — if the target framework is not
installed, importing the module raises a clear ImportError with an
install hint. The core sulci package never depends on any framework.

Available:
    sulci.integrations.langchain   →  SulciCache(BaseCache) for LangChain
                                       pip install "sulci[langchain]"

    sulci.integrations.llamaindex  →  SulciCacheLLM(LLM) for LlamaIndex
                                       pip install "sulci[llamaindex]"
"""