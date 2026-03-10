"""
sulci — Semantic caching for LLM applications.
Stop paying for the same answer twice.

Install:
    pip install "sulci[chroma]"   # ChromaDB backend
    pip install "sulci[sqlite]"   # SQLite (zero infra)
    pip install "sulci[faiss]"    # FAISS backend
    pip install "sulci[qdrant]"   # Qdrant backend
    pip install "sulci[all]"      # everything

Stateless quickstart:
    from sulci import Cache

    cache  = Cache(backend="sqlite", threshold=0.85)
    result = cache.cached_call("What is Python?", my_llm_fn)
    print(result["source"])    # "cache" or "llm"

Context-aware quickstart:
    from sulci import Cache

    cache = Cache(backend="sqlite", context_window=6)

    cache.cached_call("My container keeps crashing", my_llm_fn, session_id="u1")
    result = cache.cached_call("How do I fix it?",   my_llm_fn, session_id="u1")
    # ^ resolved in Docker/container context, not a random match
    print(result["context_depth"])  # 1
"""
from sulci.core import Cache
from sulci.context import ContextWindow, SessionStore

__version__ = "0.2.0"
__all__     = ["Cache", "ContextWindow", "SessionStore"]
