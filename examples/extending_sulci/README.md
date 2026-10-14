# Extending Sulci

This directory contains **reference implementations** of sulci's public protocols.

These files are different in purpose from the other examples in `examples/`:

- `examples/*.py` — *usage* examples ("how do I use sulci in my LangChain app?")
- `examples/extending_sulci/*.py` — *extension* examples ("how do I write my own backend?")

If you want to implement a custom backend or embedder for sulci, start here. Each
file is a worked example you can copy into your own codebase, adapt to your
infrastructure, and verify against sulci's conformance test suite.

## What's here

- **`custom_backend.py`** — In-memory dict-based backend that satisfies the full
  Backend protocol. Run it directly to see it pass a self-test, or load
  `InMemoryBackend` into your own conformance suite to validate your environment.

## Verifying your custom implementation

1. Implement your class against the protocol surface in
   `sulci/backends/protocol.py` (or `sulci/embeddings/protocol.py`).
2. Add your class to `BACKEND_CLASSES` (or `EMBEDDER_CLASSES`) in
   `tests/compat/conftest.py`.
3. Run `pytest tests/compat/` and confirm the conformance suite passes.

See `docs/protocols.md` for the full protocol contract, and
`docs/multi_tenancy_and_isolation.md` for the tenant and user isolation
guarantees your backend must satisfy. Every shipped backend sets
`ENFORCES_TENANT_ISOLATION = True` since 0.9.3, and so does `InMemoryBackend`;
a new backend is expected to as well.
