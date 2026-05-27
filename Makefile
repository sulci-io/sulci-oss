# sulci-oss Makefile
# ─────────────────────────────────────────────────────────────────────────────

# ── macOS faiss/torch libomp workaround (issue #43) ─────────────────────
# faiss-cpu and torch each ship their own libomp on macOS. When a single
# Python process loads both (which `make test-all` and `make verify` do),
# Intel's OpenMP runtime aborts unless this flag is set. Linux uses one
# system libomp and is unaffected. See CONTRIBUTING.md for details.
export KMP_DUPLICATE_LIB_OK := TRUE

PYTHON = python3

# ── Smoke tests ───────────────────────────────────────────────────────────────

## Run all smoke tests (core + LangChain + LlamaIndex + AsyncCache)
smoke:
	@echo "── Core smoke test ─────────────────────────────────────────────────"
	$(PYTHON) smoke_test.py
	@echo ""
	@echo "── LangChain integration smoke test ────────────────────────────────"
	$(PYTHON) smoke_test_langchain.py
	@echo ""
	@echo "── LlamaIndex integration smoke test ───────────────────────────────"
	$(PYTHON) smoke_test_llamaindex.py
	@echo ""
	@echo "── AsyncCache smoke test ───────────────────────────────────────────"
	$(PYTHON) smoke_test_async.py

## Run core smoke test only (no LangChain required)
smoke-core:
	$(PYTHON) smoke_test.py

## Run LangChain integration smoke test only
## Requires: pip install "sulci[sqlite,langchain]"
smoke-langchain:
	$(PYTHON) smoke_test_langchain.py

## Run LlamaIndex integration smoke test only
## Requires: pip install "sulci[sqlite,llamaindex]"
smoke-llamaindex:
	$(PYTHON) smoke_test_llamaindex.py

## Run AsyncCache smoke test only
## Requires: pip install "sulci[sqlite]"
smoke-async:
	$(PYTHON) smoke_test_async.py

# ── Tests ─────────────────────────────────────────────────────────────────────

## Run core test suite (test_core, test_context, test_backends, test_connect, test_cloud_backend, test_config, test_telemetry, test_nudge)
test:
	python -m pytest tests/test_core.py \
	                 tests/test_context.py \
	                 tests/test_backends.py \
	                 tests/test_connect.py \
	                 tests/test_cloud_backend.py \
	                 tests/test_config.py \
	                 tests/test_telemetry.py \
	                 tests/test_nudge.py \
	                 -v --tb=short

## Run AsyncCache tests only
test-async:
	python -m pytest tests/test_async_cache.py -v --tb=short

## Run integration tests (LangChain + LlamaIndex)
test-integrations:
	python -m pytest tests/test_integrations_langchain.py \
	                 tests/test_integrations_llamaindex.py \
	                 -v --tb=short

## Run all tests (core + async + all integrations)
test-all:
	python -m pytest tests/ -v --tb=short

## Run all tests with coverage report
test-cov:
	python -m pytest tests/ -v --cov=sulci --cov-report=term-missing

# ── Combined: smoke + tests ───────────────────────────────────────────────────

## Full local verification: smoke tests + full test suite
verify: smoke test-all

# ── Developer tooling (scripts/) ──────────────────────────────────────────────

## Run pytest test files one at a time, in fresh subprocesses (see scripts/README.md)
test-per-file:
	$(PYTHON) scripts/run_tests_per_file.py

## Run pytest one at a time, skipping the slowest 4 files (faster local iteration)
test-per-file-fast:
	$(PYTHON) scripts/run_tests_per_file.py --skip-slow

## Run every example + smoke test with timeout, capture pass/fail
## Mock LLM fallback if no API keys; real LLMs if OPENAI/ANTHROPIC keys are set
examples:
	$(PYTHON) scripts/run_examples.py

## Verify framework-integration examples (langchain + llamaindex) by
## exercising every LLM-credential configuration: no keys / OpenAI only /
## Anthropic only / both keys. Requires both OPENAI_API_KEY and
## ANTHROPIC_API_KEY in env (uses real API calls; ~$0.10-0.20 per run).
verify-integration-examples:
	$(PYTHON) scripts/verify_integration_examples.py

## Verify the canonical TF-IDF benchmark numbers haven't regressed
## against benchmark/baseline.json (~15s wall-clock, no network/API).
benchmark-verify:
	$(PYTHON) scripts/verify_benchmark.py

## Run the agent-workload benchmark and verify against the pinned baseline.
## Distinct from `make checkin` because it adds ~30s wall-clock and the
## headline numbers (cost saved, calls-per-session) are most useful for
## release-prep / launch-post material rather than every commit.
##
## NB: --queries default (5000) is required so the stateless block matches
## the baseline.json numbers. The agent block adds ~15s on top.
##
## Use --use-sulci for the real-MiniLM run (slower, requires sulci[sqlite]
## installed; produces the conservative numbers we cite externally).
benchmark-agent:
	$(PYTHON) benchmark/run.py --agent --context --no-sweep
	$(PYTHON) scripts/verify_benchmark.py --skip-run

## Comprehensive pre-PR check: smoke + tests-per-file + examples
## Add 'matrix' manually if you want to also verify provider detection
checkin: smoke test-per-file examples benchmark-verify
	@echo ""
	@echo "════════════════════════════════════════════════════════════════════"
	@echo " ✓ checkin verification complete"
	@echo "   For provider-detection coverage too: make verify-integration-examples"
	@echo "   For agent benchmark too:              make benchmark-agent"
	@echo "════════════════════════════════════════════════════════════════════"

# ── PHONY ─────────────────────────────────────────────────────────────────────

.PHONY: smoke smoke-core smoke-langchain smoke-llamaindex smoke-async \
        test test-async test-integrations test-all test-cov \
        test-per-file test-per-file-fast examples verify-integration-examples \
        benchmark-verify benchmark-agent checkin verify
