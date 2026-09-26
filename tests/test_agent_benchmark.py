import importlib.util
import sys
import uuid
from pathlib import Path


def _load_benchmark_module(monkeypatch, tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "benchmark" / "run.py"
    monkeypatch.setattr(sys, "argv", ["benchmark/run.py", "--out", str(tmp_path)])
    module_name = f"_sulci_benchmark_run_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _use_prompts(module, prompts):
    remaining = iter(prompts)
    module._generate_agent_prompt = lambda rng: next(remaining)


class _ExactCache:
    def __init__(self, threshold):
        self._entries = {}

    def get(self, query):
        if query in self._entries:
            return self._entries[query], 1.0, None
        return None, 0.0, None

    def set(self, query, response, group="", domain=""):
        self._entries[query] = response


def _use_exact_cache(module):
    module._BuiltinCache = _ExactCache


class _FakeClaude:
    def __init__(self, *, max_calls, latencies_ms, costs):
        self.model = "fake-haiku"
        self.max_calls = max_calls
        self.call_count = 0
        self._latencies_ms = latencies_ms
        self._costs = costs

    def call(self, prompt):
        if self.call_count >= self.max_calls:
            return None, 0.0, 0.0
        idx = self.call_count
        self.call_count += 1
        latency_ms = self._latencies_ms[min(idx, len(self._latencies_ms) - 1)]
        cost = self._costs[min(idx, len(self._costs) - 1)]
        return f"real response {idx} for {prompt}", latency_ms, cost


def test_agent_benchmark_preserves_synthetic_mode(monkeypatch, tmp_path):
    bench = _load_benchmark_module(monkeypatch, tmp_path)
    bench._claude = None
    _use_exact_cache(bench)
    _use_prompts(
        bench,
        [
            ("structural", "repeat prompt"),
            ("structural", "repeat prompt"),
        ],
    )

    data = bench.run_agent_bench(n_sessions=1, dispatches=2, use_sulci=False)

    assert data["summary"]["total_hits"] == 1
    assert data["summary"]["total_misses"] == 1
    assert "claude_active" not in data["summary"]
    assert "llm_seconds_saved" not in data["per_session"][0]


def test_agent_benchmark_uses_claude_on_misses_and_counts_hit_savings(
    monkeypatch, tmp_path
):
    bench = _load_benchmark_module(monkeypatch, tmp_path)
    _use_exact_cache(bench)
    bench._claude = _FakeClaude(
        max_calls=10,
        latencies_ms=[1000.0, 2000.0],
        costs=[0.01, 0.02],
    )
    _use_prompts(
        bench,
        [
            ("structural", "prompt a"),
            ("structural", "prompt a"),
            ("novel", "prompt b"),
            ("novel", "prompt b"),
        ],
    )

    data = bench.run_agent_bench(n_sessions=1, dispatches=4, use_sulci=False)
    summary = data["summary"]

    assert bench._claude.call_count == 2
    assert summary["total_hits"] == 2
    assert summary["total_misses"] == 2
    assert summary["claude_active"] is True
    assert summary["claude_calls_made"] == 2
    assert summary["claude_avg_latency_ms"] == 1500.0
    assert summary["total_llm_seconds_saved"] == 2.5
    assert summary["total_dollars_saved"] == 0.025
    assert data["per_session"][0]["llm_seconds_saved"] == 2.5
    assert data["per_session"][0]["dollars_saved"] == 0.025


def test_agent_benchmark_falls_back_after_claude_cap(monkeypatch, tmp_path):
    bench = _load_benchmark_module(monkeypatch, tmp_path)
    _use_exact_cache(bench)
    bench._claude = _FakeClaude(
        max_calls=1,
        latencies_ms=[1000.0],
        costs=[0.01],
    )
    _use_prompts(
        bench,
        [
            ("structural", "prompt a"),
            ("semi_structural", "prompt b"),
            ("novel", "prompt c"),
            ("structural", "prompt a"),
        ],
    )

    data = bench.run_agent_bench(n_sessions=1, dispatches=4, use_sulci=False)
    summary = data["summary"]

    assert bench._claude.call_count == 1
    assert summary["claude_calls_made"] == 1
    assert summary["claude_cap_reached"] is True
    assert summary["claude_cap_reached_session"] == 1
    assert summary["claude_fallback_misses"] == 2
    assert summary["total_hits"] == 1
    assert summary["total_misses"] == 3
    assert summary["total_llm_seconds_saved"] == 1.0
    assert summary["total_dollars_saved"] == 0.01
