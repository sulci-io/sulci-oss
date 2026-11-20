# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
tests/test_observe.py — observe mode (oss-0070, 1.5.0): Cache(observe=True) serves nothing and records locally.

The promises under test, each its own section below:
  * a would-be hit is NEVER returned, by get, cached_call or AsyncCache, and the LLM answers every call;
  * it is a miss everywhere: stats, the event sink, telemetry;
  * the sweep floor finds candidates below the serving threshold; the label is taken against the serving threshold;
    the caller's own accept() is recorded ("would_refuse"), never acted on;
  * a set answering a would-be hit stores no entry (as after a live hit) and completes a review pair; any other set
    behaves as before (follow-ups included);
  * pairs: only at 0.80 or above, at most PAIR_CAP, a uniform sample, rated ones never replaced;
  * erasure: delete_user and clear take the stored text; an erasure that cannot be confirmed raises; recording
    that fails never breaks the call;
  * the file is 0600 and two processes can write it at once;
  * the summary carries no text and no hash; the report's numbers are the measured ones; no prices or verdict;
  * refused where it cannot work (the managed backend), never accepted and ignored.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import random
import re
import sqlite3
import stat
import sys

import pytest

import sulci
from sulci import Cache
from sulci import observe
from sulci.errors import SulciErasureError
from sulci.sinks import query_hash

KEY = "sk-sulci-" + "t" * 32
Q1 = "how do I reset my password"
Q1_REWORD = "how do i reset my password please"     # FakeEmbedder: ~0.95 against Q1
Q_OTHER = "what is the refund policy"


def _cache(tmp_path, fake_embedder, **kw):
    kw.setdefault("threshold", 0.85)
    kw.setdefault("observe", True)
    if kw["observe"]:
        kw.setdefault("observe_path", str(tmp_path / "obs.db"))
    return Cache(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder, telemetry=False,
                 **kw)


def _llm(calls):
    def fn(q):
        calls.append(q)
        return sulci.with_usage("ANSWER: " + q, {"input_tokens": 100, "output_tokens": 40})
    return fn


def _rows(path, sql, args=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


class Stores:
    """Counts the backend's store() calls."""

    def __init__(self, cache):
        self.n = 0
        inner = cache._backend.store

        def store(*a, **k):
            self.n += 1
            return inner(*a, **k)
        cache._backend.store = store


class Sink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


# ── never served ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_cached_call_never_serves_and_the_llm_answers_every_call(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    calls = []
    out = [cache.cached_call(q, _llm(calls)) for q in (Q1, Q1, Q1_REWORD, Q_OTHER)]
    assert [o["source"] for o in out] == ["llm"] * 4
    assert all(o["cache_hit"] is False for o in out)
    assert calls == [Q1, Q1, Q1_REWORD, Q_OTHER]
    assert all(o["response"].startswith("ANSWER: ") for o in out)


def test_get_returns_none_with_the_candidates_similarity(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    cache.set(Q1, "stored answer")
    resp, sim, depth = cache.get(Q1)
    assert resp is None and sim == pytest.approx(1.0) and depth == 0
    assert cache._observe_label() == observe.WOULD_HIT


def test_the_same_cache_without_observe_does_serve(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, observe=False)
    cache.set(Q1, "stored answer")
    assert cache.get(Q1)[0] == "stored answer"
    assert "observe" not in cache.stats()
    assert not os.path.exists(tmp_path / "obs.db")


def test_async_cache_passes_observe_through_and_never_serves(tmp_path, fake_embedder):
    from sulci import AsyncCache
    cache = AsyncCache(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder,
                       telemetry=False, threshold=0.85, observe=True, observe_path=str(tmp_path / "obs.db"))

    async def go():
        await cache.aset(Q1, "stored answer")
        return await cache.aget(Q1)
    assert asyncio.run(go())[0] is None
    assert _rows(str(tmp_path / "obs.db"), "SELECT label FROM observe_events WHERE kind='get'") == [("would_hit",)]


# ── a miss everywhere ────────────────────────────────────────────────────────────────────────────────────────────
def test_counted_and_emitted_as_a_miss(tmp_path, fake_embedder, monkeypatch):
    sink = Sink()
    cache = _cache(tmp_path, fake_embedder, event_sink=sink)
    cache._telemetry = True
    sent = []
    monkeypatch.setattr(sulci, "_emit", lambda name, payload: sent.append((name, payload)), raising=False)
    cache.set(Q1, "stored answer")
    cache.get(Q1)
    s = cache.stats()
    assert (s["hits"], s["misses"], s["saved_cost"]) == (0, 1, 0.0)
    assert s["observe"]["would_hit"] == 1 and s["observe"]["lookups"] == 1
    gets = [e for e in sink.events if e.event_type in ("hit", "miss")]
    assert [e.event_type for e in gets] == ["miss"]
    assert gets[0].metadata == {"observe": True}
    assert gets[0].matched_query_hash is None
    got = [p for n, p in sent if n == "cache.get"]
    assert got and got[0]["hits"] == 0 and got[0]["misses"] == 1 and got[0]["threshold"] == 0.85


def test_no_conversation_turn_is_recorded_for_a_would_be_hit(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, context_window=4)
    cache.set(Q1, "stored answer")
    cache.get(Q1, session_id="s1")
    assert cache._session_turns(cache._session_key("s1", None, None)) in (0, None)


# ── the sweep floor and the labels ───────────────────────────────────────────────────────────────────────────────
def test_the_floor_finds_a_candidate_below_the_serving_threshold(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, threshold=0.99)
    cache.set(Q1, "stored answer")
    resp, sim, _ = cache.get(Q1_REWORD)
    assert resp is None and 0.75 <= sim < 0.99
    row = _rows(str(tmp_path / "obs.db"), "SELECT label, similarity, serving_threshold FROM observe_events "
                                          "WHERE kind='get'")[0]
    assert row[0] == "would_miss" and row[1] == pytest.approx(sim) and row[2] == 0.99


def test_lookup_floor_is_the_lower_of_the_floor_and_the_threshold():
    assert observe.lookup_floor(0.85) == 0.75
    assert observe.lookup_floor(0.65) == 0.65


def test_a_threshold_below_the_floor_searches_at_the_threshold(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, threshold=0.5)
    seen = []
    inner = cache._backend.search_match if hasattr(cache._backend, "search_match") else cache._backend.search
    name = "search_match" if hasattr(cache._backend, "search_match") else "search"

    def spy(*a, **k):
        seen.append(k["threshold"])
        return inner(*a, **k)
    setattr(cache._backend, name, spy)
    cache.get(Q1)
    assert seen == [0.5]


def test_the_callers_accept_is_recorded_and_never_acted_on(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    cache.set(Q1, "stored answer")
    assert cache.get(Q1, accept=lambda r, s: False)[0] is None
    assert cache._observe_label() == observe.WOULD_REFUSE
    assert cache.get(Q1, accept=lambda r, s: True)[0] is None          # yes is not acted on either
    assert cache._observe_label() == observe.WOULD_HIT

    def boom(r, s):
        raise RuntimeError("guard down")
    assert cache.get(Q1, accept=boom)[0] is None
    assert cache._observe_label() == observe.WOULD_REFUSE
    labels = [r[0] for r in _rows(str(tmp_path / "obs.db"), "SELECT label FROM observe_events WHERE kind='get' "
                                                           "ORDER BY id")]
    assert labels == ["would_refuse", "would_hit", "would_refuse"]


# ── set after a lookup ───────────────────────────────────────────────────────────────────────────────────────────
def test_a_set_answering_a_would_be_hit_stores_nothing_and_completes_a_pair(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    cache.set(Q1, "stored answer")
    stores = Stores(cache)
    assert cache.get(Q1_REWORD)[0] is None
    assert cache._set(Q1_REWORD, "fresh answer") is False
    assert stores.n == 0
    pairs = _rows(str(tmp_path / "obs.db"), "SELECT question, cached, fresh, similarity FROM observe_pairs")
    assert len(pairs) == 1 and pairs[0][:3] == (Q1_REWORD, "stored answer", "fresh answer")
    assert pairs[0][3] >= 0.80


def test_a_set_after_a_would_be_miss_stores_as_before(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    stores = Stores(cache)
    cache.get(Q_OTHER)
    assert cache._set(Q_OTHER, "answer") is True
    assert stores.n == 1
    assert _rows(str(tmp_path / "obs.db"), "SELECT count(*) FROM observe_pairs")[0][0] == 0


def test_a_set_with_no_lookup_stores_as_before(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    stores = Stores(cache)
    cache.set(Q1, "answer")
    assert stores.n == 1


def test_the_follow_up_rule_is_unchanged(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, context_window=4, context_threshold=0.85)
    stores = Stores(cache)
    cache.get(Q_OTHER, session_id="s")
    assert cache._set(Q_OTHER, "first answer", session_id="s") is True
    cache.get("and how long does it take", session_id="s")
    stored, reason = cache._set("and how long does it take", "follow-up answer", session_id="s", with_reason=True)
    assert (stored, reason) == (False, "context_dependent")
    assert stores.n == 1
    sets = _rows(str(tmp_path / "obs.db"), "SELECT label, reason FROM observe_events WHERE kind='set' ORDER BY id")
    assert sets == [("stored", None), ("not_stored", "context_dependent")]


def test_the_pending_lookup_expires(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    cache.set(Q1, "stored answer")
    now = [1_000_000.0]
    cache._observer._clock = lambda: now[0]
    cache.get(Q1_REWORD)
    now[0] += observe.PENDING_TTL_S + 1
    stores = Stores(cache)
    assert cache._set(Q1_REWORD, "fresh answer") is True                # too late to pair: stored as any miss
    assert stores.n == 1


def test_the_set_must_match_scope_and_variant(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, personalized=True)
    cache.set(Q1, "stored answer", user_id="u1")
    cache.get(Q1, user_id="u1")
    stores = Stores(cache)
    assert cache._set(Q1, "fresh", user_id="u2") is True               # another user's set: not this lookup's
    assert cache._set(Q1, "fresh", user_id="u1") is False              # this one: the pair
    assert stores.n == 1


def test_token_counts_land_on_the_lookup(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    calls = []
    cache.cached_call(Q1, _llm(calls))
    cache.cached_call(Q1, _llm(calls))
    rows = _rows(str(tmp_path / "obs.db"), "SELECT label, input_tokens, output_tokens FROM observe_events "
                                           "WHERE kind='get' ORDER BY id")
    assert rows == [("would_miss", 100, 40), ("would_hit", 100, 40)]
    m = observe.measure([str(tmp_path / "obs.db")])
    s85 = [s for s in m["sweep"] if s["threshold"] == 0.85][0]
    assert (s85["would_hits"], s85["input_tokens_saved"], s85["output_tokens_saved"], s85["token_coverage"]) == \
        (1, 100, 40, 1.0)


# ── review pairs ─────────────────────────────────────────────────────────────────────────────────────────────────
def _recorder(tmp_path, seed=0, name="r.db"):
    return observe.Recorder(str(tmp_path / name), rng=random.Random(seed))


def test_pairs_only_at_or_above_080(tmp_path, fake_embedder):
    rec = _recorder(tmp_path)
    for sim, n in ((0.79, 0), (0.80, 1)):
        c = observe.Candidate()
        c.accept("cached", sim)
        rec.record_lookup(pending_key=f"k{sim}", query="q", scope_hash="s", cand=c, serving_threshold=0.75, depth=0,
                          latency_ms=1)
        rec.record_set(pending=rec.take_pending(f"k{sim}"), query="q", scope_hash="s", label="paired", reason=None,
                       fresh="fresh", usage=None)
        assert _rows(rec.path, "SELECT count(*) FROM observe_pairs")[0][0] == n


def test_pairs_are_capped_and_rated_ones_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "PAIR_CAP", 5)
    rec = _recorder(tmp_path)
    with rec._lock:
        for i in range(5):
            rec._save_pair(0, "s", 0.9, f"q{i}", "c", "f")
        rec._conn.execute("UPDATE observe_pairs SET rating = 'acceptable'")
        for i in range(50):
            rec._save_pair(0, "s", 0.9, f"later{i}", "c", "f")
    rows = _rows(rec.path, "SELECT question, rating FROM observe_pairs")
    assert len(rows) == 5 and all(r[1] == "acceptable" for r in rows)


def test_the_pair_sample_is_uniform(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "PAIR_CAP", 20)
    early = late = 0
    for trial in range(40):
        rec = _recorder(tmp_path, seed=trial, name=f"u{trial}.db")
        with rec._lock:
            for i in range(200):
                rec._save_pair(0, "s", 0.9, str(i), "c", "f")
        kept = [int(r[0]) for r in _rows(rec.path, "SELECT question FROM observe_pairs")]
        assert len(kept) == 20
        early += sum(1 for k in kept if k < 100)
        late += sum(1 for k in kept if k >= 100)
        rec.close()
    # 800 kept in all; uniform means about half from each half (binomial sd ~14).
    assert abs(early - late) < 120, (early, late)


# ── erasure ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _paired(cache, user=None):
    kw = {"user_id": user} if user else {}
    cache.set(Q1, "stored answer", **kw)
    cache.get(Q1_REWORD, **kw)
    cache.set(Q1_REWORD, "fresh answer", **kw)


def test_delete_user_erases_that_users_pairs_only(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, personalized=True)
    _paired(cache, "u1")
    _paired(cache, "u2")
    cache.get(Q1, user_id="u1")                                         # a pending answer for u1 too
    path = str(tmp_path / "obs.db")
    assert _rows(path, "SELECT count(*) FROM observe_pairs")[0][0] == 2
    out = cache.delete_user("u1")
    assert out["observe_pairs"] == 1
    assert _rows(path, "SELECT count(*) FROM observe_pairs")[0][0] == 1
    assert _rows(path, "SELECT count(*) FROM observe_pending WHERE scope_hash = ?",
                 (cache._observe_scope(None, "u1"),))[0][0] == 0


def test_delete_user_without_observe_keeps_its_result_shape(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, observe=False, personalized=True)
    assert cache.delete_user("u1") == {"entries": 0, "sessions": 0}


def test_clear_erases_every_pair_and_pending_answer_and_keeps_the_numbers(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder)
    _paired(cache)
    cache.get(Q1)
    path = str(tmp_path / "obs.db")
    cache.clear()
    assert _rows(path, "SELECT count(*) FROM observe_pairs")[0][0] == 0
    assert _rows(path, "SELECT count(*) FROM observe_pending")[0][0] == 0
    assert _rows(path, "SELECT count(*) FROM observe_events")[0][0] > 0


def test_an_erasure_that_cannot_be_confirmed_raises(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, personalized=True)
    _paired(cache, "u1")
    cache._observer._conn.close()
    with pytest.raises(SulciErasureError):
        cache.delete_user("u1")
    with pytest.raises(SulciErasureError):
        cache.clear()


def test_recording_that_fails_never_breaks_the_call(tmp_path, fake_embedder, caplog):
    cache = _cache(tmp_path, fake_embedder)
    cache._observer._conn.close()
    calls = []
    out = cache.cached_call(Q1, _llm(calls))
    assert out["source"] == "llm" and calls == [Q1]
    assert "could not record" in caplog.text
    assert Q1 not in caplog.text


# ── the file ─────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX permissions")
def test_the_file_is_readable_by_its_owner_only(tmp_path, fake_embedder):
    _cache(tmp_path, fake_embedder)
    assert stat.S_IMODE(os.stat(tmp_path / "obs.db").st_mode) == 0o600


def _write_many(path, n):
    rec = observe.Recorder(path)
    for i in range(n):
        c = observe.Candidate()
        c.accept("x", 0.9)
        rec.record_lookup(pending_key=f"{os.getpid()}-{i}", query=f"q{i}", scope_hash="s", cand=c,
                          serving_threshold=0.85, depth=0, latency_ms=1)
    rec.close()


def test_two_processes_can_write_one_file(tmp_path):
    path = str(tmp_path / "shared.db")
    observe.Recorder(path).close()                       # created once, then two writers
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_write_many, args=(path, 150)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert _rows(path, "SELECT count(*) FROM observe_events")[0][0] == 300
    assert _rows(path, "SELECT count(*) FROM observe_run")[0][0] == 3


# ── the report and the summary ───────────────────────────────────────────────────────────────────────────────────
def test_wilson_upper_matches_the_platforms_formula():
    assert observe.wilson_upper(0, 0) is None
    assert observe.wilson_upper(0, 100) == pytest.approx(0.0370, abs=1e-4)
    assert observe.wilson_upper(5, 100) == pytest.approx(0.1118, abs=1e-4)
    assert observe.wilson_upper(1, 1) == 1.0


def test_the_sweep_counts_candidates_by_similarity(tmp_path):
    rec = _recorder(tmp_path)
    for i, (sim, guard) in enumerate(((0.76, None), (0.82, None), (0.88, True), (0.91, False), (0.97, None),
                                      (None, None))):
        c = observe.Candidate(guard=(None if guard is None else (lambda r, s, g=guard: g)))
        if sim is not None:
            c.accept("x", sim)
        rec.record_lookup(pending_key=str(i), query=f"q{i}", scope_hash="s", cand=c, serving_threshold=0.85,
                          depth=0, latency_ms=1)
    rec.record_skip("stream")
    m = observe.measure([rec.path])
    got = {s["threshold"]: s["would_hits"] for s in m["sweep"]}
    assert got == {0.75: 4, 0.80: 3, 0.85: 2, 0.90: 1, 0.95: 1}          # the 0.91 refused by the guard never counts
    assert m["calls"] == 7 and m["cacheable"] == 6
    assert m["by_label"] == {"would_miss": 3, "would_hit": 2, "would_refuse": 1}
    reasons = {(r["label"], r["reason"]): r["count"] for r in m["not_cacheable_by_reason"]}
    assert reasons[("skipped", "stream")] == 1 and reasons[("would_refuse", "your accept() said no")] == 1


def test_ratings_give_the_bound_per_threshold(tmp_path):
    rec = _recorder(tmp_path)
    with rec._lock:
        for i, (sim, rating) in enumerate(((0.82, "not_acceptable"), (0.86, "acceptable"), (0.92, "acceptable"),
                                           (0.96, "cant_tell"))):
            rec._save_pair(0, "s", sim, f"q{i}", "c", "f")
            rec._conn.execute("UPDATE observe_pairs SET rating = ? WHERE question = ?", (rating, f"q{i}"))
    m = observe.measure([rec.path])
    by = {s["threshold"]: s for s in m["sweep"]}
    assert (by[0.80]["rated"], by[0.80]["not_acceptable"], by[0.80]["cant_tell"]) == (3, 1, 1)
    assert by[0.80]["wrong_hit_upper_95"] == pytest.approx(observe.wilson_upper(1, 3))
    assert (by[0.85]["rated"], by[0.85]["not_acceptable"]) == (2, 0)
    assert m["pairs"] == {"kept": 4, "rated": 3, "cant_tell": 1, "exact_repeats": 0}
    assert m["enough"] is False
    assert "Not enough to tell yet" in observe.render(m)


def test_the_summary_holds_no_text_and_no_hashes(tmp_path, fake_embedder):
    cache = _cache(tmp_path, fake_embedder, personalized=True)
    secret_q, secret_a = "zebracorn quokka password", "flamingo nebula answer"
    cache.set(secret_q, secret_a, user_id="alice@example.com")
    cache.get(secret_q, user_id="alice@example.com")
    cache.set(secret_q, "pelican answer", user_id="alice@example.com")
    out = tmp_path / "run.json"
    assert observe.main(["summary", str(tmp_path / "obs.db"), "--out", str(out)]) == 0
    text = out.read_text()
    for word in ("zebracorn", "quokka", "flamingo", "nebula", "pelican", "alice", "example.com"):
        assert word not in text
    for h in (query_hash(secret_q), query_hash("alice@example.com"), cache._observe_scope(None, "alice@example.com")):
        assert h not in text
    assert not re.search(r"\b[0-9a-f]{16,}\b", text)
    data = json.loads(text)
    assert data["format"] == observe.SUMMARY_FORMAT and data["pairs"]["kept"] == 1


def test_the_library_holds_no_prices_and_no_verdict():
    """OSS boundary policy §1: no billing, plan or pricing logic. The verdict compares savings with plan prices,
    so it stays in the managed platform; the library reports measured numbers only."""
    import pathlib
    text = pathlib.Path(observe.__file__).read_text(encoding="utf-8").lower()
    for word in ("price", "verdict", "worth it", "$", " plan "):
        if word in ("verdict", "price"):
            # allowed only in the sentence that says there are none
            assert all("no prices" in line or "no verdict" in line or "no prices, no verdict" in line
                       for line in text.splitlines() if word in line), word
        else:
            assert word not in text, word


# ── the command line ─────────────────────────────────────────────────────────────────────────────────────────────
def test_rate_export_import_and_forget(tmp_path, fake_embedder, capsys):
    cache = _cache(tmp_path, fake_embedder)
    _paired(cache)
    path = str(tmp_path / "obs.db")
    answers = iter(["x", "n"])
    shown = []
    assert observe.rate(path, reviewer="r1", read=lambda prompt: next(answers), write=shown.append) == 1
    assert not any("0.9" in s for s in shown)                           # blind: no similarity shown
    assert _rows(path, "SELECT rating, rated_by FROM observe_pairs") == [("not_acceptable", "r1")]
    csv_path = str(tmp_path / "pairs.csv")
    assert observe.export_pairs(path, csv_path) == 1
    text = open(csv_path, encoding="utf-8").read().replace("not_acceptable", "acceptable")
    open(csv_path, "w", encoding="utf-8").write(text)
    assert observe.import_ratings(path, csv_path, reviewer="r2") == 1
    assert _rows(path, "SELECT rating FROM observe_pairs") == [("acceptable",)]
    open(csv_path, "w", encoding="utf-8").write("pair_id,question,cached_answer,fresh_answer,rating\n1,q,c,f,maybe\n")
    with pytest.raises(ValueError):
        observe.import_ratings(path, csv_path)
    assert observe.main(["status", path]) == 0
    assert observe.main(["report", path]) == 0
    assert "Hit rate by threshold" in capsys.readouterr().out
    assert observe.main(["forget", path]) == 0
    assert _rows(path, "SELECT count(*) FROM observe_pairs")[0][0] == 0


def test_several_files_add_up(tmp_path):
    paths = []
    for i in range(2):
        rec = _recorder(tmp_path, name=f"h{i}.db")
        c = observe.Candidate()
        c.accept("x", 0.9)
        rec.record_lookup(pending_key="k", query="q", scope_hash="s", cand=c, serving_threshold=0.85, depth=0,
                          latency_ms=1)
        paths.append(rec.path)
    m = observe.measure(paths)
    assert m["cacheable"] == 2 and len(m["runs"]) == 2


def test_python_dash_m_runs(tmp_path):
    import subprocess
    path = str(tmp_path / "x.db")
    observe.Recorder(path).close()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    r = subprocess.run([sys.executable, "-m", "sulci.observe", "status", path], capture_output=True, text=True,
                       env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "calls 0" in r.stdout


# ── refused where it cannot work ─────────────────────────────────────────────────────────────────────────────────
def test_refused_on_the_managed_backend():
    with pytest.raises(TypeError, match="Observe mode in the dashboard"):
        Cache(backend="sulci", api_key=KEY, telemetry=False, observe=True)


def test_async_cache_refused_on_the_managed_backend():
    from sulci import AsyncCache
    with pytest.raises(TypeError, match="Observe mode"):
        AsyncCache(backend="sulci", api_key=KEY, telemetry=False, observe=True)


def test_observe_must_be_a_bool_and_the_path_needs_it(tmp_path, fake_embedder):
    with pytest.raises(TypeError):
        _cache(tmp_path, fake_embedder, observe="yes")
    with pytest.raises(ValueError, match="only with observe=True"):
        Cache(backend="sqlite", db_path=str(tmp_path / "db"), embedding_model=fake_embedder, telemetry=False,
              observe_path=str(tmp_path / "x.db"))
