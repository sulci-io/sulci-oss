# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci.observe — observe mode: measure what the cache WOULD serve, serving nothing (oss-0070, 1.5.0).

``Cache(observe=True)`` looks up as usual and then refuses every hit, so the caller's LLM answers every call. What
the cache would have done is recorded locally, in one SQLite file (``./sulci_observe.db`` unless ``observe_path=``
says otherwise). Nothing is sent anywhere. This module holds the recorder, the measured report and its command line::

    python -m sulci.observe status  sulci_observe.db
    python -m sulci.observe rate    sulci_observe.db        # your reviewers rate would-be hits, blind
    python -m sulci.observe report  sulci_observe.db        # the measured numbers
    python -m sulci.observe summary sulci_observe.db --out run.json   # the same numbers, no text
    python -m sulci.observe forget  sulci_observe.db        # delete every stored question and answer

HOW A WOULD-BE HIT STAYS A MISS EVERYWHERE. The lookup's candidate goes through a ``Candidate`` used as
``Cache.get``'s ``accept`` (oss-0064), which records it and always answers False. ``get`` then returns
``(None, similarity, depth)``, counts a miss, records no conversation turn and emits a ``miss`` event with
``metadata={"observe": True}``. Your own ``accept=``, if you pass one, is still run on the candidate and its answer
recorded ("would_refuse"), never acted on.

THE SWEEP FLOOR. Backends report similarity 0.0 on a miss, so the lookup asks for candidates down to ``FLOOR`` (or the
serving threshold, if that is lower). The report can then show the hit rate at thresholds above and below yours.
Each call's label is still taken against the threshold that would have served it.

WHAT IS STORED. One row per lookup and per ``set``: the label, the similarity, thresholds, timings, token counts and
SHA-256 hashes (as ``sulci.sinks.query_hash``). Text only in two places: a would-be hit's cached answer while it
waits up to ``PENDING_TTL_S`` for its ``set`` ("pending"), and the review pairs (the question, the answer the cache
would have served, your LLM's answer), at most ``PAIR_CAP``, a uniform random sample, only at ``PAIR_MIN_SIMILARITY``
or above. The file is created readable by its owner only (0600). It is not encrypted: keep it where you keep the
traffic it measures, and run ``forget`` (or ``Cache.clear()``) when the review is done. ``Cache.delete_user()`` also
erases that user's pending answers and pairs.

FAIL-SOFT. Recording never changes what the caller gets: an error is logged (its type only) and the call goes on.
Erasure is the exception: ``forget`` raises if it cannot confirm the deletion.

The report states measured numbers only: no prices, no verdict. ``summary`` writes the same numbers as JSON with no
text and no hashes, for you to keep or share. See docs/observe.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sqlite3
import sys
import threading
import time
import uuid
from collections import Counter
from typing import Any, Callable, Iterable, List, Optional, Sequence

_log = logging.getLogger("sulci")

#: The lowest similarity a candidate is looked for at (the report's sweep starts here).
FLOOR = 0.75
#: Review pairs are kept only for would-be hits at this similarity or above.
PAIR_MIN_SIMILARITY = 0.80
#: At most this many review pairs per file, as a uniform random sample (rated pairs are never replaced).
PAIR_CAP = 400
#: How long a lookup waits for its ``set`` (the review pair and the call's token counts).
PENDING_TTL_S = 600
#: The thresholds the report sweeps.
THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
#: Below these, the report says there is not enough to tell.
MIN_CACHEABLE = 500
MIN_RATED = 100
#: The summary file's format.
SUMMARY_FORMAT = "sulci-observe-summary/1"
#: Ratings, as the managed platform's review uses them.
RATINGS = ("acceptable", "not_acceptable", "cant_tell")

WOULD_HIT = "would_hit"
WOULD_MISS = "would_miss"
WOULD_REFUSE = "would_refuse"
LOOKUP_LABELS = (WOULD_HIT, WOULD_MISS, WOULD_REFUSE)
#: ``set`` labels: the fresh answer completed a would-be hit (no entry written, as on a live hit); stored; not stored.
PAIRED, STORED, NOT_STORED = "paired", "stored", "not_stored"
#: Calls that never reached a lookup (sulci-proxy: a stream).
SKIPPED = "skipped"

_EPS = 1e-9


def lookup_floor(serving_threshold: float) -> float:
    """The threshold an observed lookup searches at: ``FLOOR``, or the serving threshold when that is lower."""
    return min(FLOOR, float(serving_threshold))


def _hash(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    from sulci.sinks import query_hash
    return query_hash(text)


class Candidate:
    """``Cache.get``'s ``accept`` for an observed lookup: records the best candidate and ALWAYS refuses it.

    ``guard`` is the caller's own ``accept``, if any: its answer is recorded, never acted on. One per lookup."""

    def __init__(self, guard: Optional[Callable[[str, float], bool]] = None):
        self._guard = guard
        self.response: Optional[str] = None
        self.similarity: Optional[float] = None
        self.guard_ok: Optional[bool] = None

    def accept(self, response, similarity) -> bool:
        self.response = response if isinstance(response, str) else None
        self.similarity = float(similarity)
        if self._guard is not None:
            try:
                self.guard_ok = self._guard(response, similarity) is True
            except Exception as exc:          # noqa: BLE001 - the caller's code; a guard that cannot say yes said no
                _log.warning("sulci observe: accept() raised %s; recorded as a refusal", type(exc).__name__)
                self.guard_ok = False
        return False

    def label(self, serving_threshold: float) -> str:
        if self.similarity is None or self.similarity < float(serving_threshold) - _EPS:
            return WOULD_MISS
        if self.guard_ok is False:
            return WOULD_REFUSE
        return WOULD_HIT


_SCHEMA = """
CREATE TABLE IF NOT EXISTS observe_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observe_run (
    run_id TEXT PRIMARY KEY, started_at REAL NOT NULL, sulci_version TEXT, backend TEXT, embedding_model TEXT,
    serving_threshold REAL, context_window INTEGER, context_threshold REAL, personalized INTEGER, ttl_seconds INTEGER);
CREATE TABLE IF NOT EXISTS observe_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, at REAL NOT NULL, kind TEXT NOT NULL,
    label TEXT NOT NULL, reason TEXT, similarity REAL, guard_ok INTEGER, serving_threshold REAL, context_depth INTEGER,
    query_hash TEXT, scope_hash TEXT, latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER);
CREATE INDEX IF NOT EXISTS observe_events_kind ON observe_events (kind, label);
CREATE TABLE IF NOT EXISTS observe_pending (
    pending_key TEXT PRIMARY KEY, event_id INTEGER NOT NULL, scope_hash TEXT, similarity REAL, would_hit INTEGER,
    cached TEXT, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS observe_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, at REAL NOT NULL, scope_hash TEXT,
    similarity REAL NOT NULL, question TEXT NOT NULL, cached TEXT NOT NULL, fresh TEXT NOT NULL,
    rating TEXT, rated_by TEXT, rated_at REAL);
"""


def _open(path: str) -> sqlite3.Connection:
    """Open (creating, 0600) an observe file. WAL, so several processes on one host can write it."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:                                  # not ours to change (another user's file): leave it as it is
        pass
    conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")   # WAL keeps it consistent; no fsync on every observed call
    except sqlite3.DatabaseError:
        pass
    conn.executescript(_SCHEMA)
    return conn


class Recorder:
    """Writes one observe file. Built by ``Cache(observe=True)``; not public API beyond what the docs describe."""

    def __init__(self, path: str, *, run: Optional[dict] = None, rng: Optional[random.Random] = None,
                 clock: Callable[[], float] = time.time):
        self.path = os.path.abspath(os.path.expanduser(path))
        self._rng = rng or random.SystemRandom()
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = _open(self.path)
        self.run_id = uuid.uuid4().hex
        run = dict(run or {})
        with self._lock:
            self._conn.execute(
                "INSERT INTO observe_run (run_id, started_at, sulci_version, backend, embedding_model, "
                "serving_threshold, context_window, context_threshold, personalized, ttl_seconds) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, self._clock(), run.get("sulci_version"), run.get("backend"),
                 run.get("embedding_model"), run.get("serving_threshold"), run.get("context_window"),
                 run.get("context_threshold"), 1 if run.get("personalized") else 0, run.get("ttl_seconds")))
        self.counts = Counter()

    # ── recording (fail-soft) ─────────────────────────────────────────────────────────────────────────────────
    def _soft(self, what: str, fn, *args, **kwargs):
        try:
            with self._lock:
                return fn(*args, **kwargs)
        except Exception as exc:                 # noqa: BLE001 - recording never fails the caller's call
            _log.warning("sulci observe: could not record the %s (%s)", what, type(exc).__name__)
            return None

    def record_lookup(self, *, pending_key: str, query: str, scope_hash: Optional[str], cand: Candidate,
                      serving_threshold: float, depth: int, latency_ms: float) -> str:
        """Record one observed lookup; return its label."""
        label = cand.label(serving_threshold)
        self.counts[label] += 1

        def go():
            now = self._clock()
            cur = self._conn.execute(
                "INSERT INTO observe_events (run_id, at, kind, label, similarity, guard_ok, serving_threshold, "
                "context_depth, query_hash, scope_hash, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, now, "get", label, cand.similarity,
                 None if cand.guard_ok is None else int(cand.guard_ok), float(serving_threshold), int(depth or 0),
                 _hash(query), scope_hash, float(latency_ms)))
            would_hit = label == WOULD_HIT
            keep_text = would_hit and cand.similarity is not None and cand.similarity >= PAIR_MIN_SIMILARITY - _EPS
            self._conn.execute("DELETE FROM observe_pending WHERE expires_at <= ?", (now,))
            self._conn.execute(
                "INSERT OR REPLACE INTO observe_pending (pending_key, event_id, scope_hash, similarity, would_hit, "
                "cached, expires_at) VALUES (?,?,?,?,?,?,?)",
                (pending_key, cur.lastrowid, scope_hash, cand.similarity, int(would_hit),
                 cand.response if keep_text else None, now + PENDING_TTL_S))
        self._soft("lookup", go)
        return label

    def take_pending(self, pending_key: str) -> Optional[dict]:
        """The lookup waiting for this ``set``, if any, removed from the file."""
        def go():
            now = self._clock()
            row = self._conn.execute(
                "SELECT event_id, scope_hash, similarity, would_hit, cached FROM observe_pending "
                "WHERE pending_key = ? AND expires_at > ?", (pending_key, now)).fetchone()
            self._conn.execute("DELETE FROM observe_pending WHERE pending_key = ?", (pending_key,))
            if row is None:
                return None
            return {"event_id": row[0], "scope_hash": row[1], "similarity": row[2], "would_hit": bool(row[3]),
                    "cached": row[4]}
        return self._soft("pending lookup", go)

    def record_set(self, *, pending: Optional[dict], query: str, scope_hash: Optional[str], label: str,
                   reason: Optional[str], fresh: Optional[str], usage: Any) -> None:
        self.counts["set_" + label] += 1
        tokens = _token_counts(usage)

        def go():
            now = self._clock()
            self._conn.execute(
                "INSERT INTO observe_events (run_id, at, kind, label, reason, similarity, query_hash, scope_hash, "
                "input_tokens, output_tokens) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, now, "set", label, reason, pending and pending.get("similarity"), _hash(query),
                 scope_hash, tokens[0], tokens[1]))
            if pending and tokens[0] is not None:
                # The fresh call's counts are what a hit on this lookup would have saved.
                self._conn.execute("UPDATE observe_events SET input_tokens = ?, output_tokens = ? WHERE id = ?",
                                   (tokens[0], tokens[1], pending["event_id"]))
            if (pending and pending.get("would_hit") and pending.get("cached") and isinstance(fresh, str)
                    and (pending.get("similarity") or 0.0) >= PAIR_MIN_SIMILARITY - _EPS):
                self._save_pair(now, scope_hash, float(pending["similarity"]), query, pending["cached"], fresh)
        self._soft("set", go)

    def record_skip(self, reason: str) -> None:
        """A call that never reached a lookup (sulci-proxy: a stream)."""
        self.counts["skipped"] += 1
        self._soft("skipped call", lambda: self._conn.execute(
            "INSERT INTO observe_events (run_id, at, kind, label, reason) VALUES (?,?,?,?,?)",
            (self.run_id, self._clock(), "skip", SKIPPED, reason)))

    def _save_pair(self, now, scope_hash, similarity, question, cached, fresh) -> bool:
        """Reservoir sampling over the file's pairable would-be hits; rated pairs are never replaced."""
        row = self._conn.execute("SELECT value FROM observe_meta WHERE key = 'pairs_offered'").fetchone()
        offered = (int(row[0]) if row else 0) + 1
        self._conn.execute("INSERT OR REPLACE INTO observe_meta (key, value) VALUES ('pairs_offered', ?)",
                           (str(offered),))
        held = self._conn.execute("SELECT count(*) FROM observe_pairs").fetchone()[0]
        if held >= PAIR_CAP:
            if self._rng.randrange(offered) >= PAIR_CAP:
                return False
            victim = self._conn.execute(
                "SELECT id FROM observe_pairs WHERE rating IS NULL ORDER BY id").fetchall()
            if not victim:
                return False                     # every kept pair is rated: keep those
            self._conn.execute("DELETE FROM observe_pairs WHERE id = ?",
                               (victim[self._rng.randrange(len(victim))][0],))
        self._conn.execute(
            "INSERT INTO observe_pairs (run_id, at, scope_hash, similarity, question, cached, fresh) "
            "VALUES (?,?,?,?,?,?,?)", (self.run_id, now, scope_hash, similarity, question, cached, fresh))
        self.counts["pairs_kept"] += 1
        return True

    # ── erasure (raises) ──────────────────────────────────────────────────────────────────────────────────────
    def forget(self, scope_hash: Any = None, *, everything: bool = False) -> int:
        """Delete stored text: every pair and pending answer (``everything``), or one scope's. Returns the number of
        pairs deleted. Raises on failure: an erasure that cannot be confirmed is an error."""
        with self._lock:
            return forget_file_conn(self._conn, scope_hash, everything=everything)

    def stats(self) -> dict:
        c = self.counts
        return {"path": self.path, "lookups": sum(c[x] for x in LOOKUP_LABELS), "would_hit": c[WOULD_HIT],
                "would_refuse": c[WOULD_REFUSE], "pairs_kept": c["pairs_kept"]}

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:                        # noqa: BLE001
            pass


def forget_file_conn(conn: sqlite3.Connection, scope_hash: Any = None, *, everything: bool = False) -> int:
    if everything:
        n = conn.execute("DELETE FROM observe_pairs").rowcount
        conn.execute("DELETE FROM observe_pending")
    else:
        if scope_hash is None:
            n = conn.execute("DELETE FROM observe_pairs WHERE scope_hash IS NULL").rowcount
            conn.execute("DELETE FROM observe_pending WHERE scope_hash IS NULL")
        else:
            n = conn.execute("DELETE FROM observe_pairs WHERE scope_hash = ?", (scope_hash,)).rowcount
            conn.execute("DELETE FROM observe_pending WHERE scope_hash = ?", (scope_hash,))
    return int(n or 0)


def _token_counts(usage: Any) -> tuple:
    if usage is None:
        return (None, None)
    try:
        from sulci._usage import checked_usage
        u = checked_usage(usage)
    except Exception:                            # noqa: BLE001 - counts are optional
        return (None, None)
    if u is None:
        return (None, None)
    if isinstance(u, dict):
        return (u.get("input_tokens"), u.get("output_tokens"))
    return (getattr(u, "input_tokens", None), getattr(u, "output_tokens", None))


# ── the measured report ───────────────────────────────────────────────────────────────────────────────────────

def wilson_upper(bad: int, n: int, z: float = 1.959964) -> Optional[float]:
    """The Wilson score interval's upper bound for bad/n at 95%; None when n is 0."""
    if n <= 0:
        return None
    p = bad / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / denom)


def _read(paths: Sequence[str]):
    events: List[dict] = []
    pairs: List[dict] = []
    runs: List[dict] = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        conn = sqlite3.connect("file:" + os.path.abspath(path) + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            runs += [dict(r) for r in conn.execute("SELECT * FROM observe_run ORDER BY started_at")]
            events += [dict(r) for r in conn.execute("SELECT * FROM observe_events ORDER BY id")]
            pairs += [dict(r) for r in conn.execute("SELECT similarity, rating FROM observe_pairs")]
        finally:
            conn.close()
    return runs, events, pairs


def _band(s: float) -> str:
    edges = (0.80, 0.85, 0.90, 0.95, 1.0000001)
    for lo, hi in zip(edges, edges[1:]):
        if lo - _EPS <= s < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return "below 0.80"


def measure(paths: Sequence[str]) -> dict:
    """The measured numbers from one or more observe files: no text, no hashes."""
    runs, events, pairs = _read(paths)
    lookups = [e for e in events if e["kind"] == "get"]
    skips = [e for e in events if e["kind"] == "skip"]
    sets = [e for e in events if e["kind"] == "set"]
    calls = len(lookups) + len(skips)
    ats = [e["at"] for e in events]

    def would_hit_at(e, t):
        return e["similarity"] is not None and e["similarity"] >= t - _EPS and e["guard_ok"] != 0

    reasons = Counter()
    for e in skips:
        reasons[("skipped", e["reason"] or "unknown")] += 1
    for e in lookups:
        if e["label"] == WOULD_REFUSE:
            reasons[("would_refuse", "your accept() said no")] += 1
    for e in sets:
        if e["label"] == NOT_STORED:
            reasons[("not_stored", e["reason"] or "unknown")] += 1

    rated = [(p["similarity"], p["rating"]) for p in pairs if p["rating"]]
    sweep = []
    for t in THRESHOLDS:
        hits = [e for e in lookups if would_hit_at(e, t)]
        with_tokens = [e for e in hits if e["input_tokens"] is not None]
        r = [rating for s, rating in rated if s >= t - _EPS]
        judged = [x for x in r if x in ("acceptable", "not_acceptable")]
        bad = sum(1 for x in judged if x == "not_acceptable")
        sweep.append({
            "threshold": t,
            "would_hits": len(hits),
            "hit_rate_of_calls": len(hits) / calls if calls else None,
            "hit_rate_of_cacheable": len(hits) / len(lookups) if lookups else None,
            "input_tokens_saved": sum(e["input_tokens"] or 0 for e in with_tokens),
            "output_tokens_saved": sum(e["output_tokens"] or 0 for e in with_tokens),
            "token_coverage": len(with_tokens) / len(hits) if hits else None,
            "rated": len(judged),
            "not_acceptable": bad,
            "cant_tell": sum(1 for x in r if x == "cant_tell"),
            "wrong_hit_upper_95": wilson_upper(bad, len(judged)),
        })
    bands = Counter(_band(p["similarity"]) for p in pairs)
    bands_rated = Counter(_band(s) for s, x in rated if x in ("acceptable", "not_acceptable"))
    bands_bad = Counter(_band(s) for s, x in rated if x == "not_acceptable")
    serving = sorted({r["serving_threshold"] for r in runs if r["serving_threshold"] is not None})
    return {
        "format": SUMMARY_FORMAT,
        "sulci_version": _version(),
        "runs": [{k: r[k] for k in ("started_at", "sulci_version", "backend", "embedding_model", "serving_threshold",
                                    "context_window", "context_threshold", "personalized", "ttl_seconds")}
                 for r in runs],
        "window": {"first": min(ats) if ats else None, "last": max(ats) if ats else None},
        "calls": calls,
        "cacheable": len(lookups),
        "by_label": dict(Counter(e["label"] for e in lookups)),
        "serving_thresholds": serving,
        "not_cacheable_by_reason": [{"label": k[0], "reason": k[1], "count": n} for k, n in reasons.most_common()],
        "sweep": sweep,
        "bands": [{"band": b, "pairs": bands[b], "rated": bands_rated[b], "not_acceptable": bands_bad[b]}
                  for b in sorted(bands)],
        "pairs": {"kept": len(pairs), "rated": sum(1 for _, x in rated if x in ("acceptable", "not_acceptable")),
                  "cant_tell": sum(1 for _, x in rated if x == "cant_tell")},
        "minimums": {"cacheable": MIN_CACHEABLE, "rated": MIN_RATED},
        "enough": len(lookups) >= MIN_CACHEABLE and
                  sum(1 for _, x in rated if x in ("acceptable", "not_acceptable")) >= MIN_RATED,
    }


def _version() -> Optional[str]:
    try:
        import sulci
        return getattr(sulci, "__version__", None)
    except Exception:                            # noqa: BLE001
        return None


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def render(m: dict) -> str:
    """The measured report as Markdown."""
    def when(t):
        return "—" if t is None else time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(t))
    out = ["# Sulci observe mode: measured report", ""]
    out.append(f"Window: {when(m['window']['first'])} to {when(m['window']['last'])}. "
               f"Calls: {m['calls']}; looked up (cacheable): {m['cacheable']}.")
    for r in m["runs"]:
        out.append(f"- Run: backend {r['backend']}, embedding {r['embedding_model']}, serving threshold "
                   f"{r['serving_threshold']}, context window {r['context_window']}, entries live "
                   f"{'forever' if not r['ttl_seconds'] else str(r['ttl_seconds']) + ' s'}, sulci {r['sulci_version']}.")
    out += ["", "Nothing was served from the cache. Each figure counts what the cache would have served. Similarity "
            "comes from your own embedder and backend, so these numbers describe this setup, not another.", ""]
    out += ["## Hit rate by threshold", "",
            "| Threshold | Would-be hits | Of all calls | Of cacheable calls | Input tokens saved | Output tokens saved "
            "| Token counts known | Rated | Not acceptable | Can't tell | Wrong-hit upper bound (95%) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in m["sweep"]:
        out.append(f"| {s['threshold']:.2f} | {s['would_hits']} | {_pct(s['hit_rate_of_calls'])} | "
                   f"{_pct(s['hit_rate_of_cacheable'])} | {s['input_tokens_saved']} | {s['output_tokens_saved']} | "
                   f"{_pct(s['token_coverage'])} | {s['rated']} | {s['not_acceptable']} | {s['cant_tell']} | "
                   f"{_pct(s['wrong_hit_upper_95'])} |")
    out += ["", "Requests saved at a threshold = its would-be hits. Tokens saved count only would-be hits whose "
            "following set() carried token counts (`usage=` or `sulci.with_usage`); the column before says what "
            "share did. \"Can't tell\" is never counted as acceptable.", ""]
    out += ["## Not cacheable, by reason", ""]
    if m["not_cacheable_by_reason"]:
        out += [f"- {r['label']}: {r['reason']}: {r['count']}" for r in m["not_cacheable_by_reason"]]
    else:
        out.append("- none recorded")
    out += ["", "## Review pairs", "",
            f"Kept: {m['pairs']['kept']} (at most {PAIR_CAP}, a random sample of would-be hits at "
            f"{PAIR_MIN_SIMILARITY:.2f} or above). Rated: {m['pairs']['rated']}; can't tell: {m['pairs']['cant_tell']}.",
            ""]
    if m["bands"]:
        out += ["| Similarity band | Pairs | Rated | Not acceptable |", "|---|---|---|---|"]
        out += [f"| {b['band']} | {b['pairs']} | {b['rated']} | {b['not_acceptable']} |" for b in m["bands"]]
        out.append("")
    if not m["enough"]:
        out += [f"**Not enough to tell yet:** a run needs at least {MIN_CACHEABLE} cacheable calls and "
                f"{MIN_RATED} rated pairs (this one: {m['cacheable']} and {m['pairs']['rated']}).", ""]
    return "\n".join(out)


# ── rating ────────────────────────────────────────────────────────────────────────────────────────────────────

_KEYS = {"a": "acceptable", "n": "not_acceptable", "c": "cant_tell"}


def rate(path: str, *, reviewer: Optional[str] = None, read: Callable[[str], str] = input,
         write: Callable[[str], None] = print, rng: Optional[random.Random] = None) -> int:
    """Rate unrated pairs in the terminal, in random order, without showing the similarity. Returns how many."""
    conn = _open(path)
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM observe_pairs WHERE rating IS NULL")]
        (rng or random.Random()).shuffle(ids)
        write(f"{len(ids)} pair(s) to rate. For each: would you send the CACHED answer to this asker, unchanged? "
              "a = acceptable, n = not acceptable, c = can't tell, s = skip, q = quit.")
        done = 0
        for i, pid in enumerate(ids, 1):
            q, cached, fresh = conn.execute(
                "SELECT question, cached, fresh FROM observe_pairs WHERE id = ?", (pid,)).fetchone()
            write(f"\n── {i}/{len(ids)} ──\nQUESTION:\n{q}\n\nCACHED ANSWER (what the cache would have served):\n"
                  f"{cached}\n\nFRESH ANSWER (what your LLM said, for reference):\n{fresh}\n")
            while True:
                k = (read("[a/n/c/s/q] ") or "").strip().lower()[:1]
                if k in _KEYS or k in ("s", "q"):
                    break
            if k == "q":
                break
            if k == "s":
                continue
            conn.execute("UPDATE observe_pairs SET rating = ?, rated_by = ?, rated_at = ? WHERE id = ?",
                         (_KEYS[k], reviewer, time.time(), pid))
            done += 1
        write(f"Rated {done}.")
        return done
    finally:
        conn.close()


def export_pairs(path: str, out: str, *, rng: Optional[random.Random] = None) -> int:
    """Write the pairs to a CSV for reviewers (random order, no similarity). Returns how many."""
    conn = _open(path)
    try:
        rows = conn.execute("SELECT id, question, cached, fresh, rating FROM observe_pairs").fetchall()
    finally:
        conn.close()
    rows = list(rows)
    (rng or random.Random()).shuffle(rows)
    fd = os.open(out, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "question", "cached_answer", "fresh_answer", "rating"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4] or ""])
    return len(rows)


def import_ratings(path: str, src: str, *, reviewer: Optional[str] = None) -> int:
    """Read ratings back from an exported CSV (``rating`` = acceptable / not_acceptable / cant_tell, or blank)."""
    conn = _open(path)
    n = 0
    try:
        with open(src, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rating = (row.get("rating") or "").strip().lower()
                if not rating:
                    continue
                rating = _KEYS.get(rating, rating)
                if rating not in RATINGS:
                    raise ValueError(f"pair {row.get('pair_id')}: rating {rating!r} is not one of {', '.join(RATINGS)}")
                n += conn.execute("UPDATE observe_pairs SET rating = ?, rated_by = ?, rated_at = ? WHERE id = ?",
                                  (rating, reviewer, time.time(), int(row["pair_id"]))).rowcount
    finally:
        conn.close()
    return n


def forget_file(path: str) -> int:
    """Delete every pair and pending answer from an observe file. The measured numbers stay."""
    conn = _open(path)
    try:
        n = forget_file_conn(conn, everything=True)
        conn.execute("VACUUM")
        return n
    finally:
        conn.close()


# ── command line ──────────────────────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="sulci-observe",
                                description="Read, rate and summarise a Sulci observe-mode file (Cache(observe=True)).")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("status", "counts and pairs"), ("report", "the measured report (Markdown)")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("files", nargs="+")
    s = sub.add_parser("summary", help="the measured numbers as JSON: no text, no hashes")
    s.add_argument("files", nargs="+")
    s.add_argument("--out", required=True)
    s = sub.add_parser("rate", help="rate review pairs in the terminal (blind)")
    s.add_argument("file")
    s.add_argument("--reviewer", default=None)
    s = sub.add_parser("export-pairs", help="write the pairs to a CSV for reviewers")
    s.add_argument("file")
    s.add_argument("--csv", required=True)
    s = sub.add_parser("import-ratings", help="read ratings back from that CSV")
    s.add_argument("file")
    s.add_argument("--csv", required=True)
    s.add_argument("--reviewer", default=None)
    s = sub.add_parser("forget", help="delete every stored question and answer (the numbers stay)")
    s.add_argument("file")
    a = p.parse_args(list(argv) if argv is not None else None)

    if a.cmd == "status":
        m = measure(a.files)
        print(f"calls {m['calls']}, cacheable {m['cacheable']}, by label {m['by_label']}, pairs kept "
              f"{m['pairs']['kept']}, rated {m['pairs']['rated']}, can't tell {m['pairs']['cant_tell']}")
    elif a.cmd == "report":
        print(render(measure(a.files)))
    elif a.cmd == "summary":
        m = measure(a.files)
        fd = os.open(a.out, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, sort_keys=True)
        print(f"wrote {a.out}: {m['calls']} calls, {m['cacheable']} cacheable, {m['pairs']['rated']} rated; no text")
    elif a.cmd == "rate":
        rate(a.file, reviewer=a.reviewer)
    elif a.cmd == "export-pairs":
        print(f"wrote {export_pairs(a.file, a.csv)} pair(s) to {a.csv} (it holds the text: keep it where the "
              "observe file is)")
    elif a.cmd == "import-ratings":
        print(f"read {import_ratings(a.file, a.csv, reviewer=a.reviewer)} rating(s)")
    elif a.cmd == "forget":
        print(f"deleted {forget_file(a.file)} pair(s) and every pending answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
