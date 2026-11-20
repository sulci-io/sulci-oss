# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
S18 (oss-0073): exact repeats in observe mode's review.

An exact repeat is a would-be hit at similarity >= 0.9995: the same question asked again. In the 1.5.0rc1 live check,
6 of the 10 pairs offered for rating were exact repeats. They are still rated and still count in the bound (an answer
about the time or a person can be wrong on a repeat too), but `rate` offers them last and tags them, the CSV marks
them, the report and the summary count them on their own, and they hold at most a quarter of the kept pairs. The
managed platform draws the same line (sulci-platform 0561) and its run.json import accepts the new summary fields.
"""
from __future__ import annotations

import csv
import json
import random

import pytest

from sulci import observe


def _recorder(tmp_path, seed=0, name="r.db"):
    return observe.Recorder(str(tmp_path / name), rng=random.Random(seed))


def _keep(rec, sim, n):
    with rec._lock:
        return rec._save_pair(0, "s", sim, f"q{n}", f"c{n}", f"f{n}")


def _held(rec):
    return [r[0] for r in rec._conn.execute("SELECT similarity FROM observe_pairs ORDER BY id")]


@pytest.mark.parametrize("s,want", [(1.0, True), (0.99999994, True), (0.9995, True), (0.9994, False),
                                    (0.9768, False), (0.85, False), (None, False), ("x", False)])
def test_the_line(s, want):
    assert observe.is_exact_repeat(s) is want


def test_the_constants_match_the_platform():
    assert observe.EXACT_REPEAT == 0.9995
    assert observe.REPEAT_CAP == observe.PAIR_CAP // 4 == 100


def test_repeats_never_take_more_than_their_share(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "PAIR_CAP", 8)
    monkeypatch.setattr(observe, "REPEAT_CAP", 2)
    rec = _recorder(tmp_path, seed=7)
    for n in range(30):
        _keep(rec, 1.0, n)
    assert len(_held(rec)) == 2
    for n in range(30, 36):
        assert _keep(rec, 0.9, n), "rewordings still fill the rest"
    held = _held(rec)
    assert len(held) == 8 and sum(observe.is_exact_repeat(s) for s in held) == 2


def test_the_repeat_sample_is_random_not_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "REPEAT_CAP", 2)
    rec = _recorder(tmp_path, seed=11)
    for n in range(40):
        _keep(rec, 1.0, n)
    qs = sorted(r[0] for r in rec._conn.execute("SELECT question FROM observe_pairs"))
    assert len(qs) == 2 and qs != ["q0", "q1"]


def test_a_rewording_takes_an_unrated_repeats_place_when_full(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "PAIR_CAP", 8)
    monkeypatch.setattr(observe, "REPEAT_CAP", 2)
    rec = _recorder(tmp_path)
    for n in range(2):
        _keep(rec, 1.0, n)
    for n in range(2, 8):
        _keep(rec, 0.9, n)
    rec._rng.randrange = lambda n: n - 1                 # the ordinary reservoir would say no
    assert _keep(rec, 0.88, 99) is True
    held = _held(rec)
    assert len(held) == 8 and sum(observe.is_exact_repeat(s) for s in held) == 1


def test_rated_repeats_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "REPEAT_CAP", 2)
    rec = _recorder(tmp_path)
    for n in range(2):
        _keep(rec, 1.0, n)
    rec._conn.execute("UPDATE observe_pairs SET rating = 'acceptable'")
    rec._rng.randrange = lambda n: 0                     # would always replace, if it could
    assert _keep(rec, 1.0, 9) is False
    assert len(_held(rec)) == 2


def _lookup(rec, i, sim):
    c = observe.Candidate()
    c.accept("x", sim)
    rec.record_lookup(pending_key=str(i), query=f"q{i}", scope_hash="s", cand=c, serving_threshold=0.85, depth=0,
                      latency_ms=1)


def test_the_report_and_the_summary_count_them(tmp_path):
    rec = _recorder(tmp_path)
    for i, sim in enumerate((1.0, 0.9999999, 0.91, 0.86, 0.83, 0.5)):
        _lookup(rec, i, sim)
    for n, (sim, rating) in enumerate(((1.0, "acceptable"), (0.91, "acceptable"), (0.86, None))):
        _keep(rec, sim, n)
        if rating:
            rec._conn.execute("UPDATE observe_pairs SET rating = ? WHERE question = ?", (rating, f"q{n}"))
    m = observe.measure([rec.path])
    by = {s["threshold"]: s for s in m["sweep"]}
    assert by[0.85]["would_hits"] == 4 and by[0.85]["exact_repeats"] == 2
    assert by[0.95]["would_hits"] == 2 and by[0.95]["exact_repeats"] == 2
    assert by[0.85]["rated"] == 2 and by[0.85]["rated_exact_repeats"] == 1
    assert by[0.85]["wrong_hit_upper_95"] is not None, "the bound still counts every rated pair"
    assert m["pairs"]["exact_repeats"] == 1
    md = observe.render(m)
    assert "Of which exact repeats" in md and "Exact repeats among them: 1." in md
    assert "| 0.85 | 4 | 2 |" in md
    assert "the same question asked again" in md
    json.dumps(m)                                         # the summary stays plain JSON


def test_rate_offers_rewordings_first_and_tags_repeats(tmp_path):
    rec = _recorder(tmp_path)
    for n, sim in enumerate((1.0, 0.9999, 0.88, 0.91, 0.84)):
        _keep(rec, sim, n)
    path = rec.path
    rec.close()
    shown = []
    answers = iter(["a"] * 5)
    assert observe.rate(path, read=lambda p: next(answers), write=shown.append, rng=random.Random(1)) == 5
    blocks = [s for s in shown if s.startswith("\n── ")]
    assert len(blocks) == 5
    tagged = ["exact repeat" in b.split("\n")[1] for b in blocks]
    assert tagged == [False, False, False, True, True]
    assert not any("0.9999" in b or "similarity" in b.lower() for b in blocks), "the similarity is never shown"


def test_the_csv_marks_them_and_puts_them_last(tmp_path):
    rec = _recorder(tmp_path)
    for n, sim in enumerate((1.0, 0.88, 0.9999, 0.91)):
        _keep(rec, sim, n)
    path = rec.path
    rec.close()
    out = tmp_path / "pairs.csv"
    assert observe.export_pairs(path, str(out), rng=random.Random(2)) == 4
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["exact_repeat"] for r in rows] == ["", "", "yes", "yes"]
    rows[0]["rating"] = "a"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert observe.import_ratings(path, str(out)) == 1, "the new column does not get in the way of reading back"
