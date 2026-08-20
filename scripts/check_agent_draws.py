#!/usr/bin/env python3
"""Verify the MiniLM agent draws that the public register publishes from.

WHAT THIS CHECKS, AND WHY IT IS NOT check_figures_fresh.py
----------------------------------------------------------
`docs/marketing/rebuild/check_figures_fresh.py` (sulci-platform) re-hashes the
12 recorded inputs and answers ONE question: are the bytes that produced
figures.json the bytes on disk now. It says so itself and disclaims the rest.

It cannot answer: does a run at seed N still PRODUCE seed-N's numbers. A hash
is identical whether the file is right or wrong, so the 2026-08-19 seed defect
(#145) hashed as consistently as anything else while `run.py --agent` with no
`--seed` emitted cold 27.0% / warm 99.4% -- the pair retired in fc701fb -- and
the documented publishable-figures command had been doing so since 2026-08-11.

This script closes that. Three modes, cheapest first:

  --aggregate-only   No run. Recomputes the register's published aggregates
                     from the four committed draws and compares them to the
                     values CLAIMS.md publishes. Milliseconds. Safe in CI and
                     in `make checkin`.

  --seed N           Runs the agent bench at seed N and compares the output to
                     benchmark/results/minilm/seed-N/agent_summary.json.
                     EXACT: no tolerance. A same-seed run is deterministic, so
                     any difference is a real change, not noise.

  --default          Runs the agent bench with NO --seed and compares to the
                     seed-42 draw. This is the #145 regression test: it fails
                     the moment run.py's fallback seed stops being 42.

⚠️ WHAT A PASS HERE DOES NOT MEAN
It does not mean the figures are RIGHT. It means a run at a named seed still
reproduces the draw the register was built from. Whether that draw measures
what it claims is a question for benchmark/README.md and CLAIMS.md, and no
comparison of a run against its own recorded output can answer it.

It also says nothing about the CONTEXT or STATELESS blocks in those same seed
directories -- only agent_summary.json is compared. Extending it is
mechanical; not doing so silently is the 08-10 lesson.

USAGE
    python3 scripts/check_agent_draws.py --aggregate-only
    python3 scripts/check_agent_draws.py --default
    python3 scripts/check_agent_draws.py --seed 3

Exit codes: 0 pass · 1 mismatch · 2 cannot tell (missing draws, run failed).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRAWS_ROOT = REPO_ROOT / "benchmark" / "results" / "minilm"
SEEDS = (1, 2, 3, 42)

# The seed run.py falls back to when --seed is absent. Pinned here as well as
# in run.py so that a change to one without the other is a test failure rather
# than a silent move. See #145: the fallback was 1729 while --seed's help text
# said 42, and nothing compared them.
DEFAULT_SEED = 42

# What the public register publishes, from the four draws above.
# sulci-platform/docs/marketing/CLAIMS.md -- keep in step by hand; there is no
# cross-repo consumer (that is a separate, known gap).
REGISTER = {
    "hit_rate_cold_session":  {"mean": 0.334,  "lo": 0.300,  "hi": 0.350},
    "hit_rate_warm_session":  {"mean": 0.9974, "lo": 0.9969, "hi": 0.9981},
    "aggregate_hit_rate":     {"mean": 0.950,  "lo": 0.9499, "hi": 0.9505},
    "misses_per_session_p50": {"mean": 2,      "lo": 2,      "hi": 2},
}
# The stateless figures the register publishes, also from the four draws.
# CLAIMS.md:249-251. "Correctly declined" is 1 - false_hit_rate.
REGISTER_STATELESS = {
    "recall":         {"mean": 0.999,  "lo": 0.9985, "hi": 0.9995},
    "false_hit_rate": {"mean": 0.0106, "lo": 0.0062, "hi": 0.0185},
    "precision":      {"mean": 0.9377, "lo": 0.9213, "hi": 0.9469},
}

# Rounding slack for the published mean only. The draws themselves are
# compared exactly.
MEAN_TOL = 0.0006

#: Excluded from the exact stateless comparison. baseline.json's
#: tolerance_notes already says it: "Latency varies by machine; not part of
#: the regression check." Everything else in summary.json IS exact -- a
#: same-seed, same-flags run reproduces it to the integer.
#:
#: This was doubted on 2026-08-19 and the doubt was wrong. Three runs
#: appeared to give FP 5.61 / 5.61 / 5.62% and $20.13 / $20.13 / $20.10,
#: which was read as non-determinism. They were different COMMANDS: the
#: first two carried --context. The --agent-only run reproduced seed-42's
#: summary.json exactly -- cache_hits 4020, false_positives 226,
#: fp_rate 0.0562, saved_cost 20.10. See #147.
LATENCY_KEYS = {
    "latency_hit_p50_ms", "latency_hit_p95_ms",
    "latency_miss_p50_ms", "latency_miss_p95_ms",
}

BENCH_ARGV = ["--use-sulci", "--fresh", "--no-sweep", "--agent",
              "--threshold", "0.85"]


def load_draw(seed: int, name: str = "agent_summary.json") -> dict | None:
    p = DRAWS_ROOT / f"seed-{seed}" / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_all_draws() -> dict[int, dict]:
    draws = {}
    missing = []
    for s in SEEDS:
        d = load_draw(s)
        if d is None:
            missing.append(f"seed-{s}")
        else:
            draws[s] = d
    if missing:
        print(f"CANNOT TELL: missing committed draws: {', '.join(missing)}")
        print(f"  under {DRAWS_ROOT}")
        print()
        print("  Nothing was compared. This is not a pass. The draws are")
        print("  committed artifacts, not build output -- if they are absent")
        print("  the checkout is wrong, not the benchmark.")
        sys.exit(2)
    return draws


def _aggregate_table(draws: dict, register: dict, label: str) -> bool:
    ok = True
    print(f"  {label}")
    for key, want in register.items():
        vals = [draws[s][key] for s in SEEDS]
        mean = sum(vals) / len(vals)
        lo, hi = min(vals), max(vals)
        good = (abs(mean - want["mean"]) <= MEAN_TOL
                and abs(lo - want["lo"]) <= MEAN_TOL
                and abs(hi - want["hi"]) <= MEAN_TOL)
        ok &= good
        drawstr = " ".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                           for v in vals)
        print(f"    [{'OK   ' if good else 'DRIFT'}]  {key:<24} "
              f"mean {mean:.4f} [{lo:.4f}-{hi:.4f}]   draws: {drawstr}")
        if not good:
            print(f"             register says mean {want['mean']} "
                  f"[{want['lo']}-{want['hi']}]")
    return ok


def mode_aggregate_only() -> int:
    draws = load_all_draws()
    print("=" * 72)
    print(f" Register aggregates from {len(SEEDS)} committed draws "
          f"[engine: sulci-minilm]")
    print(f" draws: {DRAWS_ROOT}/seed-{{{','.join(map(str, SEEDS))}}}")
    print("=" * 72)
    print()

    ok = _aggregate_table(draws, REGISTER, "Agent workload:")

    st = {s: load_draw(s, "summary.json") for s in SEEDS}
    if all(st.values()):
        print()
        ok &= _aggregate_table(st, REGISTER_STATELESS, "Stateless:")
    else:
        print("\n  ⚠  summary.json missing from one or more draws; "
              "stateless aggregates NOT checked.")
        ok = False

    print()
    print("=" * 72)
    if ok:
        print("  check:agent-draws  OK -- the committed draws still aggregate")
        print("                     to the figures CLAIMS.md publishes.")
        print()
        print("                     This compares committed files to a table")
        print("                     transcribed by hand. It proves the draws")
        print("                     were not edited. It does NOT prove a run")
        print("                     still produces them -- use --seed N.")
    else:
        print("  MISMATCH -- the committed draws no longer aggregate to the")
        print("  register. Either the draws were regenerated without updating")
        print("  CLAIMS.md, or REGISTER above was transcribed wrong.")
        print("  Do NOT edit REGISTER to make this pass without reading")
        print("  CLAIMS.md first.")
    print("=" * 72)
    return 0 if ok else 1


def run_bench(seed: int | None, out: Path, timeout: int) -> int:
    argv = [sys.executable, "benchmark/run.py", *BENCH_ARGV,
            "--out", str(out)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    print(f"  running: {' '.join(argv[1:])}")
    print(f"  (expect several minutes: --agent still runs the stateless pass)")
    try:
        r = subprocess.run(argv, cwd=REPO_ROOT, timeout=timeout)
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"ERROR: exceeded {timeout}s", file=sys.stderr)
        return -1


def mode_compare_run(seed: int | None, timeout: int) -> int:
    """Run at `seed` (None = run.py's fallback) and compare to the draw."""
    expect_seed = DEFAULT_SEED if seed is None else seed
    want = load_draw(expect_seed)
    if want is None:
        print(f"CANNOT TELL: no committed draw at seed-{expect_seed}.")
        return 2

    label = ("run.py's DEFAULT seed" if seed is None
             else f"an explicit --seed {seed}")
    print("=" * 72)
    print(f" Comparing {label} against the committed seed-{expect_seed} draw")
    if seed is None:
        print(f" A mismatch here means run.py's fallback seed is no longer "
              f"{DEFAULT_SEED}.")
        print(" That is #145: the fallback was 1729 and produced the retired")
        print(" cold 27.0% / warm 99.4% pair from the documented command.")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run"
        rc = run_bench(seed, out, timeout)
        if rc != 0:
            print(f"CANNOT TELL: benchmark/run.py exited {rc}. "
                  f"Nothing was compared.")
            return 2
        produced = out / "agent_summary.json"
        if not produced.exists():
            print(f"CANNOT TELL: {produced} was not written.")
            return 2
        got = json.loads(produced.read_text())

        # Stateless block, from the same invocation. NOT context_summary.json:
        # the committed context file came from a --context --agent run 1.5h
        # earlier and adding --context here would overwrite summary.json with
        # the context-variant numbers. See #152.
        st_want = load_draw(expect_seed, "summary.json")
        st_path = out / "summary.json"
        st_got = json.loads(st_path.read_text()) if st_path.exists() else None

    # Compare every numeric field the draw records, EXACTLY. A same-seed run
    # is deterministic; a tolerance here would hide precisely the class of
    # defect this script exists for.
    print()
    ok = True
    for key in sorted(k for k in want if not k.startswith("_")):
        wv, gv = want[key], got.get(key)
        if isinstance(wv, dict):
            for sub in sorted(wv):
                w2, g2 = wv[sub], (gv or {}).get(sub)
                good = w2 == g2
                ok &= good
                print(f"  [{'OK   ' if good else 'DIFF '}]  "
                      f"{key}.{sub:<28} {w2} -> {g2}")
            continue
        good = wv == gv
        ok &= good
        print(f"  [{'OK   ' if good else 'DIFF '}]  {key:<34} {wv} -> {gv}")

    if st_want is not None and st_got is not None:
        print()
        print(f"  Stateless block (latency excluded -- machine-dependent):")
        for key in sorted(k for k in st_want
                          if not k.startswith("_") and k not in LATENCY_KEYS):
            wv, gv = st_want[key], st_got.get(key)
            good = wv == gv
            ok &= good
            print(f"  [{'OK   ' if good else 'DIFF '}]  {key:<34} {wv} -> {gv}")
        for key in sorted(LATENCY_KEYS & set(st_want)):
            print(f"  [ --  ]  {key:<34} {st_want[key]} -> "
                  f"{st_got.get(key)}   (excluded)")
    elif st_want is None:
        print(f"\n  ⚠  no committed summary.json at seed-{expect_seed}; "
              f"stateless block NOT checked.")
    else:
        print(f"\n  ⚠  the run wrote no summary.json; "
              f"stateless block NOT checked.")

    print()
    print("=" * 72)
    if ok:
        print(f"  check:agent-draws  OK -- reproduces seed-{expect_seed} exactly")
        print(f"                     (agent + stateless; latency excluded).")
        print()
        print("                     context_summary.json is NOT checked. It was")
        print("                     written by a different invocation -- see #152.")
    else:
        print(f"  MISMATCH against seed-{expect_seed}.")
        print()
        print("  A same-seed agent run is deterministic, so this is a real")
        print("  change: the workload, the threshold, the embedder or the")
        print("  seed plumbing moved. Find out WHICH before regenerating the")
        print("  draw -- regenerating first destroys the evidence, and the")
        print("  draw is what the public register was built from.")
    print("=" * 72)
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--aggregate-only", action="store_true",
                   help="no run; check the committed draws against CLAIMS.md")
    g.add_argument("--default", action="store_true",
                   help="run with no --seed, compare to the seed-42 draw")
    g.add_argument("--seed", type=int, metavar="N",
                   help=f"run at seed N, compare to seed-N "
                        f"(committed: {', '.join(map(str, SEEDS))})")
    p.add_argument("--timeout", type=int, default=1800,
                   help="seconds for the benchmark run (default 1800)")
    a = p.parse_args()

    if a.aggregate_only:
        return mode_aggregate_only()
    if a.default:
        return mode_compare_run(None, a.timeout)
    if a.seed not in SEEDS:
        print(f"ERROR: no committed draw at seed {a.seed}. "
              f"Committed: {', '.join(map(str, SEEDS))}", file=sys.stderr)
        return 2
    return mode_compare_run(a.seed, a.timeout)


if __name__ == "__main__":
    sys.exit(main())
