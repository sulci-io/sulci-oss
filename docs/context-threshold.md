# Context threshold

`sulci.Cache` takes two similarity thresholds, not one.

| parameter | applies to | default |
|---|---|---|
| `threshold` | every lookup that did not blend (`context_depth == 0`) | `0.85` |
| `context_threshold` | lookups that actually blended (`context_depth > 0`) | unset — falls back to `threshold` |

## Why two

The blended lookup vector is `query_weight` × the query plus `1 - query_weight`
spread over the decayed history — 70/30 by default. Its raw cosine similarity to
any stored entry is structurally lower than an exact-match lookup's, because the
history component points somewhere the stored entry does not. A threshold
calibrated against exact-match similarity is therefore *tighter* on the blended
path than it looks.

`benchmark/run.py` has calibrated its own `--context-threshold` separately since
it was written, and says why in a comment above the assignment. `sulci.Cache`
did not until 2026-08-12.

## What the discriminator is, and what it is not

The threshold switches on **`context_depth > 0`** — whether this particular
lookup actually blended — not on `context_window > 0`.

That distinction is the whole point. A `Cache(context_window=6)` serves plenty of
lookups with no `session_id`, or with a session whose window is still empty.
`_context_vec` returns the raw query embedding for those, and they are
exact-match lookups no matter what the constructor said. Keying off
`context_window` would apply the blended threshold to all of them, which is the
defect this parameter exists to fix rather than a version of the fix.

## Precedence

1. An explicit `threshold=` on the call. A caller who names a number gets it.
2. `context_threshold`, when the lookup blended and the value is set.
3. `threshold`.

Whichever wins is the number used for the backend search, the telemetry
payload, **and** the emitted `CacheEvent`. That is deliberate: `Cache.get`'s
docstring records what happened in v0.8.0 when the event and the answer were
decided by different numbers — similarity in `[instance, effective)` emitted a
hit event while returning a miss, and billing counted hits that never happened.

## There is no recommended value, and that is itself a finding

A six-point threshold sweep on the benchmark corpus produced a clean-looking
optimum where every column improved at once. One extra held-out draw reversed
the direction of the false-hit column at every threshold.

The arithmetic explains it. The corpus holds **25 should-miss rows across 5
domains**, so the false-hit rate can only move in 4pp quanta, and the seed
decides which five sessions are held out. The error bar exceeds the effect.

Four committed draws under `benchmark/results/minilm/seed-{1,2,3,42}` say the
same thing from a different angle: at the default context threshold, context
*raises* false-hit on every draw (by 24 to 56pp), while the resolution-accuracy
delta wanders across `+3, +4, −1, −1`.

**The supported claim is that this corpus cannot discriminate. That is not the
same as the feature not working**, and only the first of those is supported.

So no default ships and none is suggested here. Sweep `context_threshold`
against your own held-out follow-ups — ones where the answer genuinely is not
recoverable from the follow-up's own text.

## Remote transport

`context_threshold` is a self-hosted parameter. On the cloud transport the
gateway performs the blending and applies its own threshold, so a value set here
has no effect; `Cache.get` warns once rather than accepting it silently.
