# ADR 0001 — `Cache()` auto-connects telemetry when an `api_key` is resolvable

- **Status:** Accepted
- **Date:** 2026-05-26
- **Shipped:** `sulci-cache==0.7.0`
- **Companion:** `sulci-platform` ADR-0021 (the gateway-side mirror)
- **Closes:** the dashboard-population footgun first surfaced in `sulci-platform` PR #249 close-out

---

## Context

The `Sulci_UI_Design_and_Dev_Tracker` §5.2 *Trust boundaries* spec lists
three equivalent opt-in signals for telemetry:

1. Explicit `sulci.connect()` call,
2. `SULCI_API_KEY` environment variable,
3. (implicit via (2)) any documented surface where the user hands their
   key to the SDK.

The spec calls this constraint "non-negotiable" — telemetry is opt-in
forever, never silent. Through v0.6.x, only path (1) actually enabled
the telemetry flush. Paths (2) and (3) registered the key for cache
backend auth but left `sulci._telemetry_enabled = False`. Consequence:

- `Cache(backend="sulci", api_key="sk-sulci-...")` worked for the cache
  ops (vectors flowed to `/v1/cache/*`, hit rates and cost-saved counters
  populated via the billing-worker rollup pipeline),
- BUT the four telemetry-backed dashboard panels — `TrendChart`,
  `AuditEventsTable`, `DeploymentsTable`, the Active-SDKs counter — stayed
  empty until the user separately called `sulci.connect()`.

This shipped in every quickstart, blog example, and customer onboarding
flow. The symptom (half-broken dashboard) was indistinguishable from a
product bug to the casual user. It affected every tier identically:
OSS-Connect users got an empty `ConnectedOssOverview` for deployments,
Pro/Business users got an empty `ProOverview` for trend chart and audit
feed. The latter is especially damaging — a paid customer's first
post-upgrade dashboard view was empty for the panels they were paying for.

The constraint was originally spec'd this way to keep the privacy
guarantee strict. But the way it shipped didn't match what the spec
*said* the three signals were — it was strictly stricter, requiring
signal (1) for telemetry but accepting (2) and (3) for cache auth. That
asymmetry was a bug, not an intended design.

---

## Decision

**`Cache.__init__` auto-calls `sulci.connect()`** when the following three
predicates all hold:

| Predicate | Source of truth |
|---|---|
| `self._telemetry` is `True` | the `telemetry=True` default kwarg on `Cache.__init__` |
| an api_key is resolvable | `{api_key kwarg, SULCI_API_KEY env, sulci._api_key set by a prior connect()}` |
| `sulci._telemetry_enabled` is `False` | the module-level flag — short-circuit when any prior `connect()` has run |

The resolved key is passed to `sulci.connect(api_key=resolved_key,
telemetry=True, prompt=False)`. The `prompt=False` is critical — the
auto-connect path must never block `Cache()` construction on a 15-minute
device-code timeout. Users who want the browser device-code flow continue
to call `sulci.connect(prompt=True)` explicitly *before* constructing a
`Cache`; the "already connected" short-circuit ensures their ordering
choice survives.

Auto-connect failures **never crash Cache construction**. Wrapped in
`try/except Exception`, falls through to a logged `WARNING` on the `sulci`
logger naming the exception type + message and pointing the operator at
`sulci.connect()` for retry. The cache itself stays fully functional —
only telemetry remains off.

The implementation lives at the end of `Cache.__init__`, after backend
loading. About 25 lines including comments. See `sulci/core.py` for the
canonical text.

---

## Consequences

### What gets better

**One canonical quickstart shape works for all three personas.** No
"did you remember to call `sulci.connect()`?" anywhere. The minimum
viable user code is:

```python
from sulci import Cache

cache = Cache(backend="sulci", api_key="sk-sulci-...")   # or env var
cache.get("hello")  # populates the entire dashboard
```

…and that same shape covers `Cache(backend="sqlite", api_key=...)` for the
OSS-Connect self-host-plus-telemetry persona. Backend choice is now
genuinely orthogonal to telemetry choice, which matches user mental
model.

**The privacy invariant is preserved.** Presence of api_key in any of the
three signal sources is explicit opt-in per the spec. v0.7.0 makes the
kwarg-to-Cache path consistent with the env-var path that the spec
already authorized — it doesn't add any new opt-in surface, it just
unifies behavior across surfaces the spec already accepted.

**`telemetry=False` remains a clean explicit opt-out anchor.**
Compliance-restricted environments, internal staging that shouldn't
pollute production dashboards, customers with legal restrictions on
non-essential outbound — all of them have a clear, documented escape.

**Advanced flows are unbroken.** `sulci.connect(prompt=True)` for
device-code, `sulci.connect()` at boot before workers spawn `Cache`
instances — both work identically because the `_telemetry_enabled`
short-circuit looks for "any prior connect() ran" rather than for a
specific argument shape.

### What gets worse

**One layer of "magic."** `Cache(api_key=...)` now has a side effect
beyond constructing a cache object. Documented in CHANGELOG, README, and
this ADR; signposted in source via a multi-line comment block. The
warning-on-failure path provides a recovery affordance: if the auto-call
fails, the operator sees the WARNING and knows to call `sulci.connect()`
manually.

**A subtle behavior change for users who pass api_key but didn't want
telemetry.** Pre-0.7.0 they got cache without telemetry by accident.
Post-0.7.0 the same code emits telemetry. Mitigation: the
`telemetry=False` kwarg makes the explicit opt-out trivial, and the
CHANGELOG entry calls out the change. Risk-assessed as low — most users
who pass api_key *want* the dashboard to populate; the silent-no-telemetry
state was almost always unintended.

---

## Alternatives considered

### Force telemetry on with no off-switch for paid tiers

**Rejected.** Two reasons:

1. Violates the §5.2 invariant ("Telemetry is opt-in forever") which had
   no exception for paid tiers in the original spec.
2. Multiple legitimate use cases for paid tenants to disable telemetry
   exist:
   - Enterprise on-prem deployments (per ADR §9.1 — VPC callhome for
     billing only, not full telemetry),
   - Compliance contexts requiring minimal non-essential outbound,
   - Internal staging environments where dev traffic shouldn't pollute
     production dashboards,
   - Customers running their own dashboard via the gateway API directly.

The brand cost of "we say it's opt-in but if you pay we force it on"
outweighs the marginal implementation gain.

### Worker-side write to `telemetry_events` for managed-cache traffic

**Considered as complementary, not alternative; deferred.** The dashboard
correctness today depends on the SDK behaving — older SDK forks or
out-of-date pins would still hit the footgun. A platform-side fix where
the billing worker writes minimal per-event rows to `telemetry_events`
from the Redis stream would decouple dashboard correctness from SDK
behavior entirely. This is the proper long-term shape but is meatier
work: gateway needs to emit more fields, worker needs a new write path,
the `telemetry_events` schema needs to flex, and a privacy review must
confirm none of the new fields violate ADR 0010 (managed-cache stream
events carry different metadata than `/v1/telemetry` payloads, which
are shape-locked).

Tracked separately. Does not block v0.7.0.

### Runtime warning only — no auto-connect

**Rejected as the only fix.** A warning when `Cache(api_key=...)` is
constructed without a prior `connect()` would help discoverability but
miss the larger population of users who don't read logs in development.
The footgun remains for every casual user, every quickstart-copier,
every blog example. The fix has to be behavioral, not advisory.

### Make `sulci.connect()` mandatory; remove `api_key=` from `Cache()`

**Rejected.** Breaking change. Every existing snippet in the
ecosystem breaks. The "drop-in library" positioning takes a hit
that auto-connect doesn't require us to take.

---

## Implementation notes

- Lives entirely in `sulci/core.py`. No protocol changes, no
  cross-repo coordination beyond the version bump.
- Test coverage in `tests/test_connect.py::TestCacheAutoConnect`,
  eight tests covering the matrix of `(api_key source, telemetry kwarg,
  prior-connect state, connect-failure)`.
- The `connect()` call is wrapped in `try/except Exception` to honor the
  pre-existing "telemetry never crashes the caller" contract (see
  `__init__.py::_emit` docstring for the same pattern at the emit layer).
- The `WARNING` log message format is stable and includes "auto-connect
  from Cache() failed" as a grep-friendly anchor for support tooling.

---

## Verification

```python
import logging, sulci
from sulci import Cache

# Demonstrates the v0.7.0 contract.
logging.basicConfig(level=logging.INFO)

# Pre-0.7.0 footgun, now fixed:
cache = Cache(backend="sulci", api_key="sk-sulci-...")
assert sulci._telemetry_enabled is True

# Explicit opt-out path remains:
sulci._telemetry_enabled = False  # simulate a fresh process
cache = Cache(backend="sulci", api_key="sk-sulci-...", telemetry=False)
assert sulci._telemetry_enabled is False

# OSS-Connect canonical flow (local cache + telemetry):
import os
os.environ["SULCI_API_KEY"] = "sk-sulci-env-test"
sulci._telemetry_enabled = False
cache = Cache(backend="sqlite")
assert sulci._telemetry_enabled is True
```

End of ADR 0001.
