# Architecture Decision Records (ADRs)

This directory holds the decision log for the `sulci-oss` library —
one file per decision, numbered sequentially, never renumbered.

An ADR captures **why** a technical choice was made, the **alternatives
considered**, and the **consequences accepted**. It is not a tutorial,
a how-to guide, or a status update. It is a point-in-time artifact of
a decision.

ADRs are the antidote to "why is it built this way?" three years from now
when everyone who was in the room has moved on.

The companion ADR registry for the platform side lives at
`sulci-platform/docs/architecture/adrs/`. Cross-references between the
two are explicit: a sulci-oss ADR that has a platform-side mirror names
the sulci-platform ADR number in its header, and vice versa.

---

## When to write an ADR

**Write one when:**

- A choice has long-term consequences for the public API or library
  contract (telemetry behavior, backend protocol, embedding interface)
- Reasonable engineers could disagree, and the choice will be questioned
  again
- The decision reverses, supersedes, or deviates from a prior one
- An external constraint (privacy, compliance, customer requirement)
  drove the choice

**Skip it when:**

- The choice is obvious or enforced by the stack
- It's a local implementation detail with no cross-cutting impact
- It's a bug fix or routine refactor

When in doubt, write one. Cheap to produce, expensive to wish you had.

---

## Numbering

- Sequential, zero-padded to 4 digits: `0001`, `0002`, ..., `9999`
- Never reused. Never renumbered. Superseded ADRs keep their number.
- Filename format: `NNNN-kebab-case-title.md`

---

## Status lifecycle

```
Proposed ──▶ Accepted ──▶ Superseded by ADR-NNNN
```

---

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [0001](0001-cache-auto-connect-telemetry.md) | `Cache()` auto-connects telemetry when an `api_key` is resolvable | Accepted | 2026-05-26 |

> **Cross-repo note:** sulci-oss ADR-0001 has a sulci-platform mirror at
> ADR-0021. Both ship as part of the coordinated v0.7.0 release
> (`sulci-cache==0.7.0` + `sulci-gateway==0.7.0`).
