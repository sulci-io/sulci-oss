# sulci-oss — Boundary & Protection Policy

sulci-oss is **public on purpose** (Apache-2.0, PyPI distribution is the GTM funnel).
You do not protect it by hiding it; you protect the business by controlling **what's
in it** and **who can publish it**.

## 1. The boundary rule (enforce on every PR)

A change belongs in sulci-oss ONLY if all are true:

- [ ] It runs entirely on the user's machine OR is a thin transport to the managed API.
- [ ] It contains no fleet-derived tuning (adaptive thresholds, learned decay schedules,
      per-domain threshold tables from production telemetry).
- [ ] It contains no billing, quota, plan, or pricing logic.
- [ ] It reveals no platform internals (gateway routes beyond the public API contract,
      infra topology, internal service names).
- [ ] Its removal from OSS would not weaken the managed product's differentiation.

Anything failing a checkbox goes to sulci-platform. The published blend math
(α=0.70, decay=0.50) and the 6 backends are already public — that ship has sailed and
that's fine; it's the *adaptive/fleet* layer that is the moat.

## 2. Release integrity (protects users AND your account)

> **Status (2026-06-05):** all three items are DONE and live-verified by the
> `v0.7.3` release (OSS PR #100 + Publish-to-PyPI run #48 — OIDC exchange in
> the logs, attestations on the PyPI file listing).

1. ✅ **PyPI Trusted Publishing** (done 2026-06-05) — the long-lived `PYPI_TOKEN`
   secret is replaced with OIDC:
   PyPI → project → Publishing → GitHub publisher registered
   (repo `sulci-io/sulci-oss`, workflow `publish.yml`, environment `pypi`).
   In `publish.yml`:
   ```yaml
   environment: pypi          # must match the publisher config exactly
   permissions:
     id-token: write
   steps:
     - uses: pypa/gh-action-pypi-publish@release/v1   # no password input
   ```
   `PYPI_TOKEN` was deleted from repo secrets AND revoked on PyPI
   (`sulci-github-actions`, all-projects scope) — the old path is dead at
   both ends.
2. ✅ **Tag protection** (set 2026-06-05) — tag ruleset `protect-release-tags`:
   pattern `v*` (matched 34 existing release tags at creation), restrict
   create/update/delete + block force pushes, Repository-admin bypass.
   publish.yml is tag-triggered, so this closes the last path by which
   non-admin push access could mint a PyPI release.
3. ✅ **Attestations** (live as of v0.7.3) — `pypa/gh-action-pypi-publish` generates
   PEP 740 attestations automatically under Trusted Publishing; leave it on.

## 3. History hygiene before GA

Run once from each repo (platform especially):

```bash
brew install gitleaks   # or: docker run ghcr.io/gitleaks/gitleaks
gitleaks detect --source . --log-opts="--all"
```

If anything real surfaces: rotate the credential FIRST, then scrub history with
`git filter-repo` if the repo is public.

> **Executed 2026-06-05 across all three repos** — sulci-platform (559 commits,
> 205 hits), sulci-web (228 commits, 0 hits), sulci-oss (148 commits, 7 hits).
> Every hit classified as deliberate test fixtures (`sk-sulci-*` keys + their
> sha256 hashes in seed/fixture SQL; Clerk `pk_test_` publishable key). **Zero
> real secrets, zero rotations.** `.gitleaks.toml` allowlists committed to
> platform + oss so the scan now reports `no leaks found` and is viable as a
> pre-push gate. Git remotes audited the same day: no embedded tokens (the old
> `https://id4git:TOKEN@…` workaround is gone everywhere).

## 4. Legal markers (verify, all present already)

- LICENSE (Apache-2.0) — includes the patent grant + termination clause: anyone who
  sues you over patents covering sulci loses their license to use it.
- NOTICE — keep "Sulci — Patent Pending" + copyright line current.
- pyproject description — "Patent Pending" string is your public notice; keep it.
