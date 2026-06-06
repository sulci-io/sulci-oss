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

1. **PyPI Trusted Publishing** — replace the long-lived `PYPI_TOKEN` secret with OIDC:
   PyPI → project → Publishing → add GitHub publisher
   (repo `id4git/sulci`, workflow `publish.yml`, environment `pypi`).
   Then in `publish.yml`:
   ```yaml
   permissions:
     id-token: write
   steps:
     - uses: pypa/gh-action-pypi-publish@release/v1   # no password input
   ```
2. **Tag protection** — GitHub → Settings → Tags → protect `v*` so only admins can
   push release tags (publish.yml is tag-triggered; an attacker with push access must
   not be able to trigger a release).
3. **Attestations** — `pypa/gh-action-pypi-publish` generates PEP 740 attestations
   automatically under Trusted Publishing; leave it on.

## 3. History hygiene before GA

Run once from each repo (platform especially):

```bash
brew install gitleaks   # or: docker run ghcr.io/gitleaks/gitleaks
gitleaks detect --source . --log-opts="--all"
```

If anything real surfaces: rotate the credential FIRST, then scrub history with
`git filter-repo` if the repo is public.

## 4. Legal markers (verify, all present already)

- LICENSE (Apache-2.0) — includes the patent grant + termination clause: anyone who
  sues you over patents covering sulci loses their license to use it.
- NOTICE — keep "Sulci — Patent Pending" + copyright line current.
- pyproject description — "Patent Pending" string is your public notice; keep it.
