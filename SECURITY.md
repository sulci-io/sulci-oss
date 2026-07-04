# Security Policy

Thanks for helping keep `sulci` and its users safe. This document explains
which versions receive security fixes, how to report a vulnerability, and
what to expect after you do.

`sulci` is the open-source Python library published to PyPI as
[`sulci`](https://pypi.org/project/sulci/) and developed at
[`sulci-io/sulci-oss`](https://github.com/sulci-io/sulci-oss). The hosted
Sulci platform (the gateway at `api.sulci.io` and the dashboard at
`dashboard.sulci.io`) is a **separate** codebase and is out of scope for
this policy; see "Scope" below.

## Supported versions

Sulci is pre-1.0 and ships frequently. Security fixes are released as a new
patch version on the current release line, and older versions are **not**
backported. If you are affected by a security issue, the fix will be to
upgrade to the latest release.

| Version         | Supported                              |
| --------------- | -------------------------------------- |
| Latest `0.7.x`  | ✅ Security fixes                       |
| `< 0.7.0`       | ❌ Please upgrade to the latest release |

The library is tested and supported on **Python 3.9–3.12** across Linux,
macOS, and Windows. Reports against unsupported Python versions may be
closed with an upgrade recommendation.

## Reporting a vulnerability

**Please do not open a public GitHub issue, pull request, or discussion for
security vulnerabilities.** Public reports expose users before a fix is
available.

Use one of these private channels instead:

1. **GitHub private vulnerability reporting (preferred).** On the
   [`sulci-io/sulci-oss`](https://github.com/sulci-io/sulci-oss) repository,
   go to the **Security** tab and choose **Report a vulnerability**. This
   opens a private advisory thread with the maintainer and lets us
   coordinate a fix and, where warranted, a CVE and published advisory.
2. **Email.** Write to **security@sulci.io**. If you would like to encrypt
   your report, ask for a PGP key in your first message and we will provide
   one before you send details.

To help us triage quickly, please include:

- A description of the vulnerability and its impact.
- The affected version(s) and, if known, the affected module or code path.
- Steps to reproduce, a proof of concept, or a failing test case.
- Any relevant configuration (backend, embedding model, whether telemetry
  or an API key was involved).

Please report privately even if you are not certain the issue is
exploitable. We would rather assess a false alarm than miss a real one.

## What to expect

We are a small team and will do our best to meet the following targets,
measured in business days:

| Stage                          | Target                                    |
| ------------------------------ | ----------------------------------------- |
| Acknowledge your report        | Within **3 business days**                |
| Initial assessment and triage  | Within **7 business days**                |
| Fix or mitigation plan         | Within **30 days** for confirmed issues   |
| Coordinated public disclosure  | After a fix ships, by mutual agreement    |

We will keep you updated as we work through triage and remediation, and we
are happy to credit you in the release notes and any published advisory
unless you prefer to remain anonymous.

## Coordinated disclosure

We follow coordinated disclosure. We ask that you give us a reasonable
opportunity to release a fix before disclosing publicly — typically until a
patched release is available, or **90 days** from your report, whichever
comes first. We will work with you on timing if a fix needs longer.

## Safe harbor

We will not pursue or support legal action against researchers who, in good
faith, discover and report vulnerabilities in accordance with this policy,
who avoid privacy violations and service disruption, and who do not access
or modify data beyond what is necessary to demonstrate the issue. If in
doubt about whether an action is authorized, ask us first at
security@sulci.io.

There is no paid bug bounty program at this time. We recognize reporters
through credit in advisories and release notes.

## Scope

**In scope** — the `sulci` library in this repository, including:

- The cache engine and public API (`sulci.core`, `sulci.context`).
- Vector backends (`sulci.backends.*`), embedding adapters
  (`sulci.embeddings.*`), session stores, and event sinks.
- The opt-in telemetry path and how API keys and configuration
  (`~/.sulci/config`) are handled by the library.
- Supply-chain integrity of the published `sulci` package (see "Verifying
  releases").

**Out of scope** — report these to the operator of the relevant service,
not through this repository:

- The hosted Sulci platform (`api.sulci.io`, `dashboard.sulci.io`) and the
  marketing site (`www.sulci.io`).
- Vulnerabilities in third-party dependencies with no exploitable path
  through `sulci`. Please report those upstream; tell us if `sulci`'s usage
  makes them exploitable.
- Findings that require a compromised host, a malicious local backend, or
  physical access, unless they cross a documented trust boundary.

## Verifying releases

Since v0.7.3, `sulci` is published to PyPI via OIDC Trusted Publishing with
[PEP 740](https://peps.python.org/pep-0740/) attestations generated
automatically by the release workflow — there is no long-lived PyPI token
in the project. Attestations are visible on the file listing at
<https://pypi.org/project/sulci/>. If you are evaluating the supply chain,
you can verify that a release artifact was built and published by this
repository's `publish.yml` workflow.

Thank you for practicing responsible disclosure.
