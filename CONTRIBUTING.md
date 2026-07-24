# Contributing to Sulci

Thank you for your interest in contributing!

## Development setup

```bash
git clone https://github.com/sulci-io/sulci-oss.git
cd sulci-oss
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[sqlite,dev]"
```

## Running tests

```bash
# Core tests (no extra dependencies)
pytest tests/test_core.py -v

# All tests (skips backends whose deps aren't installed)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=sulci --cov-report=term-missing
```

## Adding a new backend

Sulci's `Backend` protocol (`sulci/backends/protocol.py`) defines the
contract every backend must satisfy.

1. **Create `sulci/backends/yourbackend.py`** implementing the protocol:
   `store()`, `search()`, `clear()`. Use the keyword-only convention
   (`*,` separator) for `tenant_id`, `user_id`, and other partition
   kwargs — see existing backends for the pattern.

2. **Declare `ENFORCES_TENANT_ISOLATION`** as a class attribute. `True`
   if `search()` actually filters out other tenants (Qdrant pattern);
   `False` if `tenant_id` is stored as a label only.

3. **Register in `sulci/core.py` `_load_backend()`** so users can
   construct `Cache(backend="yourbackend")`.

4. **Add an extra in `pyproject.toml` `[project.optional-dependencies]`**
   so users install via `pip install "sulci[yourbackend]"`.

5. **Register the class in `tests/compat/conftest.py` `BACKEND_CLASSES`**.
   The conformance suite (`tests/compat/`) automatically runs structural,
   round-trip, and tenant-isolation tests against every entry.

6. **Optionally add backend-specific tests** in `tests/test_backends.py`
   using `_run_backend_contract()` for edge cases the protocol contract
   doesn't cover.

Worked example: `examples/extending_sulci/custom_backend.py` is a
~150-line in-memory `InMemoryBackend` that satisfies the full protocol.

## Pre-publish review

Before publishing anything to a public repo (issue body, issue comment,
PR description, commit message, or release notes), do a quick scan for
content that shouldn't appear in public.

**Don't publish in public repos:**

- Tier names (Free, Pro, Business, Enterprise) unless the tiers are
  already publicly announced on sulci.io
- Monetization mechanics (which features are paid, what's gated, pricing)
- Concrete feature candidates being evaluated for paid tiers (e.g.,
  "we'd gate X behind Y") — even hypothetically
- Internal cross-references like "FU-14" or "internal Notion doc" —
  if a reader can't act on a reference, leaving it visible just signals
  that internal planning exists at a specific pointer
- Competitor names alongside strategy commentary — factual mentions
  are fine, comparative monetization claims aren't
- Customer or contract specifics (logos, MRR, deal mechanics)

**Always fine in public:**

- Technical rationale (why a flag is keyword-only, why a backend
  enforces isolation, what a protocol guarantees)
- API design tradeoffs and alternatives considered
- Test outcomes, benchmark numbers, performance trade-offs
- Public follow-up issue numbers (FU-N where N maps to a GitHub issue
  in the same public repo)
- Honest acknowledgments of limitations or known issues

**Suggested workflow before any commit on a public repo:**

1. Read the message you're about to commit, not just the title.
2. Search for words like `tier`, `Pro`, `Enterprise`, `monetize`,
   `gate`, `paywall`, `internal`, `FU-` (if FU-N is internal-only).
3. If anything matches, ask: would the same point work without that
   word? Usually yes — rewrite to keep the technical content and drop
   the strategic framing.
4. If you can't avoid the word, the content probably belongs in
   internal planning docs instead of a public commit.

The same rule applies to issue bodies and comments. Issues on public
repos are even more searchable than commit history.

If you discover an existing public artifact already contains something
that shouldn't have been published, the standard response is: edit the
body in-place if possible, file a follow-up note to track the cleanup,
and improve process going forward. Force-pushing to rewrite published
commit history is generally not worth the operational disruption for
moderate leaks — the content is already distributed via clones and
reflog regardless.

### Retroactive cleanup

If you discover that something already published to a public surface
needs to be removed or sanitized, the fix is rarely a single place:

1. Sanitize the original artifact first (issue body, comment, commit
   message, or release notes - whatever the source is).
2. Run a repo-wide search for the same terms. Content has shadow copies
   in helper scripts, tooling files, drafts that got committed with the
   rest of a change, comments referencing the original, or generated
   files. A short grep catches most of them:

```
git grep -nE "<term1>|<term2>|<term3>" -- ':!CONTRIBUTING.md'
```

   Exclude `CONTRIBUTING.md` because the terms-to-avoid list legitimately
   contains the words.
3. Sanitize each shadow copy and commit the cleanup as a focused
   `chore:` change. The commit message should explain why the specific
   edits exist; future contributors who find them via `git log` should
   understand the reasoning.
4. Accept that the original (un-sanitized) content remains in git history
   at the commits where it landed. Force-pushing to rewrite published
   history is generally not worth the operational disruption for moderate
   leaks - see the note in the parent section.

The "sanitize once, grep the rest, accept history" pattern lets you fix
forward cleanly without tripping over force-push fallout.

## Releasing

The release workflow ships v0.X.Y to PyPI and creates a GitHub Release
when an annotated `vX.Y.Z` tag is pushed.

```bash
# 1. Bump version in pyproject.toml (single source of truth as of v0.4.0).
#    sulci.__version__ derives from this via importlib.metadata.

# 2. Refresh editable-install metadata so importlib.metadata.version()
#    sees the new version locally. Without this, your dev environment
#    keeps reporting the previous version even though pyproject.toml
#    was bumped.
pip install -e . --no-deps

# 3. Add a [X.Y.Z] entry to CHANGELOG.md following the Keep a Changelog
#    format (sections: Added, Changed, Fixed, Deprecated, Removed,
#    Security, Notes).

# 4. Run the full pre-PR check before tagging.
make checkin

# 5. Open a PR. After CI green, merge to main using a merge commit
#    (NOT squash) so individual sub-phase commits are preserved in the
#    main branch history.
#
#    On a solo repo that merge needs --admin, and that is the normal
#    path, not an exception:
#
#      gh pr merge <N> --merge --admin
#
#    main is protected by the ruleset `protect-main` (id 17006725),
#    which sets required_approving_review_count: 1. GitHub does not
#    permit self-approval, so with one maintainer every PR sits at
#    REVIEW_REQUIRED / BLOCKED permanently and admin bypass is the only
#    merge path. The review requirement is retained deliberately, for
#    when contributors arrive — do not re-derive it as a bug and do not
#    turn it off. Once a second maintainer exists, drop --admin and
#    delete this note.
#
#    Rulesets are invisible to the classic-protection API:
#    `gh api repos/sulci-io/sulci-oss/branches/main/protection` returns
#    404 "Branch not protected" while the ruleset is active. Read it
#    with /rules/branches/main or /rulesets/17006725 instead.
#
#    The twelve required contexts are the matrix job names
#    (`test (ubuntu-latest, 3.9)` and so on). tests.yml guards every
#    STEP rather than the job, deliberately: a job-level `if:` is
#    evaluated before the matrix expands, so it emits ONE collapsed
#    check named `test` instead of twelve, and a required context that
#    never reports leaves the PR Expected forever.

# 6. After merge, tag main and push.
git checkout main
git pull
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z

# 7. The publish.yml workflow triggers on the tag push and publishes
#    to PyPI automatically. There is NO PYPI_TOKEN to configure: since
#    v0.7.3 (2026-06-05) the release path is credential-free, using
#    PyPI Trusted Publishing over OIDC — publish.yml requests
#    `id-token: write` under `environment: pypi` and hands off to
#    pypa/gh-action-pypi-publish with no username or password. PEP 740
#    attestations are generated automatically. The old long-lived token
#    was deleted from repo secrets AND revoked on PyPI; see
#    docs/OSS_BOUNDARY_POLICY.md. If a release fails to authenticate,
#    the fault is in the Trusted Publisher config on PyPI or in the
#    `environment:` name — never a missing secret.
#
#    A `v*` tag is the only publish trigger, and creating one is itself
#    restricted by the `protect-release-tags` ruleset (bypass:
#    repository admin only), so write access alone cannot ship a
#    release.

# 8. After publish completes, create a GitHub Release linked to the tag
#    using the matching CHANGELOG entry as the release notes:
awk '/^## \[X\.Y\.Z\]/{flag=1; next} /^## \[/{flag=0} flag' CHANGELOG.md > /tmp/release_notes.md
gh release create vX.Y.Z \
    --title "vX.Y.Z: <one-line summary>" \
    --notes-file /tmp/release_notes.md \
    --verify-tag

# 9. Verify install from PyPI in a fresh venv.
python3 -m venv /tmp/verify-vX.Y.Z
source /tmp/verify-vX.Y.Z/bin/activate
pip install --upgrade "sulci[sqlite]"
python -c "import sulci; print(sulci.__version__)"
```

For longer commit messages or release notes, write them to a file and
use `git commit -F` or `gh release create --notes-file` to avoid shell
quoting issues.

## Code style

- Black formatting: `pip install black && black sulci/`
- Type hints encouraged but not required
- Docstrings on all public classes and methods
