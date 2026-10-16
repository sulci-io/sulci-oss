#!/usr/bin/env python3
"""
check_api_surface.py -- fail when docs/API-SURFACE.md and the code disagree.

    python3 scripts/check_api_surface.py            # check, exit 1 on drift
    python3 scripts/check_api_surface.py --show     # print the measured surface

API-SURFACE.md is the designated authority for the Cache / AsyncCache surface,
and every other document is told to link here rather than restate. It has
carried its own regeneration command since it was written -- but a command
nobody runs is not a mechanism. This is the missing half.

WHAT IT CHECKS, and why only these:

  * the public method set on each class      -- `delete_user` reached a
                                                published privacy policy as a
                                                GDPR mechanism while raising
                                                AttributeError
  * every keyword-only parameter, per method -- `metadata` was attributed to a
                                                method that never had it
  * every default in Cache.__init__          -- four were wrong at once, two of
                                                them behavioural (default
                                                backend, and whether entries
                                                expire)
  * the version the doc claims to describe   -- the doc's own header dates
                                                itself against a version

It deliberately does NOT check prose. A checker that fires on wording gets
deleted within a week; one that fires only on facts that have a single correct
value keeps working. The three defects above are all of that kind.

The doc is the thing under test, not the source. When they disagree the code
wins -- that is the file's own stated rule, so the failure message tells you to
update the doc, never the other way round.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "API-SURFACE.md"
SOURCES = (
    "sulci/core.py",
    "sulci/async_cache.py",
    "sulci/integrations/langchain.py",
    "sulci/integrations/llamaindex.py",
    "sulci/integrations/litellm.py",
    "sulci/proxy/app.py",
)
# The documented surface. _ProtocolAdaptedSessionStore is an internal adapter
# that happens to live in core.py and is deliberately absent.
#
# The two integration classes were added 2026-08-10. They are the classes with
# the most external readers -- a LangChain or LlamaIndex user reaches them
# without ever opening core.py -- and they had no drift guard at all.
# SulciLiteLLMCache added 2026-08-11 with the v0.9.0 surfaces, for the same
# reason the two above were added on 08-10: a LiteLLM user reaches it without
# ever opening core.py. The MCP server and the proxy expose FUNCTIONS
# (build_server / build_app) rather than a public class, so they are guarded
# by their own suites rather than here -- if either grows a public class,
# add it to this tuple.
CLASSES = (
    "Cache",
    "AsyncCache",
    "SulciCache",
    "SulciCacheLLM",
    "SulciLiteLLMCache",
)

# Classes whose doc table is a complete index of public methods, so a
# name-set comparison is meaningful.
#
# AsyncCache is NOT in this list, on purpose. Its table is a view of *which
# kwargs are forwarded* -- six rows against sixteen public methods -- and the
# section says so in its own heading. Adding it here would score a correct
# document red.
METHOD_SET_CHECKED = ("Cache", "SulciCache", "SulciCacheLLM", "SulciLiteLLMCache")

# Classes whose __init__ keyword-only args are compared against the doc.
# Cache is excluded: its constructor is covered in far more detail by the
# defaults comparison below. The adapters have no defaults check, so without
# this their one keyword-only constructor arg would be unguarded.
INIT_KWONLY_CHECKED = ("SulciCache", "SulciCacheLLM", "SulciLiteLLMCache")


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------
def measure() -> dict:
    """The real surface, by AST. Never imports sulci -- import side effects
    and a missing optional backend would both turn a doc check into a runtime
    failure for reasons that have nothing to do with the docs."""
    out: dict = {}
    for rel in SOURCES:
        path = ROOT / rel
        if not path.exists():
            sys.exit(f"check_api_surface: no {rel} under {ROOT}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name not in CLASSES:
                continue
            methods: dict = {}
            defaults: dict = {}
            for m in node.body:
                if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if m.name.startswith("_") and m.name != "__init__":
                    continue
                methods[m.name] = sorted(a.arg for a in m.args.kwonlyargs)
                if m.name == "__init__":
                    args = m.args.args[1:]           # drop self
                    pad = len(args) - len(m.args.defaults)
                    for i, a in enumerate(args):
                        if i >= pad:
                            d = m.args.defaults[i - pad]
                            try:
                                defaults[a.arg] = ast.literal_eval(d)
                            except (ValueError, SyntaxError):
                                defaults[a.arg] = ast.unparse(d)
            out[node.name] = {"methods": methods, "defaults": defaults}
    missing = [c for c in CLASSES if c not in out]
    if missing:
        sys.exit(f"check_api_surface: class(es) not found in source: {missing}")
    return out


def version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"',
                  (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "?"


# --------------------------------------------------------------------------
# the version of the doc's newest measurement
# --------------------------------------------------------------------------
#: The doc dates itself on `**Measured:**` lines, newest first; the version is
#: the one bolded after `sulci` on that line. The older lines are history and
#: keep the version they were measured against. The newest is the FIRST
#: `**Measured:**` a reader of the rendered doc sees, wherever it stands on its
#: line (a quote, a list item, a heading). Not one in an HTML comment, a fenced
#: block or a code span (where the doc names the marker): a reader that skipped
#: a newest line it did not expect, or read one nobody sees, would compare the
#: wrong line and could report green (both found by oss-0027's review).
#:
#: 0.10.0 (oss-0027): this is the ONE reader. `check_release_ready.py` imports
#: it, so the two checkers `make release-check` runs read the same thing. Until
#: then they did not: this file took the first `**Measured:**` match anywhere
#: and skipped its version check when none matched; check_release_ready.py
#: compared every `sulci` + version in the file, so the 0.9.2 line kept
#: `make release-check` red from 0.9.3 on. Both patterns read `x.y.z` only:
#: given a new top line at `**0.10.0rc1**` and that version in pyproject.toml,
#: this file skipped the new line and compared the next one down (0.9.3), and
#: check_release_ready.py read 0.10.0, 0.9.3 and 0.9.2 (measured on main,
#: cb67e19, with the rc1 line and version written in).
MEASURED_LINE = re.compile(r"\*\*Measured:\*\*(.*)$", re.M)
#: What a reader does not see: HTML comments, fenced blocks (closed by a fence
#: as long, with no info string) and lines indented four spaces or a tab (an
#: indented code block) are removed whole; a code span, which may wrap but not
#: cross a blank line, is removed when it holds the marker (`sulci` itself is a
#: span). A fence left over afterwards was never closed: refused.
_UNSEEN = (re.compile(r"<!--.*?-->", re.S),
           re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[^\n]*\n.*?^[ ]{0,3}\1[`~]*[ \t]*$", re.S | re.M),
           re.compile(r"^(?: {4}|\t)[^\n]*$", re.M))
_CODE_SPAN = re.compile(r"(`+)((?:(?!\n[ \t]*\n).)+?)\1", re.S)
_FENCE = re.compile(r"^[ ]{0,3}(?:`{3,}|~{3,})", re.M)
#: A release (`0.9.3`) or a pre-release (`0.10.0a1`, `0.10.0b2`, `0.10.0rc1`),
#: spelled as pyproject.toml spells it. Anything else is unreadable, and an
#: unreadable newest measurement FAILS: it is not a pass.
MEASURED_VERSION = re.compile(r"`sulci`\s+\*\*([0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?)\*\*")


def measured_version(text: str) -> str | None:
    """The version of the newest measurement in TEXT (API-SURFACE.md's body).

    None when the first `**Measured:**` a reader sees names no version this
    can read after `sulci` on its line, when there is none, or when text no
    reader sees names a different newest version. Callers fail on None."""
    text = text.replace("\r\n", "\n")         # a caller's raw CRLF text reads as the file does
    shown = text
    for unseen in _UNSEEN:
        shown = unseen.sub("", shown)
    if _FENCE.search(shown):
        return None
    shown = _CODE_SPAN.sub(lambda m: "" if "**Measured:**" in m.group(2) else m.group(0), shown)
    line = MEASURED_LINE.search(shown)
    if not line:
        return None
    m = MEASURED_VERSION.search(line.group(1))
    if not m:
        return None
    # Backstop, failing closed: the removal above is not a Markdown parser. If
    # the plain text's first readable measurement names another version,
    # refuse to read rather than choose.
    for raw in MEASURED_LINE.finditer(text):
        r = MEASURED_VERSION.search(raw.group(1))
        if r:
            return m.group(1) if r.group(1) == m.group(1) else None
    return m.group(1)


# --------------------------------------------------------------------------
# read the doc's claims
# --------------------------------------------------------------------------
def claimed() -> dict:
    """Pull the checkable claims out of the markdown.

    Anchored on structure the doc already has -- the `Cache(...)` constructor
    block and the method tables -- rather than on sentences. If a heading is
    reworded this still works; if a default changes, it does not."""
    if not DOC.exists():
        sys.exit(f"check_api_surface: {DOC} is missing")
    # UTF-8, not the platform's encoding: on Windows that is cp1252, which cannot
    # decode this doc (its ⚠️ holds byte 0x8F) -- 0.10.0 (oss-0029).
    text = DOC.read_text(encoding="utf-8")
    out: dict = {"defaults": {}, "kwonly": {}, "methods": {}, "version": None}

    out["version"] = measured_version(text)

    # constructor: the ```python block containing `Cache(`
    for block in re.findall(r'```python\n(.*?)```', text, re.S):
        if not re.search(r'\bCache\s*\(', block):
            continue
        for line in block.splitlines():
            # `cache = Cache(` is the assignment target, not a kwarg
            if re.search(r'=\s*(?:Async)?Cache\s*\(', line):
                continue
            m = re.match(r'\s*([a-z_]+)\s*=\s*([^,#]+?)\s*,?\s*(?:#.*)?$', line)
            if not m:
                continue
            key, raw = m.group(1), m.group(2).strip()
            try:
                out["defaults"][key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                out["defaults"][key] = raw
        break

    # Method tables, keyed PER CLASS. `get` / `set` / `cached_call` appear in
    # more than one table -- Cache's real signature and AsyncCache's narrower
    # sync passthrough -- so a flat dict keeps whichever came first and reports
    # the other as a contradiction that does not exist.
    for cls, seg in _segments(text).items():
        rows, names = {}, set()
        for row in re.findall(r'^\|\s*`([^`]+)`[^|]*\|([^|]*)\|', seg, re.M):
            name = re.match(r'(__init__|[a-z_]+)', row[0])
            if not name:
                continue
            names.add(name.group(1))
            kw = sorted(set(re.findall(r'`([a-z_]+)`', row[1])))
            if kw:
                rows[name.group(1)] = kw
        out["kwonly"][cls] = rows
        out["methods"][cls] = names
    return out


def _segments(text: str) -> dict:
    """Split the doc into one section per documented class.

    Was: everything before the AsyncCache heading is Cache, everything after is
    AsyncCache. That runs off the end -- with two more classes and the Backends
    and Embedding-models tables below them, "everything after" swept `chroma`,
    `qdrant` and `minilm` into the last class as though they were methods.

    Now: a `##` section belongs to a class when the first backticked token in
    its heading is exactly that class name. `## \\`Cache.__init__\\` — the real
    defaults` is therefore NOT the Cache section; it is its own thing, and the
    defaults comparison reads it separately."""
    out: dict = {}
    heads = list(re.finditer(r'^##\s+(.*)$', text, re.M))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        m = re.search(r'`([A-Za-z_][A-Za-z0-9_.]*)`', h.group(1))
        if m and m.group(1) in CLASSES:
            out.setdefault(m.group(1), text[h.start():end])
    return out


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true",
                    help="print the measured surface and exit")
    args = ap.parse_args()

    real = measure()
    ver = version()

    if args.show:
        print(f"sulci {ver} -- measured by AST\n")
        for cls, d in real.items():
            print(f"### {cls}")
            for name, kw in sorted(d["methods"].items()):
                print(f"  {name}(*, {', '.join(kw) if kw else '-'})")
            if d["defaults"]:
                print("  __init__ defaults:")
                for k, v in d["defaults"].items():
                    print(f"    {k:16s} {v!r}")
            print()
        return 0

    doc = claimed()
    fails: list[str] = []

    if doc["version"] is None:
        fails.append(
            "version: the doc's newest measurement (its first "
            "**Measured:**) names no version this can read, or text no reader "
            "sees names another -- write it as `sulci` **X.Y.Z**, or "
            "**X.Y.ZrcN** for a pre-release")
    elif doc["version"] != ver:
        fails.append(
            f"version: doc says it was measured against {doc['version']}, "
            f"pyproject.toml reads {ver}")

    for key, want in real["Cache"]["defaults"].items():
        if key not in doc["defaults"]:
            fails.append(f"default `{key}`: absent from the doc (code: {want!r})")
        elif doc["defaults"][key] != want:
            fails.append(
                f"default `{key}`: doc says {doc['defaults'][key]!r}, "
                f"code says {want!r}")
    for key in doc["defaults"]:
        if key not in real["Cache"]["defaults"]:
            fails.append(f"default `{key}`: in the doc, not in Cache.__init__")

    # Method SETS. The module docstring has always claimed this checker verifies
    # "the public method set on each class"; until 2026-08-10 nothing compared
    # them. Only the per-method kwarg loop below ran, and it skips any method
    # with no keyword-only params -- which is all eleven adapter methods. Adding
    # the adapters to SOURCES without this would have installed a guard that
    # cannot fire, and reported green while doing it.
    for cls in METHOD_SET_CHECKED:
        want = {n for n in real[cls]["methods"] if n != "__init__"}
        have = {n for n in doc["methods"].get(cls, set()) if n != "__init__"}
        for n in sorted(want - have):
            fails.append(f"{cls}.{n}: public method, absent from the doc")
        for n in sorted(have - want):
            fails.append(f"{cls}.{n}: in the doc, not a public method on the class")

    for cls in INIT_KWONLY_CHECKED:
        want = real[cls]["methods"].get("__init__", [])
        have = doc["kwonly"].get(cls, {}).get("__init__", [])
        if want != have:
            fails.append(
                f"{cls}.__init__: doc lists keyword-only "
                f"{have or '(none)'}, code has {want or '(none)'}")

    for cls, d in real.items():
        for name, want in d["methods"].items():
            if name == "__init__" or not want:
                continue
            have = doc["kwonly"].get(cls, {}).get(name)
            if have is None:
                fails.append(f"{cls}.{name}: no keyword-only row in the doc "
                             f"(code: {', '.join(want)})")
            elif have != want:
                extra = [k for k in have if k not in want]
                miss = [k for k in want if k not in have]
                bits = []
                if extra:
                    bits.append("doc claims " + ", ".join(f"`{k}`" for k in extra)
                                + " which the code does not have")
                if miss:
                    bits.append("code has " + ", ".join(f"`{k}`" for k in miss)
                                + " which the doc omits")
                fails.append(f"{cls}.{name}: " + "; ".join(bits))

    if not fails:
        n = sum(len(d["methods"]) for d in real.values())
        print(f"check-api-surface: docs/API-SURFACE.md is current "
              f"-- sulci {ver}, {n} public methods across {len(real)} classes")
        return 0

    print("check-api-surface: FAIL -- docs/API-SURFACE.md disagrees with the "
          "code.\n")
    for f in fails:
        print(f"  * {f}")
    print("\n  The code wins. That is this document's own rule -- update the "
          "doc,\n  date it, and grep the estate for anything else restating "
          "the old value.\n"
          "  Measured surface:  python3 scripts/check_api_surface.py --show")
    return 1


if __name__ == "__main__":
    sys.exit(main())
