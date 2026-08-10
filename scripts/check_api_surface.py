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
)
# The documented surface. _ProtocolAdaptedSessionStore is an internal adapter
# that happens to live in core.py and is deliberately absent.
#
# The two integration classes were added 2026-08-10. They are the classes with
# the most external readers -- a LangChain or LlamaIndex user reaches them
# without ever opening core.py -- and they had no drift guard at all.
CLASSES = ("Cache", "AsyncCache", "SulciCache", "SulciCacheLLM")

# Classes whose doc table is a complete index of public methods, so a
# name-set comparison is meaningful.
#
# AsyncCache is NOT in this list, on purpose. Its table is a view of *which
# kwargs are forwarded* -- six rows against sixteen public methods -- and the
# section says so in its own heading. Adding it here would score a correct
# document red.
METHOD_SET_CHECKED = ("Cache", "SulciCache", "SulciCacheLLM")

# Classes whose __init__ keyword-only args are compared against the doc.
# Cache is excluded: its constructor is covered in far more detail by the
# defaults comparison below. The adapters have no defaults check, so without
# this their one keyword-only constructor arg would be unguarded.
INIT_KWONLY_CHECKED = ("SulciCache", "SulciCacheLLM")


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
        tree = ast.parse(path.read_text())
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
                  (ROOT / "pyproject.toml").read_text(), re.M)
    return m.group(1) if m else "?"


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
    text = DOC.read_text()
    out: dict = {"defaults": {}, "kwonly": {}, "methods": {}, "version": None}

    m = re.search(r'\*\*Measured:\*\*[^\n]*?\*\*([0-9]+\.[0-9]+\.[0-9]+)\*\*', text)
    if m:
        out["version"] = m.group(1)

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

    if doc["version"] and doc["version"] != ver:
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
