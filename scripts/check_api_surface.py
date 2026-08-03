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
SOURCES = ("sulci/core.py", "sulci/async_cache.py")
# Only these are the documented surface. _ProtocolAdaptedSessionStore is an
# internal adapter that happens to live in core.py.
CLASSES = ("Cache", "AsyncCache")


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
    out: dict = {"defaults": {}, "kwonly": {}, "version": None}

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
    # both tables -- Cache's real signature and AsyncCache's narrower sync
    # passthrough -- so a flat dict keeps whichever came first and reports the
    # other as a contradiction that does not exist.
    m = re.search(r'^#+\s*`?AsyncCache', text, re.M)
    split = m.start() if m else len(text)
    for cls, seg in (("Cache", text[:split]), ("AsyncCache", text[split:])):
        rows = {}
        for row in re.findall(r'^\|\s*`([^`]+)`[^|]*\|([^|]*)\|', seg, re.M):
            name = re.match(r'([a-z_]+)', row[0])
            if not name:
                continue
            kw = sorted(set(re.findall(r'`([a-z_]+)`', row[1])))
            if kw:
                rows[name.group(1)] = kw
        out["kwonly"][cls] = rows
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
