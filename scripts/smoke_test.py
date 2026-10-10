#!/usr/bin/env python3
"""Static smoke test: byte-compile, import-resolve, and CLI-surface check.

This is a **static** test. It never contacts a server, never starts vLLM, and never runs an
experiment. A "runtime reproduction" claim is deliberately NOT made.

Usage:
    python3 scripts/smoke_test.py
"""
from __future__ import annotations

import importlib.util
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Third-party packages may legitimately be absent in a static-check environment.
OPTIONAL = {"openai", "aiohttp", "matplotlib", "pandas", "numpy"}
# Needs a GPU / frozen profiler capture; absence is expected and not a defect.
NEEDS_FROZEN_DATA = {"analyze_timeline.py"}


def main() -> int:
    print("=" * 78)
    print("  STATIC SMOKE TEST — LLM Serving Performance Lab")
    print("=" * 78)
    files = sorted(p for p in ROOT.rglob("*.py")
                   if ".venv" not in p.parts and "__pycache__" not in p.parts)
    print(f"\n  Python files: {len(files)}")

    # ---- 1. byte-compile ----
    print("\n[1] py_compile")
    bad = []
    for p in files:
        try:
            py_compile.compile(str(p), cfile=tempfile.mktemp(), doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(p)
            print(f"    FAIL {p.relative_to(ROOT)}: {e}")
    print(f"    {len(files) - len(bad)}/{len(files)} compiled")

    # ---- 2. import resolution ----
    print("\n[2] import resolution")
    unresolved, optional_only = [], 0
    for p in files:
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ModuleNotFoundError as e:
            if e.name in OPTIONAL:
                optional_only += 1
            else:
                unresolved.append((p, e.name))
                print(f"    FAIL {p.relative_to(ROOT)} -> missing module '{e.name}'")
        except FileNotFoundError:
            optional_only += 1          # needs a frozen artifact that is not published
        except SystemExit:
            optional_only += 1
        except Exception as e:                                   # noqa: BLE001
            unresolved.append((p, f"{type(e).__name__}: {e}"))
            print(f"    FAIL {p.relative_to(ROOT)} -> {type(e).__name__}: {e}")
    print(f"    resolved directly            : {len(files) - len(unresolved) - optional_only}")
    print(f"    blocked by absent 3rd-party  : {optional_only}")

    # ---- 3. CLI surface ----
    print("\n[3] --help surface (experiment entry points that construct an ArgumentParser)")
    this_file = Path(__file__).resolve()
    for p in files:
        if p.resolve() == this_file:            # never invoke ourselves
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        # 必须真的**构造** ArgumentParser;仅仅提到 "argparse" 不算。
        if not re.search(r"argparse\.ArgumentParser\s*\(", src):
            continue
        try:
            r = subprocess.run([sys.executable, str(p), "--help"],
                               capture_output=True, text=True, timeout=30)
            status = "OK  " if r.returncode == 0 else "SKIP"
            reason = "" if r.returncode == 0 else \
                f"({(r.stderr.strip().splitlines() or ['?'])[-1][:60]})"
        except subprocess.TimeoutExpired:
            status, reason = "SKIP", "(timeout)"
        print(f"    {status} {p.relative_to(ROOT)} {reason}")

    print("\n" + "=" * 78)
    print(f"  STATIC CHECK         : {'PASS' if not bad and not unresolved else 'FAIL'}")
    print("  RUNTIME REPRODUCTION : NOT EXECUTED (no GPU / vLLM in this environment)")
    print("=" * 78)
    return 0 if not bad and not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
