#!/usr/bin/env python3
"""Report whether this machine can run (a) the analysis scripts, (b) the experiments.

Exit code 0 = analysis path is viable. It does NOT assert that the GPU experiments can run —
that is reported separately and never claimed as PASS without an actual run.

Usage:
    python3 scripts/check_environment.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OPTIONAL_PY = {
    "openai": "required by common/request_client.py and the experiment runners",
    "aiohttp": "required by common/exp02_formal.py and common/exp03_formal.py",
    "matplotlib": "required by figures/make_figures.py",
    "pandas": "optional",
}
REQUIRED_DATA = [
    "results/exp08_condition_summary.csv",
    "results/exp08_relative_to_mbt2048.csv",
    "results/exp11_cell_summary.csv",
    "results/exp11_interaction_summary.csv",
    "results/exp12_hypothesis_status.csv",
]
ENV_VARS = ["VLLM_BIN", "MODEL_PATH", "VLLM_HOME", "VLLM_HOST", "VLLM_PORT", "RESULTS_DIR"]


def main() -> int:
    print("=" * 74)
    print("  ENVIRONMENT CHECK — LLM Serving Performance Lab")
    print("=" * 74)

    print("\n[1] Python")
    print(f"    version : {sys.version.split()[0]}")
    print(f"    exe     : {sys.executable}")
    py_ok = sys.version_info >= (3, 10)
    print(f"    >= 3.10 : {'OK' if py_ok else 'TOO OLD'}")

    print("\n[2] Python packages")
    missing = []
    for mod, why in OPTIONAL_PY.items():
        try:
            m = importlib.import_module(mod)
            print(f"    OK      {mod:<12} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"    MISSING {mod:<12} ({why})")
            missing.append(mod)
    analysis_ok = not ({"matplotlib"} & set(missing))

    print("\n[3] Frozen data files")
    data_ok = True
    for rel in REQUIRED_DATA:
        p = ROOT / rel
        exists = p.is_file()
        data_ok &= exists
        print(f"    {'OK     ' if exists else 'MISSING'} {rel}")

    print("\n[4] Environment variables (all optional; defaults apply when unset)")
    for v in ENV_VARS:
        print(f"    {v:<12} = {os.environ.get(v, '<unset -> default>')}")

    print("\n[5] GPU / system dependencies")
    for tool, note in (("nvidia-smi", "GPU presence"),
                       ("nsys", "Nsight Systems — Exp12 only"),
                       ("vllm", "vLLM server — experiment re-runs only")):
        path = shutil.which(tool)
        print(f"    {tool:<12} {path if path else 'not found'}   ({note})")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=15).stdout.strip()
            for line in out.splitlines():
                print(f"                 {line}")
        except Exception as e:                                   # noqa: BLE001
            print(f"                 (query failed: {e})")

    print("\n" + "=" * 74)
    print(f"  STATIC CHECK          : {'PASS' if py_ok and data_ok else 'FAIL'}")
    print(f"  ANALYSIS PATH         : {'READY' if analysis_ok and data_ok else 'NOT READY'}")
    print("  RUNTIME REPRODUCTION  : NOT EXECUTED (this script never runs experiments)")
    print("=" * 74)
    return 0 if (py_ok and analysis_ok and data_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
