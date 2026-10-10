#!/usr/bin/env python3
"""Verify the published results against the claims made in the README and docs.

This does **not** re-run any experiment. It re-derives every headline number **from the frozen
CSV artifacts in `results/`** and asserts that it matches what the documentation says. If a
number in the docs ever drifts from the evidence, this script fails.

Usage:
    python3 scripts/verify_results.py            # exit 0 = all claims match the artifacts
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

TOL = 0.02          # percentage points — absorbs display rounding in the docs


def rows(name: str) -> list[dict]:
    return list(csv.DictReader((RES / name).open()))


class Check:
    def __init__(self) -> None:
        self.n = 0
        self.failed: list[str] = []

    def close(self, label: str, actual: float, expected: float, tol: float = TOL) -> None:
        self.n += 1
        ok = abs(actual - expected) <= tol
        print(f"    {'OK  ' if ok else 'FAIL'} {label:<62} {actual:>10.4f}  (doc: {expected})")
        if not ok:
            self.failed.append(f"{label}: artifact={actual} doc={expected}")

    def equal(self, label: str, actual, expected) -> None:
        self.n += 1
        ok = actual == expected
        print(f"    {'OK  ' if ok else 'FAIL'} {label:<62} {str(actual):>10}  (doc: {expected})")
        if not ok:
            self.failed.append(f"{label}: artifact={actual} doc={expected}")


def main() -> int:
    c = Check()
    print("=" * 88)
    print("  VERIFY PUBLISHED RESULTS AGAINST FROZEN ARTIFACTS")
    print("=" * 88)

    # ---------- Exp08 ----------
    print("\n[Exp08] C8, MBT4096 vs MBT2048   (results/exp08_relative_to_mbt2048.csv)")
    rel = rows("exp08_relative_to_mbt2048.csv")
    # 该表同时含 comparison_mbt=512 与 4096 两行 —— 必须显式选 4096,否则会误取 512。
    c8 = [r for r in rel
          if r.get("workload") == "C8" and r.get("comparison_mbt") == "4096"]
    if len(c8) != 1:
        c.failed.append(f"expected exactly one C8/4096 row, got {len(c8)}")
    for metric, doc in (("mean_tpot_ms", -4.29), ("request_throughput", 3.64), ("mean_ttft_ms", -0.11)):
        col = next((k for k in c8[0] if k.startswith(metric) and "pct" in k), None)
        if col is None:
            print(f"    SKIP {metric} — no percentage column found")
            continue
        c.close(f"C8 4096-vs-2048 {metric} (%)", float(c8[0][col]), doc)

    print("\n[Exp08] C8 per-MBT means   (results/exp08_condition_summary.csv)")
    summ = [r for r in rows("exp08_condition_summary.csv") if r["workload"] == "C8"]
    by_mbt = {int(r["mbt"]): r for r in summ}
    c.close("C8 mean TTFT @MBT4096 (ms)", float(by_mbt[4096]["mean_ttft_ms_mean"]), 2960.609, 0.01)
    c.close("C8 mean TPOT @MBT4096 (ms)", float(by_mbt[4096]["mean_tpot_ms_mean"]), 44.407, 0.01)
    c.close("C8 req/s     @MBT4096", float(by_mbt[4096]["request_throughput_mean"]), 0.925, 0.001)
    c.equal("C8 all_valid @MBT4096", by_mbt[4096]["all_valid"], "True")
    c.close("C8 preemptions @MBT4096", float(by_mbt[4096]["total_preemptions"]), 0.0, 0.0)

    # ---------- Exp11 ----------
    print("\n[Exp11] MBT4096 vs MBT512 within reuse   (results/exp11_interaction_summary.csv)")
    con = [r for r in rows("exp11_interaction_summary.csv")
           if r["from_mbt"] == "512" and r["to_mbt"] == "4096"]

    def pick(reuse: str, metric: str) -> dict:
        return next(r for r in con if r["reuse"] == reuse and r["metric"] == metric)

    for reuse, doc in (("P0", 6.88), ("P50", 5.11), ("PHIGH", 0.25)):
        c.close(f"{reuse} throughput gain (%)", float(pick(reuse, "out_tok_s")["pooled_pct_change"]), doc)
    for reuse, doc in (("P0", -5.68), ("P50", 11.11), ("PHIGH", 21.62)):
        c.close(f"{reuse} TTFT change (%)", float(pick(reuse, "ttft_ms")["pooled_pct_change"]), doc)

    c.equal("PHIGH block-consistent?",
            pick("PHIGH", "out_tok_s")["all_blocks_same_sign"].strip().lower(), "false")
    c.equal("P0 block-consistent?",
            pick("P0", "out_tok_s")["all_blocks_same_sign"].strip().lower(), "true")

    print("\n[Exp11] manipulated variable moved   (results/exp11_cell_summary.csv)")
    cells = rows("exp11_cell_summary.csv")
    p0 = [r for r in cells if r.get("reuse") == "P0"]
    ph = [r for r in cells if r.get("reuse") == "PHIGH"]
    p0_vals = sorted(float(r["compute_per_request"]) for r in p0)
    ph_vals = sorted({float(r["compute_per_request"]) for r in ph})
    c.close("P0 compute/request min (tokens)", p0_vals[0], 2020.333, 0.01)
    c.close("P0 compute/request max (tokens)", p0_vals[-1], 2024.0, 0.01)
    c.close("PHIGH compute/request (tokens)", ph_vals[0], 112.0, 0.01)
    print("         note: P0 varies 2020.3-2024.0 across MBT levels -> docs say "
          "'~2020 tokens/request' for the P0 regime, exact 2024.0 only at MBT512")

    # ---------- Exp12 ----------
    print("\n[Exp12] hypothesis status   (results/exp12_hypothesis_status.csv)")
    hyp = {r["hypothesis"]: r for r in rows("exp12_hypothesis_status.csv")}
    c.equal("Exp12-H4 status", hyp["H4"]["status"], "NOT_SUPPORTED")
    c.equal("Exp12-H2 status", hyp["H2"]["status"], "SUPPORTED")
    c.equal("Exp12-H3 status", hyp["H3"]["status"], "SUPPORTED")
    ev = hyp["H3"]["evidence"]
    m = re.search(r"from ([\d.]+) ms to ([\d.]+) ms", ev)
    c.close("final prefill marker delta (ms)", float(m.group(1)) - float(m.group(2)), 655.7, 0.1)
    s = re.search(r"spacing was ([\d.]+) ms at MBT2048 versus ([\d.]+) ms", hyp["H4"]["evidence"])
    c.close("steady spacing @MBT2048 (ms)", float(s.group(1)), 34.200, 0.001)
    c.close("steady spacing @MBT4096 (ms)", float(s.group(2)), 34.412, 0.001)
    n = re.search(r"from (\d+) at C8 MBT2048 to (\d+) at C8 MBT4096", hyp["H2"]["evidence"])
    c.equal("prepare_prefill markers @MBT2048", int(n.group(1)), 9)
    c.equal("prepare_prefill markers @MBT4096", int(n.group(2)), 5)

    # ---------- figures reproduce ----------
    print("\n[figures] regenerate from frozen data")
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "figures" / "make_figures.py")],
                       capture_output=True, text=True)
    c.equal("make_figures.py exit 0", r.returncode, 0)
    if r.returncode:
        print(r.stderr[-500:])

    print("\n" + "=" * 88)
    print(f"  checks run : {c.n}")
    print(f"  failures   : {len(c.failed)}")
    for f in c.failed:
        print(f"    - {f}")
    print(f"  RESULT     : {'PASS' if not c.failed else 'FAIL'}")
    print("=" * 88)
    return 0 if not c.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
