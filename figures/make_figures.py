#!/usr/bin/env python3
"""Generate the figures shipped in this repository.

**Every plotted value is read from a frozen result artifact under `results/`.**
No number is typed in by hand; the script fails loudly on a mismatch instead of
silently producing a wrong chart.

Sources
-------
fig1  results/exp08_condition_summary.csv
fig2  results/exp11_interaction_summary.csv   (metric=out_tok_s, 512 -> 4096)
fig3  results/exp11_interaction_summary.csv   (metric=ttft_ms,  512 -> 4096)
fig4  results/exp12_hypothesis_status.csv     (machine-readable H2/H3/H4 evidence)

Usage
-----
    python3 figures/make_figures.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent

MBT_ORDER = [512, 2048, 4096]
REUSE_ORDER = ["P0", "P50", "PHIGH"]
COLORS = [  "#1c3c6e", "#e07b39", "#3f7d5a", "#b23a48"]
BLUE, ORANGE, GREEN, RED = COLORS


def _contrasts() -> list[dict]:
    rows = list(csv.DictReader((RES / "exp11_interaction_summary.csv").open()))
    return [r for r in rows if r["from_mbt"] == "512" and r["to_mbt"] == "4096"]


def fig1() -> None:
    rows = list(csv.DictReader((RES / "exp08_condition_summary.csv").open()))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    for i, wl in enumerate(["C1", "C2", "C4", "C8"]):
        pts = sorted((int(r["mbt"]), float(r["output_throughput_mean"]))
                     for r in rows if r["workload"] == wl)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=wl,
                color=COLORS[i], lw=1.8, ms=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(MBT_ORDER)
    ax.set_xticklabels([str(m) for m in MBT_ORDER])
    ax.set_xlabel("max_num_batched_tokens (MBT)")
    ax.set_ylabel("Output throughput (tok/s)")
    ax.set_title("Exp08 — throughput vs MBT, grouped by concurrency", fontsize=11)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(title="concurrency", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_exp08_throughput_vs_mbt.png")
    plt.close(fig)


def fig2() -> None:
    cr = _contrasts()
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=160)
    vals, cons = [], []
    for r in REUSE_ORDER:
        row = next(x for x in cr if x["reuse"] == r and x["metric"] == "out_tok_s")
        vals.append(float(row["pooled_pct_change"]))
        cons.append(row.get("all_blocks_same_sign", "").strip().lower() == "true")
    bars = ax.bar(REUSE_ORDER, vals, color=[BLUE if c else RED for c in cons],
                  edgecolor="black", linewidth=0.6, width=0.55)
    for b, v, c in zip(bars, vals, cons):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:+.2f}%",
                ha="center", fontsize=9, fontweight="bold")
        if not c:   # 固定放在轴底部,避免与矮柱重叠
            ax.text(b.get_x() + b.get_width() / 2, 0.06, "not block-\nconsistent",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.5, color=RED)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-1.6, max(vals) * 1.35)
    ax.set_ylabel("Throughput gain (4096 vs 512, %)")
    ax.set_xlabel("Prefix-reuse level")
    ax.set_title("Exp11 — larger-MBT throughput advantage shrinks as reuse grows", fontsize=11)
    ax.grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_exp11_throughput_advantage_vs_reuse.png")
    plt.close(fig)


def fig3() -> None:
    cr = _contrasts()
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=160)
    vals = [float(next(x for x in cr if x["reuse"] == r and x["metric"] == "ttft_ms")
                  ["pooled_pct_change"]) for r in REUSE_ORDER]
    bars = ax.bar(REUSE_ORDER, vals, color=[GREEN if v < 0 else RED for v in vals],
                  edgecolor="black", linewidth=0.6, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.6 if v > 0 else -1.9),
                f"{v:+.2f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(min(vals) - 5, max(vals) * 1.25)
    ax.set_ylabel("TTFT change (4096 vs 512, %)")
    ax.set_xlabel("Prefix-reuse level")
    ax.set_title("Exp11 — the TTFT effect flips from a small benefit to a penalty", fontsize=11)
    ax.grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_exp11_ttft_effect_vs_reuse.png")
    plt.close(fig)


def fig4() -> None:
    rows = {r["hypothesis"]: r for r in
            csv.DictReader((RES / "exp12_hypothesis_status.csv").open())}
    ev = " ".join(r["evidence"] for r in rows.values())
    m = re.search(r"from (\d+) at C8 MBT2048 to (\d+) at C8 MBT4096", ev)
    n2048, n4096 = int(m.group(1)), int(m.group(2))
    t = re.search(r"from ([\d.]+) ms to ([\d.]+) ms", ev)
    t2048, t4096 = float(t.group(1)), float(t.group(2))
    s = re.search(r"spacing was ([\d.]+) ms at MBT2048 versus ([\d.]+) ms", ev)
    s2048, s4096 = float(s.group(1)), float(s.group(2))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=160)
    ax = axes[0]
    ax.bar(["MBT2048", "MBT4096"], [n2048, n4096], color=[BLUE, ORANGE],
           edgecolor="black", linewidth=0.6, width=0.5)
    for i, v in enumerate([n2048, n4096]):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("prepare_prefill instances (C8)")
    ax.set_title(f"Prefill markers {n2048} → {n4096}", fontsize=10)
    ax.set_ylim(0, max(n2048, n4096) * 1.25)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax = axes[1]
    ax.bar(["MBT2048", "MBT4096"], [s2048, s4096], color=[BLUE, ORANGE],
           edgecolor="black", linewidth=0.6, width=0.5)
    for i, v in enumerate([s2048, s4096]):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("steady 64×64 spacing (ms)")
    ax.set_ylim(min(s2048, s4096) - 0.35, max(s2048, s4096) + 0.35)
    ax.set_title("Steady-state decode: essentially unchanged", fontsize=10)
    ax.grid(axis="y", alpha=0.25, ls=":")
    fig.suptitle(f"Exp12 — prefill transitions earlier ({t2048:.1f} → {t4096:.1f} ms, "
                 f"≈{t2048 - t4096:.1f} ms earlier) without accelerating steady decode",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_exp12_prefill_transition.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("✅ figures written to", OUT)
