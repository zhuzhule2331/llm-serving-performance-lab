# Results

**Every file here is derived from the frozen formal evidence of the original campaign.**
They are **aggregates**, not raw per-request data.

> ⚠️ These are **not** the original raw logs. Do not present them as raw data.
> The raw per-request logs (432 requests for Exp11, 72 measured bursts for Exp08) and the raw
> profiler captures are deliberately **not published** — see [`../../docs/limitations.md`](../docs/limitations.md) §6.

---

## Files

| File | Origin | Content |
|---|---|---|
| `exp08_condition_summary.csv` | frozen campaign output | per-cell means: TTFT, TPOT, makespan, request & output throughput, `all_valid`, `total_preemptions`, launch-skew stats |
| `exp08_relative_to_mbt2048.csv` | frozen campaign output | per-cell percentage change vs MBT2048 for each metric |
| `exp11_cell_summary.csv` | merged final campaign | per-cell summary incl. `compute_per_request` and observed cache hit ratio |
| `exp11_interaction_summary.csv` | merged final campaign | MBT-within-reuse contrasts: `pooled_pct_change`, `mean_block_delta`, block min/max, `all_blocks_same_sign` |
| `exp12_hypothesis_status.csv` | Exp12 analysis output | machine-readable H1–H4 verdicts **plus the numeric evidence string** used to plot `fig4` |
| `exp12_timeline_analysis.txt` | Exp12 analysis output | per-condition `prepare_prefill` marker counts and intervals |
| `exp12_summary.txt` | Exp12 analysis output | human-readable timeline summary |
| `exp12_C_power_boundary.txt` | recorded during profiling | contemporaneous power-limit record for condition C |
| `artifact_hashes.json` | frozen provenance manifest | SHA256 of the Exp11 core artifacts |

---

## Provenance caveats that apply to everything in this directory

| # | Caveat |
|---|---|
| 1 | **Exp11's final campaign merges a replacement block.** The original Block 2 is permanently excluded; the shipped Block 2 comes from the single preregistered full-block replacement. See `../experiments/exp11_prefix_mbt/configs/exp11_block2_recovery_policy_v1.json`. |
| 2 | **Exp12 conditions A and B have no contemporaneous power-boundary record.** Their power state is `UNKNOWN` and was not back-inferred. Only condition C has a record. |
| 3 | **Exp11 power/validity**: `P0` is the **control level** (no intended application-level shared prefix). Its observed hit ratio is ≈1.2–1.4%, not zero. |
| 4 | **`request/s` and `output tok/s` are mathematically coupled** (fixed 128-token completion). Reading both as independent confirmations is wrong. |
| 5 | Aggregates were produced by the original campaign scripts, not recomputed here. The public copy of `analyze_formal.py` regenerates the Exp11 tables from a merged campaign directory. |

---

## Regenerating the figures

```bash
python3 figures/make_figures.py
```

The script reads only the CSVs above and **asserts** the expected values, so a mismatch fails
loudly instead of silently producing a wrong chart.
