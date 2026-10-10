# Experiment map

Which file does what, and which frozen artifact each number comes from.

---

## Exp08 — MBT × Concurrency (behavioral)

| | |
|---|---|
| **Role** | behavioral observation — *does the MBT effect depend on concurrency?* |
| **Config** | MBT {512, 2048, 4096} × concurrency {C1, C2, C4, C8}; exact 2048 → 128; APC **OFF**; chunked prefill ON; `max_num_seqs=16`; `max_model_len=8192`; GPU0 220 W |
| **Scale** | 3 randomized blocks · 108 executions (72 measured + 36 warmup) · 12 cells × 6 bursts |
| **Validity** | all 72 measured bursts PASS · preemption delta 0 |

| File | Purpose |
|---|---|
| `experiments/exp08_mbt_concurrency/runner.py` | formal campaign driver: split-plot order, server lifecycle calls, validity gating, **and aggregation into the two summary CSVs** |
| `experiments/exp08_mbt_concurrency/preflight.py` | pre-campaign environment / server checks |
| `experiments/exp08_mbt_concurrency/configs/exp08_formal_protocol.txt` | frozen validity + interpretation rules |
| `experiments/exp08_mbt_concurrency/configs/exp08_formal_preregistered_order.txt` | frozen block/MBT execution order |
| `common/server_lifecycle.py` | start / health-check / stop the vLLM server, capture `server_cmdline.txt` |
| `common/request_client.py` | the synchronized-burst client: TTFT / TPOT / ITL, token-count validation, launch-skew and preemption checks |

**Result artifacts**

| Artifact | Content |
|---|---|
| `results/exp08_condition_summary.csv` | per-cell means: TTFT, TPOT, makespan, request & output throughput, validity, preemptions |
| `results/exp08_relative_to_mbt2048.csv` | per-cell percentage change vs MBT2048 for each metric |

---

## Exp12 — Decode profiling (profiler localization)

| | |
|---|---|
| **Role** | mechanism localization — *what changes on the GPU?* |
| **Conditions** | A = C1×MBT2048 · B = C8×MBT2048 · C = C8×MBT4096 |
| **Tool** | Nsight Systems 2024.5.1, trace `cuda,nvtx,osrt`, `CuptiUseRawGpuTimestamps=false` (WSL workaround) |
| **Validity** | client `VALIDITY = PASS` · preemption 0 · kernels captured, all three conditions |

| File | Purpose |
|---|---|
| `experiments/exp12_decode_profiling/analyze_timeline.py` | reads the nsys-exported per-condition SQLite, counts `prepare_prefill` markers, measures `64x64` steady spacing, emits the timeline summary |
| `experiments/exp12_decode_profiling/configs/exp12_protocol.txt` | frozen protocol |
| `experiments/exp12_decode_profiling/profiling_commands.md` | the exact `nsys profile` invocations and export steps (**raw `.nsys-rep` / `.sqlite` are not published**) |

**Result artifacts**

| Artifact | Content |
|---|---|
| `results/exp12_hypothesis_status.csv` | machine-readable H1–H4 verdicts + the exact numeric evidence string |
| `results/exp12_timeline_analysis.txt` | per-condition marker counts and intervals |
| `results/exp12_summary.txt` | human-readable timeline summary |
| `results/exp12_C_power_boundary.txt` | contemporaneous 220.00 W record for condition C |

---

## Exp11 — Prefix Reuse × MBT (interaction)

| | |
|---|---|
| **Role** | serving-level interaction test — *manipulate remaining local prefill work and re-test* |
| **Config** | reuse {P0 = control, P50, PHIGH} × MBT {512, 2048, 4096}; concurrency = 8; exact 2048 → 128; APC **ON**; chunked prefill ON; `max_num_seqs=16`; GPU0 220 W |
| **Scale** | 3 randomized split-plot blocks · 9 cells · 6 measured bursts per cell |
| **Final** | 54 measured bursts · 432 measured requests · preemption 0 |

| File | Purpose |
|---|---|
| `experiments/exp11_prefix_mbt/formal_runner.py` | formal campaign driver for the Exp11 factorial |
| `experiments/exp11_prefix_mbt/prefix_core.py` | shared core: workload construction, cell execution, condition-summary building, validity |
| `experiments/exp11_prefix_mbt/namespace_map.py` | builds and freezes the namespace map (54 replicas / 54 namespaces) |
| `experiments/exp11_prefix_mbt/namespace_reconstruction_audit.py` | independent audit: rebuilds rendered lengths + shared-prefix SHA256 and compares against the frozen map |
| `experiments/exp11_prefix_mbt/finalize_merge.py` | provenance-aware merge of the replacement block into the final campaign |
| `experiments/exp11_prefix_mbt/analyze_formal.py` | block-aware analysis: pooled change, block mean change, block range, `all_blocks_same_sign` |
| `experiments/exp11_prefix_mbt/recover_block2.py` | executes the preregistered full-block replacement policy |
| `experiments/exp11_prefix_mbt/configs/` | protocol, preregistered plan, recovery policy |

**Result artifacts**

| Artifact | Content |
|---|---|
| `results/exp11_cell_summary.csv` | per-cell summary incl. `compute_per_request` and observed hit ratio |
| `results/exp11_interaction_summary.csv` | MBT-within-reuse contrasts: pooled %, block mean, block range, `all_blocks_same_sign` |

---

## Reading order

```
results/exp08_condition_summary.csv
        → results/exp08_relative_to_mbt2048.csv
        → results/exp12_hypothesis_status.csv   (the falsification)
        → results/exp11_cell_summary.csv
        → results/exp11_interaction_summary.csv (the interaction)
        → figures/
```

## Note on code sharing

`common/request_client.py` and `common/server_lifecycle.py` are byte-identical across the three
experiments in the original archive (the same frozen files were copied into each experiment
directory). They are published **once**, under `common/`, rather than triplicated.

Exp11's server-side runner in the original archive is also a copy of Exp08's runner; the
public copy publishes it once as `experiments/exp08_mbt_concurrency/runner.py`.
