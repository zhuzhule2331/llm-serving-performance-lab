# Exp11 — Prefix Reuse × MBT (interaction experiment)

**Role:** serving-level interaction test. Manipulate **remaining local prefill work** via prefix
reuse and re-test the prediction that came out of Exp12.

| | |
|---|---|
| **Factors** | reuse {P0 = control, P50, PHIGH} × MBT {512, 2048, 4096} |
| **Fixed** | exact 2048 prompt → exact 128 completion · concurrency = 8 · **APC ON** · chunked prefill ON · `max_num_seqs=16` · GPU0 220 W |
| **Design** | 3 randomized split-plot blocks · 9 cells · 6 measured bursts per cell |
| **Final** | 54 measured bursts · 432 measured requests · preemption = 0 |
| **Headline** | throughput gain 4096-vs-512: **+6.881% → +5.113% → +0.252%** as reuse rises; TTFT flips **−5.676% → +11.113% → +21.621%** |

> **`P0` is the control level: no intended application-level shared prefix.** Its observed hit
> ratio is ≈1.2–1.4%, not zero. Do not describe `P0` as "cache disabled".

---

## Files

| File | Purpose |
|---|---|
| `formal_runner.py` | **dry-run / planning path.** In non-dry-run mode it refuses with *"Formal execution is intentionally disabled"*. Has `--help`. |
| `formal_execute.py` | **the execution path** — requires `--execute`. This is what drove the campaign (and the Block 2 replacement). Has `--help`. |
| `prefix_core.py` | shared core: workload construction, cell execution, condition-summary building, validity |
| `namespace_map.py` | builds and freezes the namespace map (54 replicas / 54 namespaces) |
| `namespace_reconstruction_audit.py` | independent audit: rebuilds rendered lengths + shared-prefix SHA256 and compares to the frozen map |
| `finalize_merge.py` | provenance-aware merge of the replacement block into the final campaign |
| `analyze_formal.py` | block-aware analysis → `results/exp11_interaction_summary.csv` |
| `recover_block2.py` | executes the preregistered full-block replacement policy |
| `configs/` | protocol, preregistered plan, recovery policy |

`prefix_core.py` imports two upstream modules that live in [`../../common/`](../../common/):
`exp03_formal.py` and (transitively) `exp02_formal.py`. They provide
`build_shared_prefix` / `fit_exact_prompt` / `get_metrics_snapshot` and are published here
because without them this experiment's code would not run. Their **filenames are unchanged** so
that the frozen `import exp03_formal as exp03` line stays byte-identical.

---

## The Block 2 story (this is the point of the experiment's provenance)

One row failed a **hard preregistered criterion**:

```
formal_row_id = 48
launch_skew   = 88.226 ms  >  25 ms threshold
```

The other 17 rows in that block looked completely normal. **The entire block was still
discarded** and re-run under a policy that was frozen *before* the retry:

```
retry_individual_failed_row : false    ← the key line
unit_replaced               : entire logical block 2
change_thresholds           : false
change_warmup_count         : false
change_server_configuration : false
replacement_attempts_allowed: 1
fail_closed                 : true
```

Reasoning: row-level retry reintroduces degrees of freedom (when to retry, how many times, what
to do if the retry looks bad). Block-level replacement removes them.

Full policy: [`configs/exp11_block2_recovery_policy_v1.json`](configs/exp11_block2_recovery_policy_v1.json).

---

## Workload identity freezing

An arbitrary namespace changes tokenization, so "unique namespace → same rendered prefix length"
cannot be assumed. The frozen procedure:

```
54 cached replicas · 54 unique namespaces
namespace map + SHA256 frozen before formal execution
independent reconstruction audit: 54/54 rendered lengths + shared-prefix SHA256 matched
no runtime search during formal execution
```

> **Formal execution should verify a preregistered workload identity,
> not search for a workload that happens to satisfy the target.**

---

## Two hypotheses with the same name — do not confuse them

| | Status |
|---|---|
| **Exp12-H4** (larger MBT produces faster/tighter steady decode) | **FALSIFIED for the profiled trace** — steady decode spacing did not improve |
| **Exp11-H4** (if MBT4096 stays beneficial under PHIGH, MBT effects are not solely explained by uncached prefill volume) | **NOT ESTABLISHED** — the conditional premise did not hold robustly (PHIGH gain only +0.25%, not block-consistent) |

"Exp11-H4 was falsified" is a **wrong** statement.

---

## Analysis output

`analyze_formal.py` reports three numbers per contrast, because one is not enough to judge
robustness:

- `pooled_pct_change` — computed from cell means (the headline number)
- `mean_block_pct_change` — per-block percentage change, then averaged
- `min/max_block_pct_change` + `all_blocks_same_sign` — the robustness check

`PHIGH`'s +0.252% pooled, with a block range of −0.30% ~ +0.84% and `all_blocks_same_sign=false`,
is precisely the case where the pooled number alone would mislead.
