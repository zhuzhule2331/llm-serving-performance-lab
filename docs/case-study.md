# Case study — Scheduler token budget × prefix reuse

> Question → experiment → profiler → prediction → interaction → engineering implication.
> Validity and provenance details live in [`methodology.md`](methodology.md).

---

## 1. The question

`max_num_batched_tokens` (**MBT**) is the vLLM scheduler's **token budget per scheduling
iteration** — the cap on how many tokens (prompt + decode) are admitted into one scheduler step.
It is a **static server flag**: changing it requires restarting the engine.

> Is there a best MBT? If not, what does the right value depend on?

---

## 2. Exp08 — does the MBT effect depend on concurrency?

**Design.** Two-factor split-plot. MBT is the *whole-plot* factor (it needs a server restart);
concurrency is randomized *within* each MBT server.

```
Fixed   : exact 2048 prompt → exact 128 completion · APC OFF · chunked prefill ON
          max_num_seqs=16 · max_model_len=8192 · GPU0 · 220 W
Factors : MBT {512, 2048, 4096} × concurrency {C1, C2, C4, C8}
Design  : 3 randomized blocks · 108 executions (72 measured + 36 warmup) · 12 cells × 6 bursts
Validity: all 72 measured bursts PASS · preemption delta = 0
```

**Result at C8** (`results/exp08_condition_summary.csv`):

| MBT | mean TTFT (ms) | mean TPOT (ms/tok) | req/s | output tok/s |
|---:|---:|---:|---:|---:|
| 512 | 2955.186 | 48.026 | 0.845 | 108.155 |
| 2048 | 2963.758 | 46.395 | 0.892 | 114.237 |
| **4096** | **2960.609** | **44.407** | **0.925** | **118.349** |

**MBT4096 vs MBT2048 at C8** (`results/exp08_relative_to_mbt2048.csv`):
TTFT **−0.106%** · TPOT **−4.286%** · makespan **−3.525%** · req/s **+3.643%**

> **Observation.** The MBT effect **is concurrency-dependent**. At C8 a larger MBT buys ~4% TPOT
> and ~3.6% throughput at essentially no TTFT cost.
>
> **Exp08 does not explain the mechanism.** That requires profiling.

See [`../figures/fig1_exp08_throughput_vs_mbt.png`](../figures/fig1_exp08_throughput_vs_mbt.png).

---

## 3. Exp12 — what changes on the GPU?

**Design.** Nsight Systems traces of three conditions. Profiling is for **mechanism**, not for
serving numbers — profiler timing does not replace formal serving timing.

| Condition | Config |
|---|---|
| **A** | C1 × MBT2048 |
| **B** | C8 × MBT2048 |
| **C** | C8 × MBT4096 |

All three: client `VALIDITY = PASS` · preemption = 0 · nsys GPU kernels captured.
Condition C has contemporaneous power-boundary evidence (220.00 W); A/B do not.

### 3.1 The decode marker changes with concurrency

| | kernel family tracking decode | instances |
|---|---|---|
| **A (C1)** | `gemv2T` | 128 (127 intervals, mean **28.918 ms**) |
| **B/C (C8)** | `64x64` BF16 GEMM | ~128 |

> The kernel that marks decode changes with batch shape. A different kernel pattern shows a
> changed execution regime — **not** the exact bottleneck by itself.

### 3.2 B vs C — prefill structure changes

| | B: C8 × MBT2048 | C: C8 × MBT4096 |
|---|---:|---:|
| `prepare_prefill` instances | **9** | **5** |
| final marker (relative, ms) | **4531.314** | **3875.643** |
| → difference | **≈ 655.7 ms earlier for MBT4096** | |

### 3.3 B vs C — steady-state decode does *not* change

| | B: MBT2048 | C: MBT4096 |
|---|---:|---:|
| steady `64x64` spacing, mean | **34.200 ms** | **34.412 ms** |
| stdev | 1.120 ms | 1.104 ms |

> **MBT4096 did NOT make steady-state decode faster.**
> This **falsifies** the hypothesis that MBT4096 improves TPOT by accelerating decode kernels.

### 3.4 Hypothesis status (`results/exp12_hypothesis_status.csv`)

| Hypothesis | Status |
|---|---|
| Exp12-H1 (C1 MBT effect small) | **NOT DIRECTLY TESTED BY EXP12** (C1×MBT4096 not profiled) |
| Exp12-H2 (MBT2048 more fragmented at C8) | **SUPPORTED** (markers 9 vs 5) |
| Exp12-H3 (MBT4096 reaches stable decode earlier) | **SUPPORTED** (~655.7 ms earlier) |
| **Exp12-H4** (MBT4096 produces faster/tighter steady decode) | **FALSIFIED for the profiled trace** |

See [`../figures/fig4_exp12_prefill_transition.png`](../figures/fig4_exp12_prefill_transition.png).

---

## 4. Exp11 — manipulate the mechanism, re-test the behavior

**The prediction from Exp12:** if the larger-MBT benefit comes from managing **uncached local
prefill work** (fragmentation / prefill–decode interference), then **removing that work via
prefix reuse should shrink the benefit.**

So Exp11 manipulates exactly that variable.

```
Fixed   : exact 2048 prompt → exact 128 completion · concurrency = 8
          APC ON · chunked prefill ON · max_num_seqs=16 · GPU0 · 220 W
Factors : reuse {P0 = control, P50, PHIGH} × MBT {512, 2048, 4096}
Design  : 3 randomized split-plot blocks · 9 cells · 6 measured bursts per cell
Final   : 54 measured bursts · 432 measured requests · preemption = 0
```

### 4.1 The manipulated variable actually moved

| Reuse | observed hit ratio | compute/request |
|---|---:|---:|
| **P0** (control) | ≈1.2–1.4% | **≈2020 tokens** |
| **P50** | 47.66% | **1072 tokens** |
| **PHIGH** | 94.53% | **112 tokens** |

Local prefill work spans a **~18× range** across the three levels.

> `P0` is the **control level: no intended application-level shared prefix.** Its observed hit
> ratio (≈1.2–1.4%) is incidental, not a designed property — do not read it as "cache disabled".

### 4.2 The throughput-side prediction is supported

**MBT4096 vs MBT512** (`results/exp11_interaction_summary.csv`):

| Reuse | output-throughput gain | block-consistent across 3 blocks? |
|---|---:|---|
| **P0** | **+6.881%** | ✅ all same sign |
| **P50** | **+5.113%** | ✅ all same sign |
| **PHIGH** | **+0.252%** | ❌ **not consistent** (range −0.30% ~ +0.84%) |

> **As local prefill work is removed, the throughput advantage of the larger MBT largely
> disappears.**

See [`../figures/fig2_exp11_throughput_advantage_vs_reuse.png`](../figures/fig2_exp11_throughput_advantage_vs_reuse.png).

### 4.3 …but the TTFT cost does not

| Reuse | TTFT change (4096 vs 512) | block-consistent? |
|---|---:|---|
| P0 | **−5.676%** | ✅ |
| P50 | **+11.113%** | ✅ |
| PHIGH | **+21.621%** | ✅ |

> **The TTFT effect flips from a small benefit into a substantial penalty.**

See [`../figures/fig3_exp11_ttft_effect_vs_reuse.png`](../figures/fig3_exp11_ttft_effect_vs_reuse.png).

### 4.4 Therefore the interaction is **metric-dependent**

| Metric | P0 → PHIGH trend | Monotone? |
|---|---|---|
| output throughput gain | +6.88% → +5.11% → +0.25% | ✅ decreasing |
| TTFT change | −5.68% → +11.11% → +21.62% | ✅ increasing penalty |
| **TPOT change** | −1.26% → −6.38% → −1.26% | ❌ **non-monotone** |

> Reporting "MBT × reuse interaction" without naming the metric would be incorrect.

### 4.5 Exp11 hypothesis verdicts

| Hypothesis | Status | Why |
|---|---|---|
| Exp11-H1 (more reuse → lower TTFT) | **SUPPORTED** | PHIGH vs P0: −87.56% at MBT4096 |
| Exp11-H2 (P0 high-C resembles prior behavior) | **PARTIALLY SUPPORTED** | larger MBT still gives throughput benefit, but MBT4096 does not universally dominate MBT2048 on latency |
| Exp11-H3 (more reuse → smaller larger-MBT advantage) | **SUPPORTED**, esp. throughput & TTFT | +6.88% → +0.25% |
| **Exp11-H4** (if MBT4096 remains beneficial under PHIGH, MBT effects are not solely explained by uncached prefill volume) | **NOT ESTABLISHED** | the conditional premise was **not robustly met**: PHIGH gain is only +0.25% and **not block-consistent** |

> ⚠️ **Exp11-H4 is NOT ESTABLISHED — not FALSIFIED.** The antecedent did not hold robustly, so
> the implication was never properly tested.
> **Do not confuse this with Exp12-H4**, which *was* falsified (steady decode did not get faster).

---

## 5. Joint mechanism conclusion

> Exp08, Exp12, and Exp11 provide **converging evidence** that the high-concurrency benefit of a
> larger scheduler token budget is **associated with managing substantial local prefill work** —
> reducing prefill fragmentation and prefill/decode interference — **rather than accelerating
> steady-state decode itself.**
>
> As prefix reuse removes most local prefill work, the throughput benefit of a larger MBT
> **largely disappears**.

This is `consistent with` / `associated with` / `converging evidence`. It is **not** a proof:

- Exp12 profiled only 3 conditions
- Exp11 did not profile the full 3×3 factorial
- No scheduler-internal or source-level evidence exists

---

## 6. Engineering implication

**Workload-aware serving configuration.** The same model and GPU can call for different MBT
settings:

- **Heavy uncached prefill** (long prompts, low cache hit) → a larger MBT pays off
- **Mostly cached prefill** (high prefix reuse) → **no robust throughput benefit, while TTFT
  worsened**

A static MBT setting will therefore be wrong under some traffic mix. The measurement
infrastructure here is the piece needed to decide *when to change it*.

## 7. What this does **not** establish

| ❌ | Why |
|---|---|
| "MBT effects are solely caused by prefill fragmentation" | no scheduler-internal measurement; Exp12 has 3 conditions only |
| "the optimal MBT is 2048" (universally) | only measured in this workload/config |
| "PHIGH's ~+104% throughput is a scheduler optimization" | it is a **prefix-cache reuse treatment effect** |
| "request/s and output tok/s are two independent confirmations" | fixed 128-token completion ⇒ mathematically coupled |
| "this generalizes to other GPUs / models / traffic" | not tested; a single 220 W RTX 3090 |
| "this is multimodal serving" | the workload is **text-only**; the model merely happens to be multimodal |
| "profiler numbers are serving numbers" | profiler timing ≠ serving timing |
