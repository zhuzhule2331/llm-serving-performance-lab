# Limitations

Read this before quoting any number in this repository.

---

## 1. Workload and model

| # | Limitation |
|---|---|
| 1 | **The workload is text-only serving.** The model is `Qwen3-VL-8B-Instruct`, which is multimodal, but **no images were served** in any experiment here. This is **not** a multimodal-serving result. |
| 2 | **One workload shape** — exact 2048-token prompt → exact 128-token completion. Nothing here says what happens with variable prompt or completion lengths. |
| 3 | **Fixed completion length** means `request/s` and `output tok/s` differ by a constant factor 128. They are **mathematically coupled** and must not be reported as two independent confirmations. |
| 4 | **Synchronized bursts, not a real arrival process.** "C8" means eight requests released together within a ≤25 ms skew window — not eight concurrent sessions from a Poisson stream. Queueing behaviour under real arrivals is not measured. |

## 2. Hardware and platform

| # | Limitation |
|---|---|
| 5 | **Single GPU**: one RTX 3090 (GPU0), **220 W** power limit, under WSL2. |
| 6 | **No multi-GPU, no tensor parallelism, no pipeline parallelism** was exercised. |
| 7 | **Absolute latency values are machine-specific.** Only *relative* comparisons across conditions on the same machine are meaningful. |
| 8 | **Power state matters.** All formal runs were at 220 W. Behaviour at a different cap is untested. |

## 3. Experimental scope

| # | Limitation |
|---|---|
| 9 | **Exp12 profiled three conditions, not the full 3×3 factorial** (A = C1×MBT2048, B = C8×MBT2048, C = C8×MBT4096). The mechanism reading is **converging evidence**, not a direct causal proof. |
| 10 | **One trace per profiled condition.** No repeated profiler runs, so trace-to-trace variability is not characterised. |
| 11 | **Exp12 conditions A and B lack contemporaneous power-boundary records.** Their power state is **UNKNOWN** and was deliberately **not** back-inferred from other experiments. This is an evidence-completeness gap; it does not invalidate the frozen profiler workload (client validity passed in all three conditions). |
| 12 | **Small sample**: 6 measured bursts per cell (3 blocks × 2 runs). **No p-values are claimed.** Reported instead: pooled percentage change, per-block percentage change, block range, and `all_blocks_same_sign`. |
| 13 | **Split-plot error structure.** MBT is a whole-plot factor and concurrency/reuse are sub-plot factors, so the error terms differ — a simple t-test would not be appropriate even with more samples. |
| 14 | **No scheduler-internal or source-level measurement.** Everything about *why* MBT behaves this way is an inference from client metrics plus external profiler traces. vLLM's scheduler was not instrumented. |

## 4. Claims that must not be made

| ❌ | Why |
|---|---|
| "MBT effects are **solely caused by** prefill fragmentation" | no scheduler-internal measurement; Exp12 covered 3 conditions |
| "the optimal MBT is 2048" | only measured in this workload/config; not a universal result |
| "we **proved** the mechanism" | the evidence is `associated with` / `consistent with` / `converging evidence` |
| "PHIGH's ~+104% throughput is a scheduler optimization" | it is a **prefix-cache reuse treatment effect**, not the result of tuning MBT |
| "`req/s` and `out tok/s` double-validate the result" | fixed 128-token completion ⇒ mathematically coupled |
| "this generalizes to H100 / A100 / other models" | untested |
| "this is multimodal serving" | the workload is text-only |
| "profiler numbers are serving numbers" | profiler timing ≠ serving timing |
| "Exp11-H4 was falsified" | it is **NOT ESTABLISHED** (the antecedent did not hold robustly). It was **Exp12-H4** that was falsified — the two are different hypotheses. |

## 5. Terminology that must stay hedged

| Use | Avoid |
|---|---|
| `associated with` | `caused by` |
| `consistent with` | `proved` |
| `converging evidence` | `definitively demonstrated` |
| `the trace instead showed` | `the real difference was` |
| `depends on remaining local prefill work` | `optimal` |
| `falsified for the profiled trace` | `wrong` |
| `not established` | `falsified` (specifically for **Exp11-H4**) |
| `within the tested regime, no single MBT was best` | `no configuration is universally optimal` |

## 6. What is deliberately not published

| Omitted | Reason |
|---|---|
| Raw `.nsys-rep` / `.sqlite` profiler captures | hundreds of MB; unusable without the matching GPU |
| Full server logs | large, and they embed host paths |
| Per-request raw logs (the 432-request dataset) | not needed to support any published claim; contains host paths |
| Model weights / checkpoints | not distributable |
| The original experiment host layout | private infrastructure |
