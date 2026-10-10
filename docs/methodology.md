# Methodology — measurement discipline and provenance

> What was actually controlled, and how failure was handled.
> Not a statistics textbook; these are the practices this project executed.

---

## 1. Two tracks

```
Measure → Validate → Locate → Verify → Study Interactions
                │
        Raw Evidence → Validity → Interpretation
```

| Stage | In this project |
|---|---|
| **Measure** | client-observed TTFT / TPOT / makespan / throughput (Exp08, Exp11) |
| **Validate** | preregistered criteria + fail-closed + full-block replacement |
| **Locate** | Exp12 nsys traces |
| **Verify** | Exp11 manipulates the variable Exp12 implicated |
| **Study Interactions** | reuse × MBT factorial |

**Core inequality:**

```
Measurement Correctness  ≠  Experimental Validity
```

---

## 2. Preregistration

Criteria and execution plans were frozen **before** data collection and could not be re-tuned
after seeing results.

| Experiment | Frozen artifacts (shipped under `experiments/*/configs/`) |
|---|---|
| Exp08 | `exp08_formal_protocol.txt`, `exp08_formal_preregistered_order.txt` |
| Exp11 | `exp11_protocol.txt`, `exp11_preregistered_plan.csv`, `exp11_block2_recovery_policy_v1.json` |
| Exp12 | `exp12_protocol.txt` |

Verbatim discipline taken from the protocols:

> - Do not redefine validity after seeing formal results.
> - A validity failure invalidates the affected block.
> - Do not selectively retain valid-looking runs from an invalid block.
> - Output throughput and request throughput are **not independent evidence**
>   because completion length is fixed.

---

## 3. Randomized / blocked design

MBT is a **whole-plot** factor — changing it requires restarting the vLLM server. Concurrency and
reuse are **sub-plot** factors randomized independently within each MBT server.

```
Exp08 : 3 blocks · MBT order [4096,2048,512] / [2048,512,4096] / [512,4096,2048]
Exp11 : 3 blocks, same split-plot structure
```

**Why this matters.** If conditions were run in block order, condition and wall-clock time would
be perfectly collinear, and hardware drift would masquerade as a condition effect — impossible to
separate after the fact.

---

## 4. Warmup exclusion

```
per condition : 1 excluded warmup + 2 measured bursts
Exp08 : 36 warmup (excluded) + 72 measured
Exp11 : 27 warmup (excluded) + 54 measured
```

Protocol wording: *"Pilot / diagnostic results are not formal performance evidence."*

---

## 5. Contemporaneous hardware boundary

Power limits were recorded **while the condition ran**, not looked up afterwards.

| Experiment | Record |
|---|---|
| Exp08 | per-MBT `power_boundary.json`; protocol requires `GPU0 power limit = 220 W` |
| Exp11 | protocol requires `power limit = 220 W` as a validity item |
| Exp12 | ✅ condition C has `results/exp12_C_power_boundary.txt` = 220.00 W<br>❌ **A/B have no equivalent contemporaneous record → marked UNKNOWN** |

> ⚠️ The A/B power limit must **not** be back-inferred from other experiments. The missing record
> is an **evidence-completeness** gap; it does **not** invalidate the already-frozen Exp12
> profiler workload (all three conditions passed client validity).

---

## 6. Launch skew

```
launch skew = latest request start − earliest request start (within one synchronized burst)
threshold   = 25 ms
```

**Why it is a hard criterion.** With poor skew, different conditions have different *actual*
overlap, so the independent variable is no longer what it claims to be.

Measured: all Exp08 cells within threshold; Exp11 within threshold except the one failed row
noted below.

---

## 7. Preemption

```
method : before / after /metrics snapshots per condition;
         take the delta of num_preemptions_total
why    : that counter is process-lifetime cumulative and resets on restart,
         so an absolute value is meaningless
require: preemption delta = 0
```

Result: Exp08 / Exp11 final campaign / Exp12 → **0** in all conditions.

---

## 8. Fail closed

Any failure against a preregistered criterion invalidates the affected unit. No exceptions.

**The important instance here:** Exp11 had **one row** fail a hard criterion
(`formal_row_id = 48`, `launch_skew = 88.226 ms > 25 ms`). The other 17 rows in that block
looked completely normal — **the entire block was still marked INVALID.**

---

## 9. Full-block replacement

**The replacement unit is a block, not a row**, fixed in advance in the recovery policy
(`experiments/exp11_prefix_mbt/configs/exp11_block2_recovery_policy_v1.json`):

```
unit_replaced                        : entire logical block 2
discard_original_block2_from_formal  : true
retry_individual_failed_row          : false   ← the key line
change_thresholds                    : false
change_warmup_count                  : false
change_server_configuration          : false
reuse_original_frozen_order          : true
reuse_original_frozen_namespace_map  : true
reuse_original_workload_identity     : true
replacement_attempts_allowed         : 1
fail_closed                          : true
```

**Why not re-run only the failed row?** Row-level retry reintroduces degrees of freedom — when to
retry, how many times, what to do if the retry looks bad. Block-level replacement removes them.

---

## 10. Workload identity freezing

**The finding:** an arbitrary namespace changes tokenization, so
"unique namespace → same rendered prefix length" cannot be assumed.

**The procedure:**

```
54 cached replicas · 54 unique namespaces
namespace map + SHA256 frozen before formal execution
independent reconstruction audit: 54/54 rendered lengths and shared-prefix SHA256 matched
no runtime search during formal execution
```

> **Formal execution should verify a preregistered workload identity,
> not search for a workload that happens to satisfy the target.**

---

## 11. Provenance

| Carrier | Content |
|---|---|
| `results/*.csv` | frozen aggregates derived from the formal campaign |
| `experiments/*/configs/` | protocols, preregistered plans, recovery policy |
| `results/exp12_hypothesis_status.csv` | per-hypothesis verdict + evidence string |
| `../PUBLIC_CODE_CHANGELOG.md` | every change made when preparing this public copy |

**Known provenance caveats (carried forward, not hidden):**

| # | Caveat |
|---|---|
| PC-1 | Exp11's original Block 2 is permanently excluded; the final campaign's Block 2 comes from the single preregistered replacement |
| PC-2 | Exp12 conditions A/B lack contemporaneous power-boundary records → **UNKNOWN**, not back-inferred |
| PC-3 | Exp08's original logs carried a stale display label in the reused burst client; workload identity is determined from the actual experiment files and configuration (raw logs unmodified) |
| PC-4 | `results/` are **derived aggregates**, not raw per-request logs |
| PC-5 | Raw profiler captures and full server logs are **not** published (size, and they embed host paths) |

---

## 12. Boundaries of this methodology

| # | Boundary |
|---|---|
| 1 | **Small sample**: 6 measured bursts per cell → **no p-value claims**. Block-level sign consistency + pooled percentage + block range are reported instead. |
| 2 | **Split-plot error structure** limits the applicability of simple significance tests. |
| 3 | **Single workload / single model / single GPU / single power point.** |
| 4 | **Synchronized bursts ≠ a real arrival process.** |
| 5 | Exp11 did not profile the full factorial → mechanism reading is converging evidence. |
| 6 | Exp12 A/B lack contemporaneous power-boundary records. |
