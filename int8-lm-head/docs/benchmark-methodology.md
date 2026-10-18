# Benchmark Methodology

This document describes exactly how the two performance numbers in this project were
obtained. It exists so that a reader can tell **what each number does and does not cover**
without reading the source.

---

## 1. Two levels, never combined

| Level | Script | Instrument | Reported metric |
|---|---|---|---|
| **Kernel** | `scripts/test_real_lm_head.py` | CUDA Events | median latency of one LM Head GEMV |
| **Serving** | `scripts/run_graph_abba.py`, `run_graph_baab.py` | HTTP client wall clock | approximate TPOT over 128 output tokens |

The kernel numbers (**1.229×–1.978× across four runs, median of the three repeats
1.303×**) and the serving number (**≈4.14%**) answer different questions.
**There is no claim that any kernel ratio becomes 4.14% through some conversion factor**,
and the two must not be substituted for one another.

---

## 2. Kernel-level microbenchmark

### 2.1 Setup

- Real `Qwen3-VL-8B-Instruct` `lm_head.weight`, shape `[151936, 4096]`, BF16 source.
- INT8 weights produced by `scripts/export_int8_lm_head.py` with per-output-row
  symmetric quantization.
- One synthetic FP32 activation vector of length 4096.
- 10 untimed warmup iterations per implementation.
- Device: **remote server GPU0, RTX 3090** (`CUDA_VISIBLE_DEVICES=0`).
  The local RTX 5060 was not used.

### 2.2 Measurement

```python
def measure(fn):
    start.record(); fn(); end.record(); end.synchronize()
    return start.elapsed_time(end)
```

20 iterations, **alternating order**:

```
i even:  BF16 → INT8
i odd:   INT8 → BF16
```

Reported value is the **median** of each 20-sample series. Alternating order is used so
that a monotonic clock ramp does not systematically favour whichever implementation runs
second.

### 2.3 Repeat protocol

The first run (20:26:16) is a full environment-capture run. It was followed by **three
shorter repeat runs** at 20:28:52, 20:28:57 and 20:29:01 — seconds apart, same machine,
same script, no code change.

Each repeat log records GPU0 temperature and free memory before the run, the identical
quantization pass, the identical accuracy trials, and the two medians.

### 2.4 Results — all four runs

| Run | Log | BF16 ms | INT8 ms | Speedup | GPU0 pre-run temp |
|---|---|---:|---:|---:|---:|
| Initial retest | `microbenchmark_real_weight_20261018_202616.txt` | 1.6502 | 1.3430 | **1.229×** | 33 °C |
| Repeat 1 | `microbenchmark_repeat_20261018/run_1.txt` | 1.5713 | 1.2058 | **1.303×** | 33 °C |
| Repeat 2 | `microbenchmark_repeat_20261018/run_2.txt` | 1.4141 | 1.0885 | **1.299×** | 43 °C |
| Repeat 3 | `microbenchmark_repeat_20261018/run_3.txt` | 1.4100 | 0.7127 | **1.978×** | 45 °C |

Derived statistics:

| Statistic | Value |
|---|---:|
| Median of the three repeats | **1.303×** |
| Median of all four | 1.301× |
| Mean of the three repeats | 1.527× |
| Range | 1.229× – 1.978× |
| Range as % of the 3-repeat median | **57.5%** |

**The 1.978× run is retained, not discarded and not promoted.** It is an outlier: the INT8
median in that run (0.7127 ms) is 34% below the next-fastest INT8 median (1.0885 ms),
while the BF16 median is essentially unchanged from repeat 2 (1.4100 vs 1.4141 ms).
A single-sided drop of that size is more consistent with a transient clock/power state
than with a stable kernel property — **but this archive does not prove that**, so the run
stays in the record.

### 2.5 Accuracy check (stable across runs)

Every run prints the same three trials (same seed) and the same CPU cross-check:

```
Trial 1: NRMSE=0.913022% MaxError=0.003076 Top1Equal=True
Trial 2: NRMSE=0.934280% MaxError=0.002947 Top1Equal=True
Trial 3: NRMSE=0.878923% MaxError=0.003885 Top1Equal=True
INT8 GPU vs CPU reference: PASS (33 rows)
```

Timing varies; numerical correctness does not. This is why the quality evaluation is
reported separately from the speedup.

### 2.6 Scope

- One shape, one activation vector, one GPU.
- Includes kernel launch overhead (CUDA Events bracket the launch).
- **Excludes** the plugin's `reshape / float / contiguous` input conversion, the
  `reshape → bfloat16` output conversion, and **offline quantization**.
- **Does not measure** end-to-end serving. The serving number is in §3.

### 2.7 Provenance

| Claim | Status |
|---|---|
| Four runs, raw logs | ✅ included in `artifacts/profiling/` |
| Environment versions | ✅ [`environment.json`](../environment.json) |
| Historical 1.645× (1.4116 → 0.8581 ms) | ⚠️ **no raw stdout in this archive** — historical record only |

The four logged runs are the authoritative kernel-level result. The historical figure must
not be quoted as the project's speedup.

### 2.8 Environment at measurement time

| Component | Version |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.13.0+cu129 |
| vLLM | 0.28.0+cu129 |
| Triton | 3.7.1 |
| CUDA runtime | 12.9.79 |
| NVCC | 12.6.85 |
| GPU driver | 572.83 |

> The NVCC / runtime mismatch (12.6 compiler, 12.9 runtime) is recorded as-is. It is one
> more reason the absolute latencies are machine-specific.

---

## 3. Serving benchmark

### 3.1 Comparators

Both comparators use the **same** server binary, model, and flags. The only difference is
`INT8_LM_HEAD_ENABLE`:

| Comparator | `INT8_LM_HEAD_ENABLE` | Path taken |
|---|---|---|
| BF16 | `0` | original `LogitsProcessor._apply_head` |
| INT8 | `1` | plugin eligibility check → custom CUDA GEMV |

This is what makes the comparison a **plugin on/off** comparison rather than a comparison
of two different builds.

### 3.2 Stage ordering

```
ABBA   A1=BF16  →  B1=INT8  →  B2=INT8  →  A2=BF16
BAAB   B1=INT8  →  A1=BF16  →  A2=BF16  →  B2=INT8
```

Each stage owns its own server process. A monotonic drift (GPU heating up, clocks
dropping) should show up as a difference between the two orderings. It does:
ABBA gives 3.49%, BAAB gives 4.78%.

### 3.3 Per-stage protocol

| Item | Value |
|---|---|
| Warmup requests | 3 |
| Measured requests | 12 |
| Output tokens | 128 (`ignore_eos=True`) |
| Sampling | `temperature=0` |
| Streaming | yes, `include_usage=True` |
| Prompt | one of three fixed Chinese CUDA/vLLM prompts, cycled by index |
| Inter-request delay | none specified in the runner; requests are issued back-to-back |
| GPU temperature guard | abort at 79 °C |

**Total measured requests: 2 runs × 4 stages × 12 = 96.**

### 3.4 Validity gates enforced by the runner

Before a stage begins collecting data, the runner asserts:

1. Port 8002 is free (no stale server).
2. The server becomes healthy within 420 s.
3. The startup log contains `Graph capturing finished` — otherwise the stage aborts.
4. For INT8 stages, the log contains `Quantized weights loaded`.
5. For BF16 stages, the log does **not** contain that string — otherwise the stage aborts.
6. Every response has exactly 128 completion tokens.

A failed gate aborts the whole run rather than silently mixing modes.

### 3.5 TPOT computation

```python
ttft = (t_first_content_chunk - t_request_start) * 1000
tpot = (t_last_content_chunk - t_first_content_chunk) * 1000 / (tokens - 1)
```

Only chunks with non-empty `choices[].delta.content` count. The request-start timestamp is
taken **before** the HTTP POST is issued, so `ttft` includes connection setup.

`tpot` deliberately excludes the first token, so it describes the **inter-token interval**,
not the prefill phase.

### 3.6 What is recorded per request

`results.csv` columns:

| Column | Meaning |
|---|---|
| `stage` | `A1_BF16`, `B1_INT8`, `B2_INT8`, `A2_BF16` |
| `mode` | `BF16` / `INT8` |
| `request` | 1-based index within the stage |
| `prompt_id` | which of the three fixed prompts |
| `tokens` | completion tokens (asserted 128) |
| `chunks` | number of non-empty content chunks received |
| `ttft_ms` | first content chunk − request start |
| `tpot_ms` | inter-token interval as defined above |
| `gpu_status` | `temperature.gpu, clocks.sm, clocks.mem` at request time |

`gpu_status` starts around 56–60 °C and rises to about 65 °C across a run; SM clock
readings fluctuate in the 1 500–1 950 MHz band. **This is why thermal drift is listed as a
limitation rather than assumed away.**

---

## 4. Why no significance test

A p-value would require an assumption about independence that the design does not support:

- Requests within a stage share one server process and one thermal state.
- The two orderings are two samples of a *process*, not 96 independent trials.
- The effect size itself differs between orderings (3.49% vs 4.78%).

The honest report is the point estimate, the per-ordering spread, and the confound list.

---

## 5. Reproducing the analysis (without re-running the GPU)

The tables in the README are computed from the archived CSVs:

```bash
python - <<'PY'
import csv, statistics as st
from pathlib import Path

rows = []
for d in ("graph_abba_20261018_180248", "graph_baab_20261018_184013"):
    with open(Path("artifacts/profiling") / d / "results.csv") as f:
        rows += list(csv.DictReader(f))

for mode in ("BF16", "INT8"):
    v = [float(r["tpot_ms"]) for r in rows if r["mode"] == mode]
    print(mode, len(v), round(st.mean(v), 4), round(st.median(v), 4))

bf = [float(r["tpot_ms"]) for r in rows if r["mode"] == "BF16"]
i8 = [float(r["tpot_ms"]) for r in rows if r["mode"] == "INT8"]
print("n =", len(rows))
print("reduction % =", round((st.mean(bf) - st.mean(i8)) / st.mean(bf) * 100, 2))
PY
```

Expected output:

```
BF16 48 25.3685 25.2698
INT8 48 24.3178 24.1236
n = 96
reduction % = 4.14
```

If your output differs, the archived CSVs were modified. They are the ground truth.
