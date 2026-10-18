# INT8 LM Head — Weight-Only Quantization + Custom CUDA GEMV for vLLM Decode

A single-token decode-path optimization study for `Qwen3-VL-8B-Instruct` on vLLM:
replace the BF16 LM Head projection with a **per-output-row symmetric INT8 weight**
representation and a hand-written **CUDA GEMV V2** kernel, then measure what actually
changes — at the kernel level and at the serving level.

> **Headline, with its scope attached:** across **four independent microbenchmark runs**
> the shape-specific kernel speedup ranged from **1.229× to 1.978×** — median of the three
> repeat runs **≈1.303×**. The spread is large and is reported in full; **the fastest run is
> not used as the representative result.** A separate 96-request ABBA/BAAB client-side serving
> benchmark shows a **≈4.14% mean approximate-TPOT reduction** under CUDA Graph + `torch.compile`.
> **These are different metrics.** Neither one is a whole-model speedup, and the 4.14%
> figure is an observation with known timing and thermal confounds, not a proven
> statistically significant gain.

---

## 1. Project Overview

The LM Head is the last projection in the decode step: `hidden [4096] → logits [151936]`.
With batch size 1 it is a **matrix–vector** product, so it is bound by how many bytes of
weight it must read, not by arithmetic throughput. In BF16 the weight is ≈1.16 GiB and is
read on every decode step.

This project asks a narrow engineering question:

> If the LM Head weight is stored as INT8 and dequantized inside a custom GEMV kernel,
> how much does the kernel itself speed up, how much of that survives into end-to-end
> serving, and what does it cost in output quality?

What was built:

| Piece | What it is |
|---|---|
| **Quantizer** | per-output-row symmetric INT8, `scale[r] = max_abs(row_r) / 127` |
| **CUDA kernel** | warp-per-row GEMV with warp-shuffle reduction, FP32 accumulation (V1 → V2) |
| **PyTorch extension** | C++/pybind11 binding with dtype / shape / device / contiguity checks |
| **vLLM integration** | a `vllm.general_plugins` entry point that patches `LogitsProcessor._apply_head`, with a strict eligibility check and BF16 fallback |
| **Benchmark harness** | ABBA / BAAB stage runner, client-side streaming TPOT, per-request GPU temperature and clock logging |
| **Quality suite** | 22 objective questions + 13 deduplicated open-ended questions with blind review |

**What this is not:** it is not an INT8 GEMM, not a Tensor-Core INT8 kernel, not a batched
or prefill optimization, and not an upstream vLLM feature. The activation stays FP32 and
the arithmetic stays FP32 — INT8 applies to the **weight storage only**.

---

## 2. Hardware & Software Environment

### Hardware

| Role | Device | Used for measurement? |
|---|---|---|
| **Remote server** | NVIDIA **RTX 3090 × 2**, 24 GB each | **GPU0 only** |
| Local development machine | NVIDIA RTX 5060 | **No** — local agent / editing only |

> ⚠️ **Every performance and quality number in this repository comes from the remote
> server's GPU0 (a single RTX 3090).** GPU1 did not participate in the reported
> comparisons. No result here is from the RTX 5060.

### Software

| Component | Version |
|---|---|
| vLLM | **0.28.0** |
| Model | `Qwen3-VL-8B-Instruct` (BF16 weights) |
| CUDA target | `sm_86` (extension is built as `int8_gemv_sm86_ext`) |
| OS | WSL2 Linux on the remote server |
| Python | **3.12.14** |
| PyTorch | **2.13.0+cu129** |
| Triton | **3.7.1** |
| CUDA Runtime | **12.9.79** |
| NVCC | **12.6.85** (`release 12.6, V12.6.85`) |
| GPU driver | **572.83** |
| Nsight Systems | **not recorded** |

Source: [`environment.json`](environment.json), captured on the server.

> ⚠️ **CUDA toolkit mismatch, recorded as-is:** the runtime ships with CUDA **12.9.79**, but
> `nvcc` reports **12.6.85**. The extension was therefore compiled by a 12.6 toolchain and
> loaded into a 12.9 runtime. This is exactly the kind of detail that can change ABI or
> kernel-selection behaviour, which is one reason the microbenchmark numbers are treated as
> machine-specific rather than portable.
>
> Only the Nsight Systems version is still missing from the archive.

### Serving configuration used for the ABBA/BAAB runs

```
vllm serve $QWEN_MODEL_DIR
  --served-model-name qwen-int8
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len 2048
  --max-num-batched-tokens 2048
  --max-num-seqs 8
  --gpu-memory-utilization 0.85
```

Environment: `CUDA_VISIBLE_DEVICES=0`, `VLLM_PLUGINS=qwen_int8_head`,
`INT8_LM_HEAD_ENABLE=0|1`, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `HF_HUB_OFFLINE=1`.

---

## 3. Kernel Design

### Data layout

| Tensor | Shape | Dtype |
|---|---|---|
| `Wq` (quantized weight) | `[151936, 4096]` | `int8` |
| `scales` | `[151936]` | `float32` |
| `x` (activation) | `[4096]` | `float32` |
| `y` (output) | `[151936]` | `float32` |

`y[r] = scales[r] * Σ_j Wq[r, j] * x[j]`

### Quantization scheme

Per **output row**, symmetric around zero:

```python
max_abs = w.abs().amax(dim=1)
scale   = max_abs / 127.0            # 1.0 where max_abs == 0
q       = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
```

- One FP32 scale per output row (151 936 scales, ≈594 KiB total).
- The range is `[-127, 127]`; `-128` is intentionally unused.
- Dequantization happens **inside** the kernel at write-back time (`sum * scales[row]`),
  so no separate dequantized BF16 weight ever exists in VRAM.

### V1 — one warp per block

```
grid  = M          block = 32
row   = blockIdx.x
lane  = threadIdx.x
for j = lane; j < N; j += 32: sum += float(Wq[row*N + j]) * x[j]
warp-shuffle reduction → lane 0 writes y[row] = sum * scales[row]
```

Correct, but launches 151 936 tiny blocks and leaves most SM resources idle per block.

### V2 — four warps per block (the version used for all measurements)

```
grid  = ceil(M / 4)     block = 128
warp_id = threadIdx.x / 32
lane    = threadIdx.x % 32
row     = blockIdx.x * 4 + warp_id
```

Each warp still owns exactly one output row, so the reduction is unchanged; packing four
rows into a block cuts the block count 4× and improves per-SM occupancy.

### Why this can be faster at all

The activation is FP32 and the multiply–accumulate is FP32, so there is **no INT8
arithmetic-throughput advantage**. The win comes from **weight bytes read**:

- BF16 weight: 151 936 × 4096 × 2 B ≈ **1.16 GiB** per decode step
- INT8 weight: 151 936 × 4096 × 1 B ≈ **0.58 GiB** per decode step

At batch size 1, the LM Head is weight-bandwidth-bound, so halving the bytes is the
mechanism. The trade-off is a per-element `int8 → float` conversion in the inner loop.

### Correctness check

`test_extension.py` compares the kernel against a PyTorch reference on synthetic tensors;
`test_real_lm_head.py` does the same on the real `[151936, 4096]` weight and additionally
cross-checks 33 sampled rows against a CPU reference (`atol=1e-3, rtol=1e-4`) before
running the microbenchmark.

---

## 4. vLLM Integration

The integration is a **plugin**, not a fork. `setup.py` declares:

```python
entry_points={"vllm.general_plugins": ["qwen_int8_head=int8_vllm_plugin:register"]}
```

`setup.py` and `int8_vllm_plugin.py` sit **at the project root**, because both
`py_modules=["int8_vllm_plugin"]` and the plugin's own `Path(__file__).parent` data paths
require that layout.

**Packaging status** (checked statically on 2026-10-18):

| Check | Result |
|---|---|
| `python setup.py --name` / `--version` | ✅ `qwen-int8-lmhead-experiment` / `0.1.0` |
| `py_modules` resolves the file | ✅ `egg_info` lists `int8_vllm_plugin.py` in `SOURCES.txt` |
| Entry point emitted | ✅ `qwen_int8_head = int8_vllm_plugin:register` |
| Module imports, `register` callable | ✅ |
| Plugin data path resolves to project root | ✅ `<root>/artifacts/lm_head_int8_rowwise.safetensors` |
| **Actual install into a vLLM env + loader invocation** | ❌ **NOT VERIFIED** — vLLM is not installed on the preparation machine |

> ⚠️ **Install mode matters.** The plugin resolves its data files relative to
> `Path(__file__).parent`:
>
> - `artifacts/lm_head_int8_rowwise.safetensors`
> - `build/` (the compiled extension)
>
> A **non-editable** `pip install .` copies `int8_vllm_plugin.py` into `site-packages/`,
> so those paths resolve inside `site-packages/` and **will not be found**.
> Use an **editable install from the source directory** instead:
>
> ```bash
> cd int8-lm-head
> pip install -e .
> ```
>
> This fixes **only the Python module location**. It does **not** build the CUDA extension
> (see §9.3), and it does **not** provide `artifacts/lm_head_int8_rowwise.safetensors`
> (see §9.4). Both remain prerequisites, and **neither is shipped**.

`register()` returns immediately unless `INT8_LM_HEAD_ENABLE=1`, so the same environment
can serve either the baseline or the INT8 path.

The patch targets `LogitsProcessor._apply_head` and only takes the custom path when **all**
of the following hold:

| Condition | Why |
|---|---|
| `hidden_states.is_cuda` | kernel requires CUDA |
| `dtype == torch.bfloat16` | matches the model's head dtype |
| `shape[-1] == 4096` and `numel() == 4096` | single-token decode only |
| `lm_head.weight.shape == (151936, 4096)` | kernel is shape-specialized |
| `lm_head.tp_size == 1` | no tensor parallelism |
| `embedding_bias is None` | kernel has no bias term |
| `isinstance(quant_method, UnquantizedEmbeddingMethod)` | do not stack on another quantizer |
| `head_dtype is None or == hidden dtype` | no mixed head dtype |

Anything else falls through to the original BF16 implementation. On the first eligible
call the INT8 weights are lazily loaded from a safetensors artifact and cached on the
`lm_head` module.

The forward path is:

```
hidden [1, 4096] bf16
  → reshape(-1).float().contiguous()      # [4096] fp32
  → int8_gemv_sm86_ext.forward(x, Wq, scales)
  → reshape(..., 151936).to(bfloat16)
```

Diagnostics are opt-in via `INT8_LM_HEAD_TRACE`, `INT8_LM_HEAD_PROFILE` (CUDA Events
around the three sub-steps) and `INT8_LM_HEAD_NSYS` (`cudaProfilerStart` / `Stop` around
decode calls 8–71).

> ⚠️ **The original BF16 LM Head weight stays resident in VRAM.** The INT8 copy is loaded
> *in addition*. This project therefore provides **no evidence of peak-memory savings**.

---

## 5. Benchmark Methodology

### Two different measurements

| Level | Tool | Metric | What it answers |
|---|---|---|---|
| Kernel | CUDA Events, interleaved order | median latency of one GEMV | how much faster is the operator itself |
| Serving | client-side streaming HTTP | approximate TPOT over 128 output tokens | how much survives into end-to-end decode |

These are reported separately and never multiplied together.

### Serving benchmark design

Two orderings are run so that any monotonic thermal or clock drift shows up as a
difference between them:

```
ABBA:  A1=BF16 → B1=INT8 → B2=INT8 → A2=BF16
BAAB:  B1=INT8 → A1=BF16 → A2=BF16 → B2=INT8
```

- **12 measured requests per stage, 3 warmups per stage**, 4 stages.
- **2 runs × 4 stages × 12 = 96 measured requests** (48 BF16 + 48 INT8).
- Each request: `temperature=0`, `ignore_eos=True`, `max_tokens=128`, `stream=True`,
  `stream_options={"include_usage": True}`.
- The server is **restarted between stages**. The runner refuses to continue unless the
  log confirms CUDA Graph capture (`Graph capturing finished`) and, for INT8 stages,
  weight loading (`Quantized weights loaded`).
- GPU temperature, SM clock and memory clock are recorded for every request.
- The run aborts if GPU0 temperature reaches **79 °C**.

### TPOT definition (client-side approximation)

```
TPOT ≈ (t_last_content_chunk − t_first_content_chunk) / (completion_tokens − 1)
```

This is measured in the HTTP client from SSE chunk arrival times. It includes streaming
chunk granularity, client scheduling jitter and any inter-token gap visible on the wire.
**It is not a CUDA-event measurement of the decode loop.**

### What is deliberately not done

- No synchronization trick is used to make the two modes look more different.
- No request is dropped. All 96 rows are in the archived CSVs, including the ones whose
  TPOT is above the stage mean.
- No p-value is computed. The two run orderings disagree in magnitude (see below), which
  is exactly why a significance claim would be unsupported.

---

## 6. Performance Results

### 6.1 Kernel-level microbenchmark — four runs

All four runs use real `Qwen3-VL-8B-Instruct` `lm_head.weight` (`[151936, 4096]`, BF16
source) and one synthetic FP32 activation vector, on **remote server GPU0 (RTX 3090)**.

| Run | BF16 `torch.mm` | INT8 GEMV V2 | Speedup | Raw log |
|---|---:|---:|---:|---|
| Initial retest | 1.6502 ms | 1.3430 ms | **1.229×** | [`microbenchmark_real_weight_20261018_202616.txt`](artifacts/profiling/microbenchmark_real_weight_20261018_202616.txt) |
| Repeat 1 | 1.5713 ms | 1.2058 ms | **1.303×** | [`run_1.txt`](artifacts/profiling/microbenchmark_repeat_20261018/run_1.txt) |
| Repeat 2 | 1.4141 ms | 1.0885 ms | **1.299×** | [`run_2.txt`](artifacts/profiling/microbenchmark_repeat_20261018/run_2.txt) |
| Repeat 3 | 1.4100 ms | 0.7127 ms | **1.978×** | [`run_3.txt`](artifacts/profiling/microbenchmark_repeat_20261018/run_3.txt) |

- **Median of the three repeat runs: 1.303×.** Median of all four: 1.301×.
- Range across all four runs: **1.229× – 1.978×**.
- The 1.978× run is **an outlier, not the representative result**. It is reported because
  omitting it would misrepresent the variance.
- Machine-readable repeat record:
  [`summary.json`](artifacts/profiling/microbenchmark_repeat_20261018/summary.json).

**Why the spread is large.** The GPU temperature before each repeat rose from **33 °C → 43 °C
→ 45 °C**, and the INT8 median fell monotonically (1.3430 → 1.2058 → 1.0885 → 0.7127 ms)
while the BF16 median also drifted (1.6502 → 1.5713 → 1.4141 → 1.4100 ms). Both the
numerator and the denominator move between processes, so the ratio is not stable. The
runs are short (seconds apart), so clock boost state and page-cache warmth are plausible
contributors; **this archive does not separate them.**

**Correctness is stable even though the timing is not.** Every run reports:
`NRMSE ≈ 0.88–0.93%`, `Top1Equal=True` on all three trials, and
`INT8 GPU vs CPU reference: PASS (33 rows)`.

> **Historical figure, not a current result.** An earlier run reported **1.645×**
> (1.4116 → 0.8581 ms). **Its raw stdout is not in this archive.** It is retained only as a
> historical record and must not be quoted as the project's speedup.

**Scope.** One fixed shape, one input vector, median of 20 interleaved measurements per
run. **Not** a whole-model speedup, and **not** transferable to other shapes, batch sizes,
or GPUs. Offline quantization and the input `bf16 → fp32` conversion are **excluded** from
the measured interval.

### 6.2 End-to-end serving (CUDA Graph + `torch.compile`)

Computed directly from `artifacts/profiling/*/results.csv`:

| Run | BF16 mean TPOT | INT8 mean TPOT | Δ | Ratio |
|---|---:|---:|---:|---:|
| ABBA (`graph_abba_20261018_180248`) | 25.119 ms | 24.244 ms | −0.876 ms | 1.036× |
| BAAB (`graph_baab_20261018_184013`) | 25.618 ms | 24.392 ms | −1.226 ms | 1.050× |
| **Combined (n = 96)** | **25.369 ms** | **24.318 ms** | **−1.051 ms** | **1.043×** |

**Mean approximate-TPOT reduction: 4.14%.**

Per-stage medians (from `summary.json`):

| Stage | ABBA median | BAAB median |
|---|---:|---:|
| BF16 (A1) | 24.385 ms | 25.581 ms |
| INT8 (B1) | 24.078 ms | 23.751 ms |
| INT8 (B2) | 24.385 ms | 24.855 ms |
| BF16 (A2) | 25.774 ms | 25.816 ms |

### How to read this honestly

- The **two orderings give different magnitudes** (3.49% vs 4.78%). Ordering, thermal
  drift and clock state are not fully separated by two runs.
- In ABBA, INT8 stage B2 (24.385 ms) is identical to BF16 stage A1 (24.385 ms). The effect
  is **not a clean per-stage separation**.
- The serving path runs under **CUDA Graph + `torch.compile`**, so the measured delta
  **cannot be attributed to the CUDA kernel alone** — graph replay, compile decisions and
  the surrounding plugin overhead all contribute.
- **Eager mode was integrated and exercised, but its end-to-end advantage is not robust**
  and is not claimed here.
- No confidence interval or significance test is reported, and none should be inferred.

### 6.3 Execution modes

| Mode | What happens | Status in this project |
|---|---|---|
| **Eager** | the patched `_apply_head` runs every decode step | integrated and validated; end-to-end benefit **not robust** |
| **CUDA Graph + `torch.compile`** | the decode step is captured and replayed; the custom op is part of the captured graph | integrated; all 96 formal requests were collected in this mode |

---

## 7. Quality Evaluation

### 7.1 Objective set (22 questions, automatic grading)

| Category | n | BF16 | INT8 |
|---|---:|---:|---:|
| Basic computation | 5 | 5 | 5 |
| Basic reasoning | 5 | 2 | 2 |
| Advanced problem solving | 3 | 3 | 3 |
| Advanced computation | 5 | 4 | 4 |
| Advanced reasoning | 4 | 4 | 3 |
| **Total** | **22** | **18** | **17** |

The single-point gap is **Q026**, where the INT8 run hit its 384-token budget before
producing a parseable answer. A **768-token targeted retest** was run afterwards:
**both models answered correctly in 3/3 rounds**. The retest is reported alongside the
original score and does **not** overwrite it.

Two questions (Q011, Q015) failed automatic parsing for **both** models because they
emitted a bare `答案：结果`; the surrounding prose pointed in the right direction. The
automatic score is therefore **not a pure knowledge-accuracy measure**.

### 7.2 Open-ended set (13 deduplicated questions, single blind review)

| Slice | BF16 | INT8 |
|---|---:|---:|
| Original answers only (13 q) | 37/52 | 38/52 |
| Long-answer retest only (8 q) | 22/32 | 20/32 |
| **Deduplicated 13 q (used for the headline)** | **41/52** (78.85%) | **39/52** (75.00%) |

The two-point difference comes entirely from **Q032 (4 vs 3)** and **Q033 (3 vs 2)**;
the other 11 questions scored identically.

**Traceability.** The per-question scores and the unblinding map are in
[`artifacts/quality_ab/blind_review_unblinded_scores.csv`](artifacts/quality_ab/blind_review_unblinded_scores.csv)
(21 rows: 13 original + 8 long-retest). The headline uses the original row for Q001–Q005 and
the long-retest row for the other eight; that selection sums to **41/52 vs 39/52**. The
derivation is reproduced step by step in
[`docs/quality-evaluation.md`](docs/quality-evaluation.md#35-unblinding-and-deduplication-how-4152-and-3952-are-computed).

> ⚠️ `blind_review.json` is the **unscored blind packet** (`score_X_0_to_4` and
> `score_Y_0_to_4` are `null` in every row). It is not the scored file. The scores are in
> the `_unblinded_scores.csv` above.
>
> This was **one subjective review pass** with no second rater, no inter-rater agreement and
> no hypothesis test. **41/52 vs 39/52 is not a statistically significant quality result.**

### 7.3 What may and may not be claimed

| May claim | May **not** claim |
|---|---|
| INT8 output differs from BF16 on a small number of questions | "lossless" / "严格无损" |
| objective 18/22 vs 17/22 on one 22-question set | "the model is 4.5% worse" |
| blind review 41/52 vs 39/52, one reviewer, truncated answers | "quality degradation is statistically proven" |

This was **one subjective review of truncated answers**, plus one or a few generations
per question. It is a sanity check, not a quality study.

---

## 8. Limitations

1. **Kernel speedup ≠ serving speedup.** The kernel-level figure is a **shape-specific
   microbenchmark spanning 1.229×–1.978× across four runs (median of three repeats 1.303×)**;
   4.14% is an end-to-end client-side observation under CUDA Graph. Do not multiply,
   conflate, or quote a single run as "the" speedup.
2. **No statistical significance.** Two orderings, 12 requests per stage, no confidence
   interval, observable thermal/clock drift.
3. **Client-side timing.** TPOT is derived from SSE chunk arrival times, not from CUDA
   events or server-side instrumentation.
4. **CUDA Graph confound.** The end-to-end number is measured under graph replay +
   `torch.compile`; the kernel's isolated contribution is not separated.
5. **Eager mode not established.** The integration works, but no robust end-to-end
   advantage is demonstrated there.
6. **No memory savings demonstrated.** The BF16 LM Head remains resident; the INT8 tensor
   is additional.
7. **Not lossless.** Quantization introduces per-row scale error; the quality suite shows
   small, question-dependent differences.
8. **Narrow applicability.** Weight-only INT8, batch = 1 single-token decode, one model,
   one shape, one GPU. It is not a prefill or batched-GEMM optimization.
9. **No upstream integration.** This is a `vllm.general_plugins` monkey-patch, not an
   accepted vLLM change.
10. **Microbenchmark variance is large and unexplained.** Four runs on the same machine
    span 1.229×–1.978×. Temperature and clock state correlate with the drift, but this
    archive does not isolate the cause.
11. **Partial environment record.** Python / PyTorch / Triton / CUDA runtime / NVCC / driver
    are recorded in [`environment.json`](environment.json). **Nsight Systems version is still
    missing.**
12. **Raw profiler traces are not included** (`.nsys-rep` / `.sqlite` are hundreds of MB
    and are not useful without the matching GPU). Only analysis scripts and derived
    summaries are shipped.

---

## 9. Reproduction Instructions

### 9.1 Prerequisites

- A CUDA GPU with `sm_86` (this work used one RTX 3090)
- vLLM **0.28.0** with `VLLM_PLUGINS` plugin support
- `Qwen3-VL-8B-Instruct` weights available locally
- PyTorch with CUDA, `safetensors`, `requests`

### 9.2 Environment variables

| Variable | Meaning |
|---|---|
| `QWEN_MODEL_DIR` | snapshot directory of `Qwen3-VL-8B-Instruct` |
| `QWEN_LM_HEAD_SHARD` | safetensors shard containing `lm_head.weight` |
| `INT8_LM_HEAD_ENABLE` | `1` = custom INT8 path, `0` = BF16 baseline |
| `INT8_LM_HEAD_TRACE` | `1` = print custom/fallback counters |
| `INT8_LM_HEAD_PROFILE` | `1` = CUDA Events around the three sub-steps |
| `INT8_LM_HEAD_NSYS` | `1` = `cudaProfilerStart/Stop` around decode calls 8–71 |

All scripts are run with the **project root** (`int8-lm-head/`) as the working directory,
because they write to `artifacts/`.

### 9.3 Build the extension — **NOT VERIFIED**

> ⚠️ **`setup.py` is not a CUDA extension build script.** It only declares the vLLM
> plugin entry point (`vllm.general_plugins`). Running **`pip install -e .` from the source
> directory** installs the plugin registration in editable mode; it will **not** produce
> `int8_gemv_sm86_ext`.
>
> Use `-e` deliberately: a non-editable `pip install .` moves the module into
> `site-packages/` and breaks the plugin's `Path(__file__).parent` data paths (see §4).
>
> **This step does not compile CUDA code.**

What the archive *does* establish:

| Evidence | Where |
|---|---|
| The extension module is named `int8_gemv_sm86_ext` | `scripts/test_real_lm_head.py`, `int8_vllm_plugin.py` |
| It is imported from a `build/` directory next to the plugin | same files (`sys.path.insert(0, .../"build")`) |
| The binding uses `torch/extension.h` + `PYBIND11_MODULE` | `src/int8_binding.cpp` |
| `int8_launch.cu` `#include`s `int8_gemv_v2.cu` | `src/int8_launch.cu` |
| CUDA target is `sm_86` | module name + RTX 3090 in `environment.json` |

**The exact build invocation is not in the archive** — no extension `setup.py`, no
`build.ninja`, no shell history. Therefore:

> ❌ **This README intentionally provides no build command.** Any command here would be
> reconstructed, not verified.
> ✅ Rebuilding requires deriving it from the sources above and is **NOT VERIFIED** by this
> repository. Treat the extension binary as unavailable until you rebuild it yourself.

### 9.4 Quantize the LM Head

```bash
export QWEN_LM_HEAD_SHARD=/path/to/model-00004-of-00004.safetensors
python scripts/export_int8_lm_head.py
# writes artifacts/lm_head_int8_rowwise.safetensors  (~594 MiB)
```

### 9.5 Kernel correctness and microbenchmark

```bash
python scripts/test_extension.py        # synthetic shapes
python scripts/test_real_lm_head.py     # real weight + microbenchmark
```

### 9.6 Serving benchmark

```bash
# Terminal 1 — start the server for the current stage (INT8_LM_HEAD_ENABLE=0/1)
INT8_LM_HEAD_ENABLE=0 bash scripts/launch_server.sh

# Terminal 2 — run the ordered comparison (starts/stops servers itself)
python scripts/run_graph_abba.py
python scripts/run_graph_baab.py
```

### 9.7 Quality suite

```bash
python scripts/quality_ab_suite.py      # needs BF16 on :8001 and INT8 on :8002
python scripts/retest_q026.py
python scripts/retest_open_truncated.py
```

### 9.8 Analysis helpers

`scripts/analyze_decode_timeline.py`, `analyze_gap_api.py`, `analyze_gap_neighbors.py` and
`analyze_gpu_gaps.py` read a local Nsight Systems `.sqlite` export. **The trace itself is
not included**, so these scripts will not run out of the box.

---

## Repository layout

```
int8-lm-head/
├── setup.py        vLLM plugin packaging (entry point) — NOT the CUDA build script
├── int8_vllm_plugin.py  the plugin itself (must sit next to setup.py)
├── src/            CUDA kernels + pybind11 binding
├── scripts/        quantization, benchmarks, quality suite, analysis
├── artifacts/
│   ├── profiling/  ABBA / BAAB results.csv + summary.json
│   │               microbenchmark: 1 full log + 3 repeat logs + summary.json
│   └── quality_ab/ objective + open-ended evaluation JSON
├── docs/           methodology, quality evaluation, limitations, code changelog
├── environment.json  server software versions
├── README.md       this file
└── README.zh-CN.md 中文版
```

## License / provenance

See the repository root [`LICENSE_NOTE.md`](../LICENSE_NOTE.md). All numbers in
`artifacts/` are copied from the original server run; the only edits made during
publication were absolute host-path substitutions, listed in
[`docs/public-code-changelog.md`](docs/public-code-changelog.md). No measurement,
scheduling or statistical logic was altered.
