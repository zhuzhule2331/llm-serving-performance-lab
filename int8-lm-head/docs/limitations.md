# Limitations

This file is intentionally blunt. If a sentence here contradicts a sentence elsewhere in
the repository, **this file wins**.

---

## 1. Measurement limitations

### 1.1 Kernel speedup is shape-specific

The kernel ratio (**1.229×–1.978× across four runs**) is measured at exactly
`[151936, 4096]` with a single FP32 activation vector. It is a **microbenchmark**, not a
model-level or workload-level speedup. It will not transfer to:

- other vocabulary sizes or hidden dimensions,
- batch > 1 (the kernel assumes one row per warp and one input vector),
- prefill (which is a GEMM, not a GEMV),
- other GPUs with different memory bandwidth or SM counts.

### 1.2 End-to-end timing is client-side

`tpot_ms` is computed from **SSE chunk arrival times** in an HTTP client. It includes:

- streaming chunk granularity (a chunk may contain more than one token's worth of work),
- client-side scheduling jitter,
- network/loopback overhead,
- server-side batching decisions invisible to the client.

It is **not** a CUDA-event measurement of the decode loop. A 4.14% client-side difference
should not be read as a 4.14% change in GPU time.

### 1.3 No statistical significance

- 12 measured requests per stage, 4 stages, 2 orderings.
- No confidence interval, no hypothesis test.
- Requests within a stage share one process and one thermal state, so treating 96 rows as
  96 independent trials would be wrong.
- The two orderings disagree in magnitude: **ABBA 3.49%, BAAB 4.78%**.
- In ABBA, INT8 stage B2 and BF16 stage A1 both have median 24.385 ms.

### 1.4 Thermal and clock drift

`gpu_status` in `results.csv` shows GPU temperature rising from roughly 56–60 °C to about
65 °C within a run, with SM clock fluctuating between roughly 1 500 and 1 950 MHz. The two
orderings are an attempt to expose this drift, not to eliminate it.

### 1.5 CUDA Graph confound

All 96 formal requests were collected with **CUDA Graph + `torch.compile`**. The measured
delta therefore includes:

- the custom kernel,
- graph replay effects,
- compilation decisions,
- plugin-side reshape / cast / Python overhead.

**The kernel's isolated contribution is not separated.** Do not attribute 4.14% to the
CUDA kernel alone.

### 1.6 Eager mode not established

The Eager path was integrated, the plugin was verified to call the custom kernel, and the
fallback was verified. But the **end-to-end advantage in Eager is not robust** and is not
claimed.

---

## 2. Implementation limitations

### 2.1 No memory savings

The original BF16 LM Head weight **remains resident**. The INT8 tensor (≈594 MiB) is loaded
in addition to it. This project therefore says **nothing** about peak memory reduction.

Any statement like "显存降为 1/4" or "saved 4× memory" is unsupported.

### 2.2 Not lossless

Per-output-row symmetric INT8 quantization introduces scale error. The quality evaluation
shows small but real, question-dependent differences. "Lossless" and "完全一致" are both
wrong.

### 2.3 Weight-only INT8

The activation stays FP32 and the multiply–accumulate stays FP32. This is **not** W8A8 and
does **not** use INT8 Tensor Cores. The speedup comes from reduced weight bytes read, not
from higher arithmetic throughput.

### 2.4 Narrow eligibility

The plugin only takes the custom path when **all** of these are true:

- CUDA tensor, BF16 hidden states,
- `hidden.shape[-1] == 4096` and `hidden.numel() == 4096` (exactly one token),
- weight shape exactly `(151936, 4096)`,
- `tp_size == 1`,
- no embedding bias,
- `UnquantizedEmbeddingMethod`,
- compatible head dtype.

Anything else silently falls back to BF16. This is a **single-model, single-shape**
optimization.

### 2.5 Monkey-patch, not upstream

The integration replaces `LogitsProcessor._apply_head` at runtime through
`vllm.general_plugins`. It is tied to vLLM **0.28.0** internals. A vLLM upgrade can break it
without warning, and it has not been proposed upstream.

### 2.6 Build step not archived

The CUDA extension build command was not preserved. `setup.py` is the **plugin
packaging** file, not the extension build script — installing it registers the plugin but
does **not** produce `int8_gemv_sm86_ext`.

**Install mode:** use `pip install -e .` **from the source directory**. A non-editable
`pip install .` copies the module into `site-packages/`, where the plugin's
`Path(__file__).parent` data paths (`artifacts/…`, `build/`) do not exist. Editable install
fixes only Python module resolution; it does **not** build the CUDA extension and does
**not** supply the quantized weight file.

What is known: the module name, that it is imported from a `build/` directory, that the
binding uses `torch/extension.h` + `PYBIND11_MODULE`, and that `int8_launch.cu` includes
`int8_gemv_v2.cu`. **The exact build invocation is absent, so this repository deliberately
publishes no build command. Build reproducibility is NOT VERIFIED.**

---

## 3. Evidence limitations

### 3.1 Microbenchmark variance is large

Four runs of the same script on the same machine produced **1.229×, 1.303×, 1.299× and
1.978×**. The median of the three repeats is **1.303×**; the range is **57.5% of that
median**.

Observed correlates:

- GPU0 pre-run temperature rose 33 °C → 33 °C → 43 °C → 45 °C across the four runs.
- INT8 medians fell monotonically: 1.3430 → 1.2058 → 1.0885 → 0.7127 ms.
- BF16 medians also fell: 1.6502 → 1.5713 → 1.4141 → 1.4100 ms.

Because both numerator and denominator move, the ratio is unstable. **This archive does
not isolate clock boost, page-cache warmth, or thermal state as the cause.** The runs are
seconds apart and are not independent process launches on a cold machine.

**The 1.978× run is an outlier and is not the representative result.** It is retained in
the record because deleting it would hide the variance.

### 3.2 The historical 1.645× has no raw output

An earlier run reported **1.645×** (BF16 1.4116 ms → INT8 0.8581 ms). **Its raw stdout is
not part of this archive.** It is kept only as a historical record and must not be quoted
as the project's speedup. The four logged runs supersede it.

### 3.3 Partial environment record

[`environment.json`](../environment.json) records Python 3.12.14, PyTorch 2.13.0+cu129,
vLLM 0.28.0+cu129, Triton 3.7.1, CUDA runtime 12.9.79, NVCC 12.6.85 and driver 572.83.

**Still missing: the Nsight Systems version.** Also note the NVCC 12.6 / runtime 12.9
mismatch, recorded as-is.

### 3.4 Raw profiler traces excluded

`.nsys-rep` and `.sqlite` traces are hundreds of megabytes and are useless without the
matching GPU. Four `analyze_*.py` scripts in `scripts/` therefore **cannot run** against
this repository alone.

### 3.5 Quality review is single-pass and subjective

One reviewer, truncated answers, one to a few generations per question.

- `blind_review.json` is the **unscored blind packet** — `score_X_0_to_4` and
  `score_Y_0_to_4` are `null` in every row.
- The scores live in `artifacts/quality_ab/blind_review_unblinded_scores.csv`. They were
  produced by a human reviewer and are **not regenerable by script**.
- There is no second rater, no inter-rater agreement, no confidence interval and no
  hypothesis test. **41/52 vs 39/52 is not a statistically significant quality result**
  and cannot support a population-level claim about model capability.

---

## 4. Things this project explicitly does **not** claim

```
✗  整模型 1.645× / 1.303× / 任一单次加速比
✗  "稳定 1.645×" 或 "稳定 1.3×"
✗  所有场景稳定 4.14% 收益
✗  已证明统计显著
✗  INT8 精度无损
✗  显存节省 4× / 显存占用降低
✗  多卡 / TP=2 扩展已验证
✗  RTX 5060 上的任何测量结果
✗  GPU1 参与了性能对照
✗  ncu 算子内部级分析
✗  已合入 vLLM 上游
✗  Eager 模式端到端优势
```

If a recruiter, interviewer or reader asks about any of the above, the correct answer is
that it was **not** established by this work.
