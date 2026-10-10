# Profiling commands

> ⚠️ **Provenance honesty note.** The **exact `nsys profile` invocation was not recorded** in the
> frozen Exp12 evidence. It is therefore **not reproduced verbatim here** — inventing a
> plausible-looking command would misrepresent the record.
>
> What follows is (a) what *is* recorded, and (b) the reconstruction needed to repeat the
> procedure, clearly labelled as a reconstruction.

---

## 1. What IS recorded in the frozen evidence

| Item | Value | Source |
|---|---|---|
| Tool | Nsight Systems **2024.5.1** | Exp12 protocol / methodology notes |
| Trace set | `cuda,nvtx,osrt` | Exp12 protocol |
| Timestamp setting | `CuptiUseRawGpuTimestamps=false` (WSL2 workaround) | Exp12 protocol |
| Condition A capture | `exp12_A_c1_mbt2048.nsys-rep` (+ `.sqlite`) | SHA256SUMS |
| Condition B capture | `exp12_B_c8_mbt2048.nsys-rep` (+ `.sqlite`) | SHA256SUMS |
| Condition C capture | `exp12_C_c8_mbt4096.nsys-rep` (+ `.sqlite`) | SHA256SUMS |
| Post-processing script | `analyze_timeline.py` | this repo |
| Machine-readable outputs | `results/exp12_timeline_analysis.txt`, `results/exp12_summary.txt`, `results/exp12_hypothesis_status.csv` | this repo |

The **raw `.nsys-rep` / `.sqlite` files are deliberately not published** — hundreds of MB, and
unusable without the matching GPU. Their SHA256 digests are recorded in the original
`exp12_SHA256SUMS.txt` (not shipped; the manifest also lists the analysis outputs).

---

## 2. Reconstruction (NOT the original command)

> Everything in this section is **reconstructed from the recorded settings**. It reproduces the
> procedure; it is not a transcript of what was run.

The server must already be running at the target MBT (see
[`../../docs/reproduction-guide.md`](../../docs/reproduction-guide.md) §B1). Profiling wraps the
**client burst**, not the server.

```bash
# WSL2 requires raw GPU timestamps to be disabled, otherwise the timeline is unusable.
export CUPTI_USE_RAW_GPU_TIMESTAMPS=false

COND=A_c1_mbt2048          # A_c1_mbt2048 | B_c8_mbt2048 | C_c8_mbt4096

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output="exp12_${COND}" \
  --force-overwrite=true \
  python3 ../../common/request_client.py --concurrency <C> --prompt-tokens 2048 --completion-tokens 128
```

Then export the report to SQLite, which is what `analyze_timeline.py` reads:

```bash
nsys export --type sqlite --output="exp12_${COND}.sqlite" "exp12_${COND}.nsys-rep"
```

### Condition mapping

| Condition | Concurrency | MBT | Capture basename |
|---|---:|---:|---|
| **A** | 1 | 2048 | `exp12_A_c1_mbt2048` |
| **B** | 8 | 2048 | `exp12_B_c8_mbt2048` |
| **C** | 8 | 4096 | `exp12_C_c8_mbt4096` |

---

## 3. Post-processing

```bash
python3 analyze_timeline.py --help
```

The script reads the per-condition SQLite and produces:

- `prepare_prefill` **instance counts** (the prefill iteration marker)
- the **relative time of the final** `prepare_prefill` marker
- the **steady-state `64x64` kernel spacing** in the after-final-marker region

---

## 4. Discipline that must not be relaxed

| # | Rule |
|---|---|
| 1 | **A capture existing ≠ the workload being valid.** Each profiled condition independently passes client validity (checked, not assumed). |
| 2 | **Profiler timing ≠ serving timing.** Profiling adds overhead and each trace is a separate run. These traces are mechanism evidence only. |
| 3 | **CUDA API time ≠ GPU execution time.** `cuda_api_sum.csv` is host-side API call time; `cuda_gpu_kern_sum.csv` is GPU kernel execution. Never mix them. |
| 4 | **A kernel-composition change ≠ a bottleneck proof.** It shows the execution regime changed — not what the bottleneck is. |
| 5 | **Kernel instance counts ≠ token counts ≠ scheduler-wave counts.** `prepare_prefill` is an iteration marker, not a cost. |
| 6 | **`prepare_prefill` semantics are inferred** from naming and structure; vLLM's source was not inspected. |

## 5. Power provenance

Condition **C** has a contemporaneous power-boundary record (`results/exp12_C_power_boundary.txt`
= 220.00 W). **Conditions A and B do not** — their power state is `UNKNOWN` and must **not** be
back-inferred from other experiments.
