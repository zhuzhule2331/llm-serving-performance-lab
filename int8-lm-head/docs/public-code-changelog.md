# Public code changelog

> Provenance record for the code and data shipped under `int8-lm-head/`.
> **Exactly one class of behavioural change was made when preparing this public copy:
> absolute host paths replaced by environment variables.** A second class of
> non-behavioural change is the relocation of files into `src/` and `scripts/`.
> No measurement, scheduling or statistical logic was altered.

Source package: `03-作品集/llm_serving_evidence_20261018/`, verified against its
`MANIFEST.tsv` (SHA-256).

---

## Summary

| Category | Count |
|---|---:|
| Files copied byte-identically (2026-10-18 first batch) | **16** |
| Files copied byte-identically (2026-10-18 final evidence batch) | **6** |
| Files with host-path substitution | **5** |
| Metadata JSON with path field rewritten | **2** |
| Files relocated without modification | 22 (same files as above) |
| Newly authored files | **2** (`scripts/launch_server.sh`, this changelog) |
| Relocated files (path resolution only) | **2** (`setup.py`, `int8_vllm_plugin.py`) + 4 scripts with corrected root resolution |
| Quality scoring sheet added | **1** (byte-identical copy) |
| Files with altered measurement logic | **0** |

---

## 1. Copied byte-identically

Verified by SHA-256 against `MANIFEST.tsv`:

| Public path | SHA-256 (first 16) |
|---|---|
| `src/int8_gemv_v2.cu` | `5aa01374b5f2ae9a` |
| `src/int8_launch.cu` | `e6d204d63eb2674f` |
| `src/int8_binding.cpp` | `d90eb49ff13802ef` |
| `setup.py` (byte-identical; relocated to project root) | `38e3338998286a56` |
| `int8_vllm_plugin.py` (byte-identical; relocated to project root) | `e8afcb0234de612a` |
| `scripts/test_extension.py` | `3d815b5d4fc48987` |
| `scripts/bench_gpu0_abba.py` | `2a987739e8e5d6cd` |
| `scripts/quality_ab_suite.py` | `4da1894c0a7572af` |
| `scripts/retest_q026.py` | `97a72b494d2c427b` |
| `scripts/retest_open_truncated.py` | `c76a4ffd9adf6961` |
| `scripts/test_cold_warm_ttft.py` | `e6c1b62170d62f29` |
| `scripts/analyze_decode_timeline.py` | `e8bf38a7ba43ce8e` |
| `scripts/analyze_gap_api.py` | `f071ed3bdb5dbdd4` |
| `scripts/analyze_gap_neighbors.py` | `a0214abbf0c8772b` |
| `scripts/analyze_gpu_gaps.py` | `285d597bb5828c71` |
| `scripts/nsys_smoke.py` | `023d8b6741072f4b` |

The two `results.csv` files under `artifacts/profiling/` are also **byte-identical**:

| File | SHA-256 (first 16) |
|---|---|
| `graph_abba_20261018_180248/results.csv` | `08a2598eb19f476a` |
| `graph_baab_20261018_184013/results.csv` | `8127c177d7559462` |

All six JSON files under `artifacts/quality_ab/` were copied without modification.

### Final evidence batch (2026-10-18, second transfer)

Copied byte-identically from `llm_serving_final_evidence_20261018/`, verified against that
package's `MANIFEST.tsv`:

| Public path | SHA-256 (first 16) |
|---|---|
| `environment.json` | `ec3a0c23e008821d` |
| `artifacts/profiling/microbenchmark_real_weight_20261018_202616.txt` | `6164977c7b2d35fd` |
| `artifacts/profiling/microbenchmark_repeat_20261018/run_1.txt` | `0781963fb58fdac7` |
| `artifacts/profiling/microbenchmark_repeat_20261018/run_2.txt` | `712c57248dfc3776` |
| `artifacts/profiling/microbenchmark_repeat_20261018/run_3.txt` | `704aa4eb42608772` |
| `artifacts/profiling/microbenchmark_repeat_20261018/summary.json` | `557ecfe04f2240df` |

The private handoff file `BENCHMARK_FINAL_SUMMARY.md` was **not** copied verbatim: it carries
a "PRIVATE handoff material" banner. Its four-run table and limitations were instead written
into `README.md`, `README.zh-CN.md` and `docs/benchmark-methodology.md` with the same values.

### Quality scoring sheet (2026-10-18, third pass)

| Public path | SHA-256 (first 16) | Source |
|---|---|---|
| `artifacts/quality_ab/blind_review_unblinded_scores.csv` | `34e7fcde096d0f97` | `resume_handoff_20261018/quality_unblinded_scores_20261018.csv`, copied byte-identically |

This is the **scored** sheet. The separate `blind_review.json` is the **unscored** blind
packet (all `score_X_0_to_4` / `score_Y_0_to_4` values are `null`) and is not described as
a scored file anywhere in this repository.

---

## 2b. Layout corrections (2026-10-18, third pass)

**Behavioural effect: `POSSIBLE (path resolution only)`.** No measurement, scheduling or
statistical logic changed.

| File | Change | Reason |
|---|---|---|
| `setup.py` | moved `src/setup.py` → project root | `py_modules=["int8_vllm_plugin"]` resolves relative to the directory containing `setup.py` |
| `int8_vllm_plugin.py` | moved `scripts/int8_vllm_plugin.py` → project root | the plugin resolves `artifacts/lm_head_int8_rowwise.safetensors` and `build/` via `Path(__file__).parent`, which must be the project root |
| `scripts/export_int8_lm_head.py` | `Path(__file__).parent` → `.parents[1]` for the output dir | writes to `<root>/artifacts/`, not `<root>/scripts/artifacts/` |
| `scripts/capture_decode_hidden.py` | same change | same reason |
| `scripts/test_extension.py` | build dir `parent` → `parents[1]` | imports the extension from `<root>/build/` |
| `scripts/test_real_lm_head.py` | `BASE` `parent` → `parents[1]` | same reason |

**Verification performed after the move** (static, no vLLM install):

- `python setup.py --name` → `qwen-int8-lmhead-experiment`; `--version` → `0.1.0`.
- `python setup.py egg_info` → `SOURCES.txt` lists `int8_vllm_plugin.py`;
  `entry_points.txt` contains `qwen_int8_head = int8_vllm_plugin:register`.
- `import int8_vllm_plugin; callable(register)` → `True`.
- The entry-point string resolves to the function.
- `ARTIFACT` resolves to `<root>/artifacts/lm_head_int8_rowwise.safetensors`.

**Not verified:** installation into a vLLM environment and vLLM's plugin loader actually
invoking `register()` — vLLM is not installed on the preparation machine. This is stated in
the README as **NOT VERIFIED**.

---

## 2. Host-path substitutions

**Behavioural effect: `POSSIBLE (path resolution only)`.** Every replacement is listed below.

### `scripts/run_graph_abba.py`

- **source**: `artifacts/profiling/run_graph_abba.py`
- **change**: `ROOT` hard-coded to `<host>/...` → `Path(__file__).resolve().parents[1]`;
  `MODEL` hard-coded snapshot path → `os.environ["QWEN_MODEL_DIR"]`
- **reason**: remove host path; make the script runnable from any checkout
- **sha256 source**: `9d6e7f8b7d29ef67…`
- **sha256 public**: `c0be2eb8209d1d4d…`

### `scripts/run_graph_baab.py`

- **source**: `artifacts/profiling/run_graph_baab.py`
- **change**: identical substitution to `run_graph_abba.py`
- **sha256 source**: `b32c5e18a92beca9…`
- **sha256 public**: `55955712e0dc5e33…`

### `scripts/capture_decode_hidden.py`

- **source**: `capture_decode_hidden.py`
- **change**: `ROOT` hard-coded snapshot path → `os.environ["QWEN_MODEL_DIR"]`
- **sha256 source**: `08e5d8df5a881b3f…`
- **sha256 public**: `1d4d77192f96d0d1…`

### `scripts/export_int8_lm_head.py`

- **source**: `export_int8_lm_head.py`
- **change**: `SOURCE` hard-coded safetensors shard path → `os.environ["QWEN_LM_HEAD_SHARD"]`;
  added `import os`
- **sha256 source**: `c7a3e7c1c735b0ec…`
- **sha256 public**: `4d1d1ebdca7d3cba…`

### `scripts/test_real_lm_head.py`

- **source**: `test_real_lm_head.py`
- **change**: `SHARD` hard-coded safetensors shard path → `os.environ["QWEN_LM_HEAD_SHARD"]`;
  added `import os`
- **sha256 source**: `77d3d631072afb86…`
- **sha256 public**: `06566435068c7b94…`

### `artifacts/profiling/*/summary.json` (×2)

- **change**: the `results_csv` field contained the original absolute host path.
  It was rewritten to the repository-relative path
  `artifacts/profiling/<run_id>/results.csv`.
- **behavioural effect**: **NONE** — this field is a metadata pointer; no measurement,
  count, mean, median or ratio was touched.
- Every other field is unchanged.

---

## 3. Newly authored files

### `scripts/launch_server.sh`

- **not archived with the original experiment.**
- Reconstructed from the `CMD` list and the environment block inside
  `scripts/run_graph_abba.py`.
- Flags are identical to that `CMD`; nothing was added or removed.
- Marked as reconstructed in the script header.

### `src/int8_gemv_v1.cu`

- **Source is not the evidence package** — it comes from the local
  `03-作品集/cuda_kernel_lab/int8_gemv/int8_gemv_v1.cu`.
- V1 was the first working version; **all reported measurements used V2**.
- V1 is published for design context only. It was not benchmarked in the formal runs.

### This changelog

- Newly authored.

---

## 4. Files deliberately **not** published

| File | Reason |
|---|---|
| `bench_dual_gpu.py` | Two-server script (`:8001` / `:8002`) from an **exploratory dual-GPU phase**. The formal comparison used a single GPU0 server on `:8002`. Publishing it without context would contradict the hardware boundary. |
| `nap_gpu_study.py` | Exploratory cold/warm first-token study that assigns **INT8 to GPU0 and BF16 to GPU1**. This is **not** the formal design and would misrepresent the hardware claim. |
| `nap_gpu0_idle_study.py`, `nap_gpu0_bf16_study.py` | Related exploratory cold/warm studies with **no archived results**. Out of scope. |
| `artifacts/profiling/bench_graph.py` | Early informal `bench_graph` script, superseded by `run_graph_abba.py` / `run_graph_baab.py`. Publishing both would be redundant and confuse which produced the reported numbers. |
| `build/`, `.so`, `.o`, compiled ELF binaries | Build artifacts, not reproducible without the missing build command. |
| `artifacts/lm_head_int8_rowwise.safetensors`, model weights, HF cache | Model weights (~594 MiB / ~1.16 GiB) — excluded by repository policy. |
| `*.nsys-rep`, `*.sqlite` | Raw profiler traces, hundreds of MB. |
| Service logs (`*.log`) under `artifacts/profiling/` | Raw runtime logs may contain host paths; excluded. |

---

## 5. Verification statement

- Source hashes were checked against `llm_serving_evidence_20261018/MANIFEST.tsv` and
  `llm_serving_final_evidence_20261018/.../MANIFEST.tsv`.
- 16 of 21 code/script files are byte-identical (first batch); 6 of 6 new evidence files are
  byte-identical (final batch).
- The 5 changed files differ **only** in the path substitutions listed above.
- Both `results.csv` files are byte-identical.
- No number in `artifacts/` was recomputed, rounded or edited.
