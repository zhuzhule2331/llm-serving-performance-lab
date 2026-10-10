# Reproduction guide

Two levels of reproduction are possible, and they are **not** the same thing:

| Level | What it needs | Status in this repo |
|---|---|---|
| **A. Re-analyze / re-plot** | Python only | ✅ fully supported |
| **B. Re-run the serving experiments** | 1× RTX 3090, vLLM 0.28.0, Nsight Systems | ⚠️ scripted but **not verified in this environment** |

---

## 0. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env        # then edit
```

`env.example` documents every configurable path. The frozen scripts originally had these
hard-coded to the experiment host; they are now environment variables **with the same effect**.
See [`../../PUBLIC_CODE_CHANGELOG.md`](../../PUBLIC_CODE_CHANGELOG.md).

| Variable | Meaning | Default |
|---|---|---|
| `VLLM_BIN` | path to the `vllm` executable | `vllm` (from `PATH`) |
| `MODEL_PATH` | path to the Qwen3-VL-8B-Instruct snapshot **or** HF repo id | `models/Qwen3-VL-8B-Instruct` |
| `VLLM_HOME` | base dir used when composing cache/server paths | `./` |
| `VLLM_HOST` / `VLLM_PORT` | OpenAI-compatible endpoint | `127.0.0.1` / `8000` |
| `RESULTS_DIR` | output dir used by the analysis scripts | `results` |

> ⚠️ `API_KEY` in the client is the literal string `"EMPTY"`. That is vLLM's placeholder for a
> server started **without** `--api-key`. It is not a credential.

---

## A. Re-analyze and re-plot (no GPU needed)

```bash
# static checks
python3 scripts/check_environment.py
python3 scripts/verify_results.py

# regenerate the figures from the frozen CSVs
python3 figures/make_figures.py
```

`figures/make_figures.py` reads only `results/*.csv` and **asserts** the values it expects, so a
mismatch fails loudly rather than producing a silently wrong chart.

---

## B. Re-run the experiments (single RTX 3090)

> **Not executed here.** This section documents how the original campaign was driven. It has not
> been validated in this repository's environment. Treat it as a procedure, not a tested path.

### B1. Start the vLLM server at a given MBT

```bash
$VLLM_BIN serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 8000 \
  --max-num-seqs 16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching          # Exp08 / Exp12 ; Exp11 drops this flag
```

MBT is a **whole-plot** factor: the server must be restarted between MBT levels.

### B2. Drive one experiment

```bash
# Exp08 — behavioral
python3 experiments/exp08_mbt_concurrency/preflight.py
python3 experiments/exp08_mbt_concurrency/runner.py --help

# Exp11 — interaction (needs the frozen namespace map)
python3 experiments/exp11_prefix_mbt/namespace_map.py
python3 experiments/exp11_prefix_mbt/formal_runner.py --help

# Exp11 — analysis / merge
python3 experiments/exp11_prefix_mbt/finalize_merge.py --help
python3 experiments/exp11_prefix_mbt/analyze_formal.py --help
```

### B3. Exp12 profiling

See [`../experiments/exp12_decode_profiling/profiling_commands.md`](../experiments/exp12_decode_profiling/profiling_commands.md)
for the exact `nsys profile` invocations and the SQLite export step, then:

```bash
python3 experiments/exp12_decode_profiling/analyze_timeline.py --help
```

### B4. Preconditions you must reproduce

These are **validity criteria**, not suggestions. A run that violates them is invalid, not noisy.

| # | Requirement |
|---|---|
| 1 | `GPU0 power limit = 220 W`, recorded **while the condition runs** |
| 2 | launch skew within a synchronized burst ≤ **25 ms** |
| 3 | preemption delta = **0** (from before/after `/metrics` snapshots) |
| 4 | 1 warmup burst excluded + 2 measured bursts per condition |
| 5 | completion length exactly 128 tokens (asserted, not assumed) |
| 6 | workload identity verified against a frozen map — **no runtime search** |

---

## What can and cannot match

| Quantity | Reproducible elsewhere? |
|---|---|
| Ratio-style results (40xx-vs-512 percentage changes, block sign consistency) | **Plausibly**, on similar hardware |
| Absolute latencies (ms) | **No** — machine-, driver-, and power-cap-specific |
| Kernel names and marker counts in Exp12 | **Only** with the same vLLM build and GPU |
| The `.nsys-rep` traces | **No** — not published (size) |

---

## Scope of validation in THIS repository

```
STATIC CHECK (py_compile, import, --help, path scan)   : PERFORMED
RUNTIME REPRODUCTION (GPU / vLLM execution)            : NOT EXECUTED
```

This repository was prepared on a machine **without a GPU and without vLLM installed**. No
runtime result is claimed. See [`../../PUBLIC_REPO_AUDIT.md`](../../PUBLIC_REPO_AUDIT.md).
