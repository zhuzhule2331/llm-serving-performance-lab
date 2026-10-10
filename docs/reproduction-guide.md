# Reproduction guide

This repository supports **different levels of reproduction**, and they are not the same thing.
Read the level table before assuming anything runs.

---

## 0. Reproduction levels

| Level | What it means | Status |
|---|---|---|
| **A** | **Published aggregate verification** — re-check the documented numbers against `results/` | ✅ **SUPPORTED** |
| **B** | **Figure regeneration** — rebuild the 4 figures from `results/` | ✅ **SUPPORTED** |
| **C** | **Experiment code static / import validation** — `py_compile`, imports, CLI surface, path resolution | ✅ **SUPPORTED** |
| **D** | **Fresh GPU serving reproduction from a clean clone** — start vLLM and re-run the campaigns | ❌ **NOT VERIFIED** |
| **E** | **Exp12 raw trace re-analysis** — recompute the timeline summary from the profiler captures | ❌ **NOT POSSIBLE FROM THIS REPO** |

> **A/B/C passing does not imply D.** No experiment was executed while preparing this repository.
> Level E is impossible because the raw `.sqlite` / `.nsys-rep` captures are not published.

---

## 1. Levels A and B — run these (no GPU needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Level A — assert every documented number against the frozen CSVs
python3 scripts/verify_results.py          # expect: 28 checks, 0 failures

# Level B — regenerate the four figures from results/
python3 figures/make_figures.py

# environment report (static; never runs an experiment)
python3 scripts/check_environment.py
```

`figures/make_figures.py` reads only `results/*.csv` and **asserts** the values it expects, so a
mismatch fails loudly rather than producing a silently wrong chart.

`env.example` documents every configurable path. The frozen scripts originally had these
hard-coded to the experiment host; they are now environment variables **with the same effect**.
See [`public-code-changelog.md`](public-code-changelog.md).

| Variable | Meaning | Default |
|---|---|---|
| `VLLM_BIN` | path to the `vllm` executable | `vllm` (from `PATH`) |
| `MODEL_PATH` | path to the Qwen3-VL-8B-Instruct snapshot **or** HF repo id | `models/Qwen3-VL-8B-Instruct` |
| `VLLM_HOME` | base dir used when composing cache/server paths | `./` |
| `VLLM_HOST` / `VLLM_PORT` | OpenAI-compatible endpoint | `127.0.0.1` / `8000` |
| `RESULTS_DIR` | output dir used by the analysis scripts | `results` |

> ⚠️ `API_KEY` in the client is the literal string `"EMPTY"` — vLLM's placeholder for a server
> started **without** `--api-key`. It is not a credential.

---

## 2. Level C — what static validation covers

| Check | Command |
|---|---|
| byte-compile all modules | `python3 scripts/smoke_test.py` |
| import resolution | same |
| CLI surface | same |
| frozen-artifact assertions | `python3 scripts/verify_results.py` |

**`py_compile` PASS ≠ runtime PASS.** See [`public-repo-audit.md`](public-repo-audit.md) §3.

---

## 3. Which files are actually runnable

**Read this before typing any command.** Most of `experiments/` is **archival / reference code**
that ran on the experiment host under a specific driver. Only four entry points expose a CLI.

| File | Has `--help`? | Can it run from a clean clone? |
|---|---|---|
| `common/request_client.py` | ✅ | the burst client; needs a live vLLM server |
| `experiments/exp11_prefix_mbt/formal_runner.py` | ✅ | **dry-run path only** — see §4.2 |
| `experiments/exp11_prefix_mbt/formal_execute.py` | ✅ | needs a GPU, a running vLLM server, and `--execute` |
| `experiments/exp11_prefix_mbt/recover_block2.py` | ✅ | executes the block-replacement policy; needs campaign dirs (not published) |
| `experiments/exp08_mbt_concurrency/runner.py` | ❌ **no parser** | archival driver — see §4.1 |
| `experiments/exp08_mbt_concurrency/preflight.py` | ❌ **no parser** | archival |
| `experiments/exp11_prefix_mbt/analyze_formal.py` | ❌ **no parser** | needs a merged-campaign directory (not published) |
| `experiments/exp11_prefix_mbt/finalize_merge.py` | ❌ **no parser** | needs campaign directories (not published) |
| `experiments/exp12_decode_profiling/analyze_timeline.py` | ❌ **no parser** | **reference implementation only** — see §4.3 |

> Commands such as `runner.py --help` or `analyze_timeline.py --help` **do not exist**. Those
> files accept no arguments; invoking them starts real work or fails on missing data.

---

## 4. Level D and E — not verified / not possible

> **Not executed here.** These sections document how the original campaign was driven. They have
> **not** been validated in this repository's environment. Treat them as a procedure.

### 4.1 Exp08 — archival driver

`runner.py` is the frozen campaign driver. It orchestrates the split-plot order, server lifecycle,
and validity gating, and **writes the two summary CSVs itself** (there is no separate analysis
stage). It has no argument parser; parameters are module constants. It expects a live vLLM server
on `$VLLM_HOST:$VLLM_PORT` and reaches the burst client at `common/request_client.py`.

**Status: archival / reference.** The path is fixed, but the driver has not been executed from a
clean clone. No `--help` is available.

### 4.2 Exp11 — two distinct drivers

| File | Role |
|---|---|
| `formal_runner.py` | **dry-run / planning path.** In non-dry-run mode it refuses with *"Formal execution is intentionally disabled"*. |
| `formal_execute.py` | the **execution** path; requires `--execute`. This is what actually drove the campaign. |

`formal_execute.py` verifies the frozen namespace map against a **hard-coded SHA256** before doing
anything. The map is shipped as a **frozen input** at
`experiments/exp11_prefix_mbt/configs/exp11_formal_namespace_map.csv` — it is *not* regenerated at
run time.

**Generation order (if you ever rebuild the workload identity):**

```
1. namespace_map.py             -> produces exp11_formal_namespace_map.csv + _SHA256.txt
                                   (refuses to overwrite an existing map)
2. freeze it under configs/     -> this is the workload identity
3. formal_execute.py            -> verifies the SHA256 and refuses on mismatch
```

If you regenerate the map, its SHA256 will change and `formal_execute.py`'s hard-coded
`EXPECTED_NAMESPACE_MAP_SHA256` will **reject** it. That is the intended fail-closed behaviour, not
a bug: the frozen map *is* the preregistered workload identity.

Both the plan CSV and its SHA record are shipped under `configs/`. The plan's SHA256 **matches**
the value hard-coded in `formal_runner.py` — verified in this repository.

### 4.3 Exp12 — reference implementation only

`analyze_timeline.py` reads the **nsys-exported per-condition SQLite** and produces the marker
counts and steady-state spacing. That `.sqlite` is **not published** (hundreds of MB, unusable
without the matching GPU).

**Therefore it cannot reproduce the timeline output from a clean clone.** It ships as the
**analysis implementation** — the code that defines how the published numbers were computed —
alongside the frozen outputs in `results/`.

It has **no argument parser**.

The profiler procedure is documented in
[`../experiments/exp12_decode_profiling/profiling_commands.md`](../experiments/exp12_decode_profiling/profiling_commands.md)
as **recorded settings plus a clearly labelled reconstructed invocation** — the exact original
command was not recorded in the frozen evidence.

### 4.4 Preconditions you must reproduce

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

## 5. What can and cannot match

| Quantity | Reproducible elsewhere? |
|---|---|
| Ratio-style results (percentage changes, block sign consistency) | **Plausibly**, on similar hardware |
| Absolute latencies (ms) | **No** — machine-, driver-, and power-cap-specific |
| Kernel names and marker counts in Exp12 | **Only** with the same vLLM build and GPU |
| The `.nsys-rep` traces | **No** — not published (size) |

## 6. Validation performed in THIS repository

```
Level A  published aggregate verification   : PASS  (28 checks, 0 failures)
Level B  figure regeneration                : PASS
Level C  static / import validation         : PASS
Level D  fresh GPU serving reproduction     : NOT VERIFIED
Level E  Exp12 raw trace re-analysis        : NOT POSSIBLE (captures not published)
```

Prepared on a machine **without vLLM and without Nsight Systems**. No runtime result is claimed.
See [`public-repo-audit.md`](public-repo-audit.md).
