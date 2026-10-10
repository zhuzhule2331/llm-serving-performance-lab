# Repository audit

> Pre-publication and post-publication checks run against this repository.
> **No git operation is performed by any audit script.**

---

## 1. Inventory

| Metric | Value |
|---|---:|
| Total files | 54 |
| Python files | 19 |
| Repository size | ~0.9 MB |

---

## 2. Static checks

| Check | Result |
|---|---|
| `py_compile` on every `.py` | ✅ **19 / 19 PASS** |
| Import resolution | ✅ **0 unresolved** (modules blocked only by absent third-party packages are expected) |
| CLI surface | ✅ reported per entry point — see §4 |
| `scripts/check_environment.py` | ✅ runs; reports `ANALYSIS PATH: READY` |
| `scripts/verify_results.py` | ✅ **28 checks · 0 failures** |
| `figures/make_figures.py` | ✅ exit 0; 4 figures regenerated from frozen CSVs |

### Runtime

```
STATIC CHECK          : PASS
RUNTIME REPRODUCTION  : NOT EXECUTED
```

Prepared on a machine **without vLLM and without Nsight Systems**. **No runtime result is
claimed.** See the reproduction levels in [`reproduction-guide.md`](reproduction-guide.md).

---

## 3. Security / privacy

| Check | Result |
|---|---|
| Absolute local paths (POSIX home dirs, Windows user profiles) | ✅ **0** |
| Secrets (HF / OpenAI / AWS tokens, private-key blocks, password literals) | ✅ **0** |
| Private or company references | ✅ **0** |
| Model weights / checkpoints | ✅ **0** |
| Files > 10 MB · > 50 MB | ✅ **0 · 0** |
| Profiler binaries (`.nsys-rep` / `.ncu-rep` / `.sqlite`) | ✅ **0** |
| Archives (`.tar*` / `.zip` / `.zst`) | ✅ **0** |
| Symlinks · empty files | ✅ **0 · 0** |

### Change classes applied to the frozen code

| Class | Count | Behavioural effect |
|---|---:|---|
| Verbatim copy (provenance header only) | 12 | **NONE** |
| Rename only | 8 | **NONE** |
| Host path → environment variable | 12 | **POSSIBLE — path resolution only** |
| Import rewrite (consequence of renaming) | 9 sites / 5 files | **NONE — module resolution only** |
| Measurement / scheduling / statistical logic altered | **0** | — |

Per-file detail and SHA256: [`public-code-changelog.md`](public-code-changelog.md).

---

## 4. CLI surface (as actually implemented)

Only four entry points construct an `ArgumentParser`. Everything else is **archival / reference
code** and does not accept `--help` — see [`experiment-map.md`](experiment-map.md) §"Which files
are runnable".

| Entry point | `--help` | Notes |
|---|---|---|
| `common/request_client.py` | ✅ | the synchronized-burst client |
| `experiments/exp11_prefix_mbt/formal_runner.py` | ✅ | dry-run path only |
| `experiments/exp11_prefix_mbt/formal_execute.py` | ✅ | `--execute` required for a real run |
| `experiments/exp11_prefix_mbt/recover_block2.py` | ✅ | executes the block-replacement policy |
| `experiments/exp08_mbt_concurrency/runner.py` | ❌ | no parser — see §5 |
| `experiments/exp08_mbt_concurrency/preflight.py` | ❌ | no parser |
| `experiments/exp11_prefix_mbt/analyze_formal.py` | ❌ | no parser |
| `experiments/exp11_prefix_mbt/finalize_merge.py` | ❌ | no parser |
| `experiments/exp12_decode_profiling/analyze_timeline.py` | ❌ | no parser; offline analyzer |

---

## 5. Known gaps (stated, not hidden)

| # | Gap |
|---|---|
| 1 | The **exact `nsys profile` invocation was not recorded** in the frozen evidence. `profiling_commands.md` separates recorded settings from a clearly labelled reconstruction. The command was **not** fabricated. |
| 2 | Raw profiler captures and raw per-request logs are not published (size, and they embed host paths). |
| 3 | `analyze_timeline.py` cannot reproduce the timeline summary from a clean clone — its input `.sqlite` is not published. It ships as the analysis implementation. |
| 4 | Exp12 conditions A/B have no contemporaneous power-boundary record → `UNKNOWN`, not back-inferred. |
| 5 | No `LICENSE` file. Choosing one is the author's decision; `LICENSE_NOTE.md` lays out the options and deliberately does not pick. |

---

## 6. Claim boundary

The README and `docs/` use `associated with` / `consistent with` / `converging evidence`
throughout and never `proved`, `solely caused by`, or `universally optimal`.

Every headline number in the documentation is asserted against the frozen CSVs by
`scripts/verify_results.py`, so documentation drift fails loudly instead of going unnoticed.
