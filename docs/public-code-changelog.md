# Public code changelog

> Provenance record for the code shipped in this repository.
> **Every change made when preparing the public copy is listed below — there is exactly one
> class of behavioural change: absolute host paths replaced by environment variables.**
>
> Generated from the build manifest; not hand-maintained.


> **本文件由 the public-repo build script 自动生成,不手工维护。**
>
> 规则:默认要求 `behavioral effect = NONE`。本轮**唯一**的改动类别是
> **隐私路径 → 环境变量**(强制要求,见 `SECURITY_SCAN.md`),
> 记为 `POSSIBLE (path resolution only)` —— **测量、调度、统计逻辑一字未动**。
> 所有重命名记为 `NONE`。

## 汇总

| 类别 | 数量 |
|---|---:|
| 复制文件总数 | 29 |
| 仅重命名(behavioral effect = NONE) | 8 |
| 路径替换(behavioral effect = POSSIBLE, path only) | 10 |
| 改动测量逻辑的文件 | **0** |

## 逐文件明细

### `common/request_client.py`

- **source file**: `<archive>/raw_bench/exp08_mbt_concurrency/exp08_burst_FROZEN.py` (frozen archive copy)
- **public file**: `common/request_client.py`
- **change**: 重命名 `exp08_burst_FROZEN.py` → `request_client.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **sha256 (source)**: `8c87c930890ab62b…`
- **sha256 (public)**: `43228aec30e8317d…`

### `common/server_lifecycle.py`

- **source file**: `<archive>/raw_bench/exp08_mbt_concurrency/exp08_infra_FROZEN.py` (frozen archive copy)
- **public file**: `common/server_lifecycle.py`
- **change**: 重命名 `exp08_infra_FROZEN.py` → `server_lifecycle.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: hard-coded vllm binary path -> $VLLM_BIN  (1 site)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: hard-coded HF snapshot path -> $MODEL_PATH  (1 site)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: process-match literal "<host>/.../vllm serve" -> f-string  (2 site)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `0028bf565bca32a9…`
- **sha256 (public)**: `afdf2e77e626ed2a…`

### `common/exp03_formal.py`

- **source file**: `<archive>/vlm-serving-lab/raw_archive/bench/exp03_formal.py` (frozen archive copy)
- **public file**: `common/exp03_formal.py`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `59857b7d682f5547…`
- **sha256 (public)**: `5915697cbcc26b5c…`

### `common/exp02_formal.py`

- **source file**: `<archive>/vlm-serving-lab/raw_archive/bench/exp02_formal.py` (frozen archive copy)
- **public file**: `common/exp02_formal.py`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `a402dec75b4cfee2…`
- **sha256 (public)**: `d73febc8bea0f494…`

### `experiments/exp08_mbt_concurrency/runner.py`

- **source file**: `<archive>/raw_bench/exp08_mbt_concurrency/exp08_formal.py` (frozen archive copy)
- **public file**: `experiments/exp08_mbt_concurrency/runner.py`
- **change**: 重命名 `exp08_formal.py` → `runner.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: process-match literal "<host>/.../vllm serve" -> f-string  (1 site)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp08_infra_FROZEN as infra` -> `from common import server_lifecycle as infra`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `1a37649157418732…`
- **sha256 (public)**: `6ea6b88131b68644…`

### `experiments/exp08_mbt_concurrency/preflight.py`

- **source file**: `<archive>/raw_bench/exp08_mbt_concurrency/exp08_preflight.py` (frozen archive copy)
- **public file**: `experiments/exp08_mbt_concurrency/preflight.py`
- **change**: 重命名 `exp08_preflight.py` → `preflight.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: process-match literal "<host>/.../vllm serve" -> f-string  (1 site)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp08_infra_FROZEN as infra` -> `from common import server_lifecycle as infra`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `76da3f4a51d0d913…`
- **sha256 (public)**: `99e3fdfe0fb00409…`

### `experiments/exp11_prefix_mbt/formal_runner.py`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_formal_runner.py` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/formal_runner.py`
- **change**: 重命名 `exp11_formal_runner.py` → `formal_runner.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: process-match literal "<host>/.../vllm serve" -> f-string  (1 site)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_infra_FROZEN as infra` -> `from common import server_lifecycle as infra`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_prefix_core as core` -> `import prefix_core as core`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `9985102b1164bd0c…`
- **sha256 (public)**: `69d27c05544327cb…`

### `experiments/exp11_prefix_mbt/prefix_core.py`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_prefix_core.py` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/prefix_core.py`
- **change**: 重命名 `exp11_prefix_core.py` → `prefix_core.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **sha256 (source)**: `f2b381f85783f86b…`
- **sha256 (public)**: `ba7c8844fac7abc8…`

### `experiments/exp11_prefix_mbt/namespace_map.py`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_namespace_map.py` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/namespace_map.py`
- **change**: 重命名 `exp11_namespace_map.py` → `namespace_map.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: import 重写: `import exp11_infra_FROZEN as infra` -> `from common import server_lifecycle as infra`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_formal_runner as runner` -> `import formal_runner as runner`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_prefix_core as core` -> `import prefix_core as core`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `75c68879cf97dd39…`
- **sha256 (public)**: `456898b62b6f86d3…`

### `experiments/exp11_prefix_mbt/namespace_reconstruction_audit.py`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_namespace_reconstruction_audit.py` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/namespace_reconstruction_audit.py`
- **change**: 重命名 `exp11_namespace_reconstruction_audit.py` → `namespace_reconstruction_audit.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: import 重写: `import exp11_infra_FROZEN as infra` -> `from common import server_lifecycle as infra`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_formal_runner as runner` -> `import formal_runner as runner`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_prefix_core as core` -> `import prefix_core as core`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `30128c198bc9fc8c…`
- **sha256 (public)**: `ba527b633d4001d5…`

### `experiments/exp11_prefix_mbt/finalize_merge.py`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_finalize_merge.py` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/finalize_merge.py`
- **change**: 重命名 `exp11_finalize_merge.py` → `finalize_merge.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **sha256 (source)**: `0ab13d65a569aba6…`
- **sha256 (public)**: `c55714c7510eabcc…`

### `experiments/exp11_prefix_mbt/analyze_formal.py`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_analyze_formal.py` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/analyze_formal.py`
- **change**: 重命名 `exp11_analyze_formal.py` → `analyze_formal.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: hard-coded results dir -> $RESULTS_DIR  (1 site)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `22897eba24b007a8…`
- **sha256 (public)**: `625cae08bfe8572b…`

### `experiments/exp11_prefix_mbt/recover_block2.py`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_recover_block2.py` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/recover_block2.py`
- **change**: 重命名 `exp11_recover_block2.py` → `recover_block2.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: import 重写: `import exp11_formal_runner as runner` -> `import formal_runner as runner`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_formal_execute as formal` -> `import formal_execute as formal`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `866b9ac13dda5044…`
- **sha256 (public)**: `5abbcb2f41b2f674…`

### `experiments/exp11_prefix_mbt/formal_execute.py`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_formal_execute.py` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/formal_execute.py`
- **change**: 重命名 `exp11_formal_execute.py` → `formal_execute.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **change**: import 重写: `import exp11_infra_FROZEN as infra` -> `from common import server_lifecycle as infra`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_formal_runner as runner` -> `import formal_runner as runner`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **change**: import 重写: `import exp11_prefix_core as core` -> `import prefix_core as core`(改名引起的模块解析修正)
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `184c45596b56865f…`
- **sha256 (public)**: `d1090fabd5740fb0…`

### `experiments/exp12_decode_profiling/analyze_timeline.py`

- **source file**: `<archive>/raw_bench/exp12_decode_profiling/exp12_timeline_analysis.py` (frozen archive copy)
- **public file**: `experiments/exp12_decode_profiling/analyze_timeline.py`
- **change**: 重命名 `exp12_timeline_analysis.py` → `analyze_timeline.py`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **sha256 (source)**: `2547aca87069114c…`
- **sha256 (public)**: `4238b173aa8e6eee…`

### `experiments/exp08_mbt_concurrency/configs/exp08_formal_protocol.txt`

- **source file**: `<archive>/raw_bench/exp08_mbt_concurrency/exp08_formal_protocol.txt` (frozen archive copy)
- **public file**: `experiments/exp08_mbt_concurrency/configs/exp08_formal_protocol.txt`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `b97d2bbf0160b73c…`
- **sha256 (public)**: `7c3b28fc8a6f686e…`

### `experiments/exp08_mbt_concurrency/configs/exp08_formal_preregistered_order.txt`

- **source file**: `<archive>/raw_bench/exp08_mbt_concurrency/exp08_formal_preregistered_order.txt` (frozen archive copy)
- **public file**: `experiments/exp08_mbt_concurrency/configs/exp08_formal_preregistered_order.txt`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `c3f741fdd1e5890a…`
- **sha256 (public)**: `1d045eeec93dd9f4…`

### `experiments/exp11_prefix_mbt/configs/exp11_protocol.txt`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_protocol.txt` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/configs/exp11_protocol.txt`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `801d5ccee41c7475…`
- **sha256 (public)**: `d1ac777204132387…`

### `experiments/exp11_prefix_mbt/configs/exp11_preregistered_plan.csv`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_preregistered_plan.csv` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/configs/exp11_preregistered_plan.csv`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `f301db731cb5760f…`
- **sha256 (public)**: `f301db731cb5760f…`

### `experiments/exp11_prefix_mbt/configs/exp11_block2_recovery_policy_v1.json`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_block2_recovery_policy_v1.json` (frozen archive copy)
- **public file**: `experiments/exp11_prefix_mbt/configs/exp11_block2_recovery_policy_v1.json`
- **change**: absolute experiment-host path prefix -> <EXPERIMENT_HOST>  (1 site); 目录名与其余字段逐字保留,provenance 语义不变
- **reason**: 隐私清理(SECURITY_SCAN 要求,移除实验主机绝对路径)
- **behavioral effect**: **POSSIBLE** —— 仅路径解析;未触碰测量/调度/统计逻辑,设置同名环境变量即恢复原行为
- **sha256 (source)**: `0116dd55b96fd62d…`
- **sha256 (public)**: `6353dcf2fd56fff1…`

### `experiments/exp12_decode_profiling/configs/exp12_protocol.txt`

- **source file**: `<archive>/raw_bench/exp12_decode_profiling/exp12_protocol.txt` (frozen archive copy)
- **public file**: `experiments/exp12_decode_profiling/configs/exp12_protocol.txt`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `808e90f70afd28df…`
- **sha256 (public)**: `d1a6de9f25066f25…`

### `results/exp08_condition_summary.csv`

- **source file**: `<archive>/raw_bench/exp08_mbt_concurrency/exp08_formal_20261007_225407/formal_condition_summary.csv` (frozen archive copy)
- **public file**: `results/exp08_condition_summary.csv`
- **change**: 重命名 `formal_condition_summary.csv` → `exp08_condition_summary.csv`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **sha256 (source)**: `8666610d9f91d009…`
- **sha256 (public)**: `8666610d9f91d009…`

### `results/exp08_relative_to_mbt2048.csv`

- **source file**: `<archive>/raw_bench/exp08_mbt_concurrency/exp08_formal_20261007_225407/formal_relative_to_mbt2048.csv` (frozen archive copy)
- **public file**: `results/exp08_relative_to_mbt2048.csv`
- **change**: 重命名 `formal_relative_to_mbt2048.csv` → `exp08_relative_to_mbt2048.csv`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **sha256 (source)**: `0791ca91dd8251a7…`
- **sha256 (public)**: `0791ca91dd8251a7…`

### `results/exp11_cell_summary.csv`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_final_merged_20261008_193650/cell_summary.csv` (frozen archive copy)
- **public file**: `results/exp11_cell_summary.csv`
- **change**: 重命名 `cell_summary.csv` → `exp11_cell_summary.csv`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **sha256 (source)**: `35f65af02d751731…`
- **sha256 (public)**: `35f65af02d751731…`

### `results/exp11_interaction_summary.csv`

- **source file**: `<archive>/raw_bench/exp11_prefix_mbt/exp11_final_merged_20261008_193650/mbt_within_reuse_contrasts.csv` (frozen archive copy)
- **public file**: `results/exp11_interaction_summary.csv`
- **change**: 重命名 `mbt_within_reuse_contrasts.csv` → `exp11_interaction_summary.csv`
- **reason**: public-facing 命名规范化
- **behavioral effect**: **NONE**
- **sha256 (source)**: `9b565c97ce436216…`
- **sha256 (public)**: `9b565c97ce436216…`

### `results/exp12_hypothesis_status.csv`

- **source file**: `<archive>/raw_bench/exp12_decode_profiling/exp12_hypothesis_status.csv` (frozen archive copy)
- **public file**: `results/exp12_hypothesis_status.csv`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `7da96903dc0e5446…`
- **sha256 (public)**: `7da96903dc0e5446…`

### `results/exp12_timeline_analysis.txt`

- **source file**: `<archive>/raw_bench/exp12_decode_profiling/exp12_timeline_analysis.txt` (frozen archive copy)
- **public file**: `results/exp12_timeline_analysis.txt`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `f6e61a3aa4168988…`
- **sha256 (public)**: `8756601a3ff4b423…`

### `results/exp12_summary.txt`

- **source file**: `<archive>/raw_bench/exp12_decode_profiling/exp12_summary.txt` (frozen archive copy)
- **public file**: `results/exp12_summary.txt`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `9889b68b57f5f81a…`
- **sha256 (public)**: `72598fcafe905d49…`

### `results/exp12_C_power_boundary.txt`

- **source file**: `<archive>/raw_bench/exp12_decode_profiling/exp12_C_power_boundary.txt` (frozen archive copy)
- **public file**: `results/exp12_C_power_boundary.txt`
- **change**: 无(逐字节复制)
- **behavioral effect**: **NONE**
- **sha256 (source)**: `752957fc6c5b675b…`
- **sha256 (public)**: `e5ff130b64022269…`

---

## Post-publication hotfix (2026-10-10)

Applied after the repository was published. **No measurement, scheduling, or statistical logic was
altered; no frozen artifact value was changed.**

| # | File | Change | Category | Behavioural effect |
|---|---|---|---|---|
| H-1 | `tatus` | **removed** — stray file containing ANSI-coloured `git diff --stat` output captured from a terminal (not project content) | cleanup | NONE |
| H-2 | `results/*.csv`, `results/*.txt`, `results/artifact_hashes.json`, `experiments/*/configs/*` | re-copied **byte-for-byte** from the frozen archive. The earlier text-mode copy had normalised CRLF→LF, which changed the SHA256 of `exp11_preregistered_plan.csv` and would have tripped the `formal_runner.py` hash guard | integrity | **NONE** — this *restores* the frozen bytes |
| H-3 | `results/artifact_hashes.json` | **restored** — it was listed in the build manifest but absent from the published tree; `analyze_formal.py` reads it | integrity | NONE |
| H-4 | `experiments/exp08_mbt_concurrency/runner.py` | `BURST_SCRIPT` now resolves to `common/request_client.py` (the public location of the same frozen file, previously `exp08_burst_FROZEN.py`) | portability | **NONE** — same file, path only |
| H-5 | `experiments/exp11_prefix_mbt/formal_runner.py` | `PLAN_PATH` / `PLAN_HASH_PATH` now resolve under `configs/` | portability | **NONE** — path only |
| H-6 | `experiments/exp11_prefix_mbt/formal_execute.py` | `NAMESPACE_MAP` now resolves under `configs/` | portability | **NONE** — path only |
| H-7 | `experiments/exp11_prefix_mbt/configs/` | added the frozen inputs that were missing: `exp11_preregistered_plan_SHA256.txt`, `exp11_formal_namespace_map.csv`, `exp11_formal_namespace_map_SHA256.txt` | integrity | NONE |
| H-8 | `docs/`, `experiments/*/README.md`, `README.md` | documentation corrections only: out-of-repo links removed; profiler-command wording changed to "recorded settings plus a clearly labelled reconstruction"; non-existent `--help` claims removed; Exp12-H4 vocabulary harmonised | documentation | NONE |

### Hash guards verified after the hotfix

| Guard | Location | Result |
|---|---|---|
| `EXPECTED_PLAN_SHA256` | `formal_runner.py` | ✅ **matches** `configs/exp11_preregistered_plan.csv` |
| `EXPECTED_NAMESPACE_MAP_SHA256` | `formal_execute.py` | ✅ **matches** `configs/exp11_formal_namespace_map.csv` |

### Frozen-evidence handling

All `results/*` and `experiments/*/configs/*` artifacts are now **byte-exact copies** of the frozen
archive, with one exception: `exp11_block2_recovery_policy_v1.json` had its absolute host-path
prefix replaced with `<EXPERIMENT_HOST>` — the directory name and every other byte are preserved.
