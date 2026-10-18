# INT8 LM Head —— 权重 INT8 量化 + 自研 CUDA GEMV 的 vLLM Decode 路径优化

针对 `Qwen3-VL-8B-Instruct` 在 vLLM 上的**单 Token Decode** 路径：把 LM Head 的 BF16 权重
换成**逐输出行对称 INT8**表示，用**手写 CUDA GEMV V2** 完成计算，然后分别测量
**算子级**和**服务级**到底改变了多少。

> **核心结论,连同它的作用域一起说:** 四次独立 microbenchmark 的算子加速比在
> **1.229× – 1.978×** 之间,最近三次复测的中位数为 **≈1.303×**。波动很大,**全部披露,不取最快一次代表整体**。
> ABBA/BAAB 共 96 次请求的客户端近似 TPOT 观测到 **平均约 4.14% 降幅**(CUDA Graph + `torch.compile`)。
> **这是两个不同层级的指标。** 二者都不是整模型加速;4.14% 是带计时与热漂移混杂的**观测值**,
> 不能声称已证明统计显著性。

---

## 1. 项目概述

LM Head 是 decode 的最后一次投影：`hidden [4096] → logits [151936]`。
batch=1 时它是**矩阵–向量乘**，瓶颈在**要读多少字节的权重**，而不是算力。
BF16 下该权重约 **1.16 GiB**，每个 decode step 都要完整读一遍。

本项目只问一个具体的工程问题：

> 如果 LM Head 权重以 INT8 存储、在自定义 GEMV kernel 内部完成反量化，
> **算子本身**能快多少？**端到端服务**能留下多少？**输出质量**付出什么代价？

实现内容：

| 组成 | 说明 |
|---|---|
| **量化器** | 逐输出行对称 INT8，`scale[r] = max_abs(row_r) / 127` |
| **CUDA kernel** | warp-per-row GEMV + warp shuffle 归约，FP32 累加(V1 → V2) |
| **PyTorch 扩展** | C++/pybind11 绑定，含 dtype / shape / device / 连续性检查 |
| **vLLM 集成** | `vllm.general_plugins` entry point，patch `LogitsProcessor._apply_head`，严格准入 + BF16 回退 |
| **Benchmark** | ABBA / BAAB 分阶段运行器，客户端流式 TPOT，逐请求记录 GPU 温度与时钟 |
| **质量评测** | 22 道客观题 + 13 道去重开放题盲评 |

**本项目不是什么：** 不是 INT8 GEMM，不用 Tensor Core INT8，不做 batch/prefill 优化，
也不是上游 vLLM 功能。**激活值保持 FP32，运算保持 FP32 —— INT8 只作用于权重存储。**

---

## 2. 硬件与软件环境

### 硬件

| 角色 | 设备 | 是否用于测量 |
|---|---|---|
| **远程服务器** | NVIDIA **RTX 3090 × 2**，各 24 GB | **仅 GPU0** |
| 本地开发机 | NVIDIA RTX 5060 | **否** —— 仅本地 Agent / 编辑 |

> ⚠️ **本仓库所有性能与质量数字均来自远程服务器的 GPU0(单张 RTX 3090)。**
> GPU1 未参与所报告的对照。**没有任何结果来自 RTX 5060。**

### 软件

| 组件 | 版本 |
|---|---|
| vLLM | **0.28.0** |
| 模型 | `Qwen3-VL-8B-Instruct`(BF16 权重) |
| CUDA 目标架构 | `sm_86`(扩展命名为 `int8_gemv_sm86_ext`) |
| 操作系统 | 远程服务器 WSL2 Linux |
| Python | **3.12.14** |
| PyTorch | **2.13.0+cu129** |
| Triton | **3.7.1** |
| CUDA Runtime | **12.9.79** |
| NVCC | **12.6.85**(`release 12.6, V12.6.85`) |
| GPU 驱动 | **572.83** |
| Nsight Systems | **未记录** |

来源：[`environment.json`](environment.json)(服务器端采集)。

> ⚠️ **CUDA 工具链版本不一致,如实记录:** 运行时是 CUDA **12.9.79**,但 `nvcc` 报告 **12.6.85**。
> 也就是说扩展由 12.6 工具链编译、加载进 12.9 运行时。这正是可能影响 ABI 与 kernel 选择的细节,
> 也是这些 microbenchmark 数字只能当作**机器特定结果**、不可移植的原因之一。
>
> 目前归档里仍然缺失的只有 Nsight Systems 版本。

### ABBA/BAAB 使用的服务配置

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

环境变量：`CUDA_VISIBLE_DEVICES=0`、`VLLM_PLUGINS=qwen_int8_head`、
`INT8_LM_HEAD_ENABLE=0|1`、`VLLM_WSL2_ENABLE_PIN_MEMORY=1`、
`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`HF_HUB_OFFLINE=1`。

---

## 3. Kernel 设计

### 数据布局

| 张量 | 形状 | 类型 |
|---|---|---|
| `Wq`(量化权重) | `[151936, 4096]` | `int8` |
| `scales` | `[151936]` | `float32` |
| `x`(激活) | `[4096]` | `float32` |
| `y`(输出) | `[151936]` | `float32` |

`y[r] = scales[r] * Σ_j Wq[r, j] * x[j]`

### 量化方案

逐**输出行**、关于零点对称：

```python
max_abs = w.abs().amax(dim=1)
scale   = max_abs / 127.0            # max_abs == 0 时取 1.0
q       = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
```

- 每个输出行一个 FP32 scale(151 936 个,合计约 594 KiB)。
- 取值范围 `[-127, 127]`，**有意不使用 `-128`**。
- 反量化**在 kernel 内部**写回时完成(`sum * scales[row]`),
  显存中不存在一份反量化后的 BF16 副本。

### V1 —— 一个 warp 一个 block

```
grid  = M          block = 32
row   = blockIdx.x
lane  = threadIdx.x
for j = lane; j < N; j += 32: sum += float(Wq[row*N + j]) * x[j]
warp shuffle 归约 → lane 0 写 y[row] = sum * scales[row]
```

正确，但会启动 151 936 个极小的 block，每个 block 的 SM 资源利用率很低。

### V2 —— 一个 block 四个 warp(所有测量使用的版本)

```
grid  = ceil(M / 4)     block = 128
warp_id = threadIdx.x / 32
lane    = threadIdx.x % 32
row     = blockIdx.x * 4 + warp_id
```

每个 warp 仍独占一个输出行，归约逻辑不变；把 4 行打包进一个 block，
**block 数量降为 1/4**，提高每 SM 占用率。

### 为什么可能更快

激活是 FP32、乘加是 FP32，因此**没有 INT8 算术吞吐优势**。收益来自**权重读取字节数**：

- BF16 权重:151 936 × 4096 × 2 B ≈ **1.16 GiB** / decode step
- INT8 权重:151 936 × 4096 × 1 B ≈ **0.58 GiB** / decode step

batch=1 时 LM Head 受权重带宽约束，**减半权重字节就是机制本身**。
代价是内层循环里每个元素多一次 `int8 → float` 转换。

### 正确性检查

`test_extension.py` 用合成张量对拍 PyTorch 参考；`test_real_lm_head.py` 用真实
`[151936, 4096]` 权重做同样对拍,并额外抽取 33 行与 CPU 参考交叉验证
(`atol=1e-3, rtol=1e-4`),之后才跑 microbenchmark。

---

## 4. vLLM 集成

集成方式是**插件**，不是 fork。`setup.py` 声明：

```python
entry_points={"vllm.general_plugins": ["qwen_int8_head=int8_vllm_plugin:register"]}
```

`setup.py` 与 `int8_vllm_plugin.py` **放在项目根目录** —— `py_modules=["int8_vllm_plugin"]`
和插件自身的 `Path(__file__).parent` 数据路径都要求这个布局。

**打包状态**（2026-10-18 静态检查）：

| 检查项 | 结果 |
|---|---|
| `python setup.py --name` / `--version` | ✅ `qwen-int8-lmhead-experiment` / `0.1.0` |
| `py_modules` 能找到文件 | ✅ `egg_info` 的 `SOURCES.txt` 列出 `int8_vllm_plugin.py` |
| 生成的 entry point | ✅ `qwen_int8_head = int8_vllm_plugin:register` |
| 模块可导入、`register` 可调用 | ✅ |
| 插件数据路径解析到项目根 | ✅ `<root>/artifacts/lm_head_int8_rowwise.safetensors` |
| **实际安装进 vLLM 环境并由 loader 调用** | ❌ **NOT VERIFIED** —— 准备机器上没有安装 vLLM |

> ⚠️ **安装方式会影响插件能否找到数据文件。** 插件用 `Path(__file__).parent` 定位：
>
> - `artifacts/lm_head_int8_rowwise.safetensors`
> - `build/`（编译好的扩展）
>
> **非 editable 的 `pip install .`** 会把 `int8_vllm_plugin.py` 复制进 `site-packages/`，
> 上述路径就会在 `site-packages/` 下解析，**必然找不到**。
> 应在**源码目录**里使用 **editable 安装**：
>
> ```bash
> cd int8-lm-head
> pip install -e .
> ```
>
> 这**只解决 Python 插件模块的定位**，**不负责** CUDA Extension 编译（见 §9.3），
> 也**不提供** `artifacts/lm_head_int8_rowwise.safetensors`（见 §9.4）。
> 这两项仍是前置条件，且**都不随仓库发布**。

`register()` 在 `INT8_LM_HEAD_ENABLE != 1` 时直接返回 —— 同一套环境既能跑 baseline 也能跑 INT8。

patch 目标是 `LogitsProcessor._apply_head`，**只有全部条件成立**才走自定义路径：

| 条件 | 原因 |
|---|---|
| `hidden_states.is_cuda` | kernel 需要 CUDA |
| `dtype == torch.bfloat16` | 与模型 head dtype 一致 |
| `shape[-1] == 4096` 且 `numel() == 4096` | **只支持单 Token Decode** |
| `lm_head.weight.shape == (151936, 4096)` | kernel 为固定形状特化 |
| `lm_head.tp_size == 1` | 不支持张量并行 |
| `embedding_bias is None` | kernel 无 bias 项 |
| `isinstance(quant_method, UnquantizedEmbeddingMethod)` | 不与其它量化器叠加 |
| `head_dtype is None or == hidden dtype` | 不支持混合 head dtype |

不满足任一条件即**回退到原始 BF16 实现**。首次命中时才惰性加载 INT8 权重并缓存在
`lm_head` 模块上。

前向路径：

```
hidden [1, 4096] bf16
  → reshape(-1).float().contiguous()      # [4096] fp32
  → int8_gemv_sm86_ext.forward(x, Wq, scales)
  → reshape(..., 151936).to(bfloat16)
```

诊断开关：`INT8_LM_HEAD_TRACE`、`INT8_LM_HEAD_PROFILE`(三步的 CUDA Events)、
`INT8_LM_HEAD_NSYS`(第 8–71 次 decode 调用外层 `cudaProfilerStart/Stop`)。

> ⚠️ **原始 BF16 LM Head 权重仍驻留显存。** INT8 副本是**额外加载**的。
> 因此本项目**不提供任何峰值显存节省的证据**。

---

## 5. Benchmark 方法

### 两个不同层级的测量

| 层级 | 工具 | 指标 | 回答什么 |
|---|---|---|---|
| 算子 | CUDA Events，交错顺序 | 单次 GEMV 的中位延迟 | 算子本身快多少 |
| 服务 | 客户端流式 HTTP | 128 输出 Token 的近似 TPOT | 端到端 decode 留下多少 |

**两者分开报告，绝不相乘。**

### 服务级 Benchmark 设计

跑两种顺序,让单调的温度/时钟漂移暴露为两次运行之间的差异：

```
ABBA:  A1=BF16 → B1=INT8 → B2=INT8 → A2=BF16
BAAB:  B1=INT8 → A1=BF16 → A2=BF16 → B2=INT8
```

- 每阶段 **12 次测量请求 + 3 次预热**，共 4 个阶段。
- **2 轮 × 4 阶段 × 12 = 96 次测量请求**(48 BF16 + 48 INT8)。
- 每次请求:`temperature=0`、`ignore_eos=True`、`max_tokens=128`、`stream=True`、
  `stream_options={"include_usage": True}`。
- **阶段之间重启服务**。运行器要求日志中出现 CUDA Graph 捕获确认
  (`Graph capturing finished`);INT8 阶段还要求 `Quantized weights loaded`,否则中止。
- 每次请求记录 GPU 温度、SM 时钟、显存时钟。
- GPU0 温度达到 **79 °C** 即中止。

### TPOT 定义(客户端近似)

```
TPOT ≈ (t_最后一个非空 content chunk − t_第一个非空 content chunk) / (completion_tokens − 1)
```

在 HTTP 客户端由 SSE chunk 到达时间计算。它包含**流式 chunk 粒度、客户端调度抖动**,
以及链路上可见的 token 间隔。**它不是 CUDA event 对 decode 循环的测量。**

### 有意不做的事

- 不用同步技巧放大两种模式的差异。
- 不丢弃任何请求。96 行全部在归档 CSV 中,包括高于阶段均值的行。
- **不计算 p 值**。两种顺序的幅度不一致(见下),这正是"不能声称显著性"的原因。

---

## 6. 性能结果

### 6.1 算子级 microbenchmark —— 四次运行

四次都使用真实 `Qwen3-VL-8B-Instruct` 的 `lm_head.weight`（`[151936, 4096]`，BF16 源）
与一个合成 FP32 激活向量，运行在**远程服务器 GPU0（RTX 3090）**。

| 运行 | BF16 `torch.mm` | INT8 GEMV V2 | 加速比 | 原始日志 |
|---|---:|---:|---:|---|
| 初始复测 | 1.6502 ms | 1.3430 ms | **1.229×** | [`microbenchmark_real_weight_20261018_202616.txt`](artifacts/profiling/microbenchmark_real_weight_20261018_202616.txt) |
| 复测 1 | 1.5713 ms | 1.2058 ms | **1.303×** | [`run_1.txt`](artifacts/profiling/microbenchmark_repeat_20261018/run_1.txt) |
| 复测 2 | 1.4141 ms | 1.0885 ms | **1.299×** | [`run_2.txt`](artifacts/profiling/microbenchmark_repeat_20261018/run_2.txt) |
| 复测 3 | 1.4100 ms | 0.7127 ms | **1.978×** | [`run_3.txt`](artifacts/profiling/microbenchmark_repeat_20261018/run_3.txt) |

- **最近三次复测的加速比中位数：1.303×。** 四次全部的中位数：1.301×。
- 四次全距：**1.229× – 1.978×**。
- **1.978× 是离群值,不是代表值。** 仍然列出,因为删掉它会掩盖真实的波动。
- 机器可读记录：
  [`summary.json`](artifacts/profiling/microbenchmark_repeat_20261018/summary.json)。

**为什么波动这么大。** 每次复测前的 GPU 温度依次为 **33 °C → 43 °C → 45 °C**,
INT8 中位数单调下降（1.3430 → 1.2058 → 1.0885 → 0.7127 ms）,
BF16 中位数也在漂移（1.6502 → 1.5713 → 1.4141 → 1.4100 ms）。
**分子和分母同时在动,所以比值不稳定。** 几次运行相隔仅数秒,时钟 boost 状态与 page cache
热度都可能是原因；**本归档没有把它们分离开。**

**计时不稳定,但正确性稳定。** 每一次运行都报告：
`NRMSE ≈ 0.88–0.93%`、三个 trial 全部 `Top1Equal=True`、
`INT8 GPU vs CPU reference: PASS (33 rows)`。

> **历史数字,不是当前结论。** 更早的一次运行报告过 **1.645×**（1.4116 → 0.8581 ms）。
> **它的原始 stdout 不在本归档中。** 只作为历史记录保留,**不得当作本项目的加速比引用**。

**作用域。** 固定形状、单个输入向量、每次运行 20 次交错测量的中位数。
**不是整模型加速**,也**不能外推**到其他形状、batch size 或 GPU。
**离线量化与输入 `bf16 → fp32` 转换不计入测量区间。**

### 6.2 端到端服务(CUDA Graph + `torch.compile`)

直接由 `artifacts/profiling/*/results.csv` 计算：

| 运行 | BF16 平均 TPOT | INT8 平均 TPOT | 差 | 比值 |
|---|---:|---:|---:|---:|
| ABBA(`graph_abba_20261018_180248`) | 25.119 ms | 24.244 ms | −0.876 ms | 1.036× |
| BAAB(`graph_baab_20261018_184013`) | 25.618 ms | 24.392 ms | −1.226 ms | 1.050× |
| **合并(n = 96)** | **25.369 ms** | **24.318 ms** | **−1.051 ms** | **1.043×** |

**近似 TPOT 平均降幅:4.14%。**

各阶段中位数(来自 `summary.json`)：

| 阶段 | ABBA 中位数 | BAAB 中位数 |
|---|---:|---:|
| BF16(A1) | 24.385 ms | 25.581 ms |
| INT8(B1) | 24.078 ms | 23.751 ms |
| INT8(B2) | 24.385 ms | 24.855 ms |
| BF16(A2) | 25.774 ms | 25.816 ms |

### 如何诚实解读

- **两种顺序给出的幅度不同**(3.49% vs 4.78%)。顺序、热漂移与时钟状态**没有被两轮运行完全分离**。
- ABBA 中 INT8 的 B2(24.385 ms)与 BF16 的 A1(24.385 ms)**完全相同**。效应**不是干净的分阶段分离**。
- 服务路径运行在 **CUDA Graph + `torch.compile`** 下,因此测得的差值
  **不能单独归因于 CUDA kernel** —— graph replay、编译决策、插件开销都有贡献。
- **Eager 模式已集成并验证调用,但其端到端优势不稳健**,本项目不对此作声称。
- **不报告置信区间或显著性检验**,也不应从中推断。

### 6.3 执行模式

| 模式 | 发生什么 | 本项目状态 |
|---|---|---|
| **Eager** | 每个 decode step 都执行 patch 后的 `_apply_head` | 已集成验证;**端到端收益不稳健** |
| **CUDA Graph + `torch.compile`** | decode step 被捕获并重放,自定义算子进入捕获图 | 已集成;**96 次正式请求均在此模式下采集** |

---

## 7. 质量评测

### 7.1 客观题(22 题,自动评分)

| 类别 | 题数 | BF16 | INT8 |
|---|---:|---:|---:|
| 基本计算能力 | 5 | 5 | 5 |
| 基本推理思考能力 | 5 | 2 | 2 |
| 进阶解题能力 | 3 | 3 | 3 |
| 进阶计算能力 | 5 | 4 | 4 |
| 进阶推理能力 | 4 | 4 | 3 |
| **合计** | **22** | **18** | **17** |

唯一的 1 分差距来自 **Q026**:INT8 在 384 Token 预算内未形成可解析答案。
随后做了 **768 Token 定向复测:两个模型均 3/3 正确**。
复测与原始成绩**并列报告,不覆盖原始成绩**。

Q011、Q015 两个模型都因机械输出 `答案：结果` 导致自动解析失败,
而解析前的正文方向正确 —— 因此自动分数**不是纯知识准确率**。

### 7.2 开放题(13 道去重题,单次盲评)

| 切片 | BF16 | INT8 |
|---|---:|---:|
| 仅原始回答(13 题) | 37/52 | 38/52 |
| 仅长回答复测(8 题) | 22/32 | 20/32 |
| **去重 13 题(用于 headline)** | **41/52**(78.85%) | **39/52**(75.00%) |

两分差距完全来自 **Q032(4 vs 3)** 与 **Q033(3 vs 2)**,其余 11 题评分相同。

**可追溯性。** 逐题分数与揭盲映射在
[`artifacts/quality_ab/blind_review_unblinded_scores.csv`](artifacts/quality_ab/blind_review_unblinded_scores.csv)
（21 行：13 行原始 + 8 行长回答复测）。headline 的取法是 Q001–Q005 取原始行、其余 8 题取复测行，
求和即 **41/52 vs 39/52**。完整推导见
[`docs/quality-evaluation.md`](docs/quality-evaluation.md#35-unblinding-and-deduplication-how-4152-and-3952-are-computed)。

> ⚠️ `blind_review.json` 是**未评分的盲评包**（每一行的 `score_X_0_to_4` 与
> `score_Y_0_to_4` 都是 `null`），**不是已评分文件**。分数在上面那份 `_unblinded_scores.csv` 里。
>
> 这是**单次主观评审**：没有第二位评审、没有评审者间一致性、没有假设检验。
> **41/52 vs 39/52 不是统计显著的质量结论。**

### 7.3 可以说 / 不可以说

| ✅ 可以说 | ❌ 不可以说 |
|---|---|
| INT8 输出在少数题上与 BF16 不同 | "无损" / "lossless" |
| 某一套 22 题上客观分 18/22 vs 17/22 | "模型能力下降 4.5%" |
| 盲评 41/52 vs 39/52,单一评审,回答被截断 | "质量退化已被统计证明" |

这是**一名评审对截断回答的一次主观审阅** + 每题一到数次生成,是 sanity check,不是质量研究。

---

## 8. 局限

1. **算子加速 ≠ 服务加速。** 算子级是形状特化的 microbenchmark,**四次运行在 1.229×–1.978× 之间,最近三次中位 1.303×**;4.14% 是 CUDA Graph 下的客户端端到端观测。**不要相乘,不要互相替代,也不要把单次运行当成"那个"加速比。**
2. **无统计显著性。** 两种顺序、每阶段 12 次请求、无置信区间、存在可观测的热/时钟漂移。
3. **客户端计时。** TPOT 来自 SSE chunk 到达时间,不是 CUDA event 或服务端埋点。
4. **CUDA Graph 混杂。** 端到端数字在 graph replay + `torch.compile` 下测得,kernel 的独立贡献未被分离。
5. **Eager 模式未建立。** 集成可用,但没有稳健的端到端优势。
6. **未证明显存节省。** BF16 LM Head 仍在显存中,INT8 是额外的一份。
7. **不是严格无损。** 逐行 scale 引入量化误差,质量评测显示小幅、依题而异的差异。
8. **适用范围窄。** 权重 INT8、batch=1 单 Token Decode、单模型、单形状、单 GPU。**不是 prefill 或 batched GEMM 优化。**
9. **未合入上游。** 这是 `vllm.general_plugins` 的 monkey-patch,不是被 vLLM 接受的改动。
10. **microbenchmark 波动大且未解释。** 同一台机器四次运行跨越 1.229×–1.978×。温度与时钟状态和漂移相关,但本归档没有定位原因。
11. **环境记录仍有缺口。** Python / PyTorch / Triton / CUDA runtime / NVCC / 驱动 已记录在 [`environment.json`](environment.json);**Nsight Systems 版本仍缺失**。
12. **不包含原始 profiler trace。** `.nsys-rep` / `.sqlite` 数百 MB,脱离同一块 GPU 也无意义;仓库只放分析脚本与派生摘要。

---

## 9. 复现步骤

### 9.1 前置条件

- `sm_86` 的 CUDA GPU(本项目使用单张 RTX 3090)
- vLLM **0.28.0**(支持 `VLLM_PLUGINS`)
- 本地可用的 `Qwen3-VL-8B-Instruct` 权重
- PyTorch + CUDA、`safetensors`、`requests`

### 9.2 环境变量

| 变量 | 含义 |
|---|---|
| `QWEN_MODEL_DIR` | `Qwen3-VL-8B-Instruct` 的 snapshot 目录 |
| `QWEN_LM_HEAD_SHARD` | 含 `lm_head.weight` 的 safetensors 分片 |
| `INT8_LM_HEAD_ENABLE` | `1` = 自定义 INT8 路径,`0` = BF16 baseline |
| `INT8_LM_HEAD_TRACE` | `1` = 打印 custom/fallback 计数 |
| `INT8_LM_HEAD_PROFILE` | `1` = 三步的 CUDA Events |
| `INT8_LM_HEAD_NSYS` | `1` = 第 8–71 次 decode 调用外层 `cudaProfilerStart/Stop` |

所有脚本以**项目根目录**(`int8-lm-head/`)为工作目录运行,因为它们写入 `artifacts/`。

### 9.3 编译扩展 —— **NOT VERIFIED**

> ⚠️ **`setup.py` 不是 CUDA 扩展构建脚本。** 它只声明 vLLM 插件 entry point
> （`vllm.general_plugins`）。在**源码目录**执行 **`pip install -e .`** 只会以 editable
> 方式安装插件注册，**不会**产生 `int8_gemv_sm86_ext`。
>
> 请务必带 `-e`：非 editable 的 `pip install .` 会把模块复制进 `site-packages/`，
> 破坏插件的 `Path(__file__).parent` 数据路径（见 §4）。
>
> **这一步不编译任何 CUDA 代码。**

归档**能够确认**的事实：

| 证据 | 出处 |
|---|---|
| 扩展模块名为 `int8_gemv_sm86_ext` | `scripts/test_real_lm_head.py`、`int8_vllm_plugin.py` |
| 它从插件旁边的 `build/` 目录导入 | 同上（`sys.path.insert(0, .../"build")`） |
| 绑定使用 `torch/extension.h` + `PYBIND11_MODULE` | `src/int8_binding.cpp` |
| `int8_launch.cu` `#include` 了 `int8_gemv_v2.cu` | `src/int8_launch.cu` |
| CUDA 目标架构为 `sm_86` | 模块名 + `environment.json` 中的 RTX 3090 |

**确切的构建命令不在归档中** —— 没有扩展的 `setup.py`、没有 `build.ninja`、没有 shell history。因此：

> ❌ **本 README 有意不提供构建命令。** 写在这里的任何命令都是重建的,不是验证过的。
> ✅ 重建需要从上述源码推导,**本仓库不保证其可复现性（NOT VERIFIED）**。
> 在你自行重建之前,应把扩展二进制视为不可用。

### 9.4 量化 LM Head

```bash
export QWEN_LM_HEAD_SHARD=/path/to/model-00004-of-00004.safetensors
python scripts/export_int8_lm_head.py
# 输出 artifacts/lm_head_int8_rowwise.safetensors(约 594 MiB)
```

### 9.5 Kernel 正确性与 microbenchmark

```bash
python scripts/test_extension.py        # 合成形状
python scripts/test_real_lm_head.py     # 真实权重 + microbenchmark
```

### 9.6 服务级 benchmark

```bash
# 终端 1 —— 为当前阶段启动服务(INT8_LM_HEAD_ENABLE=0/1)
INT8_LM_HEAD_ENABLE=0 bash scripts/launch_server.sh

# 终端 2 —— 有序对照(脚本会自行启停服务)
python scripts/run_graph_abba.py
python scripts/run_graph_baab.py
```

### 9.7 质量评测

```bash
python scripts/quality_ab_suite.py      # 需要 BF16 在 :8001、INT8 在 :8002
python scripts/retest_q026.py
python scripts/retest_open_truncated.py
```

### 9.8 分析脚本

`scripts/analyze_decode_timeline.py`、`analyze_gap_api.py`、`analyze_gap_neighbors.py`、
`analyze_gpu_gaps.py` 读取本地导出的 Nsight Systems `.sqlite`。
**trace 本身不包含在仓库中**,因此这些脚本**无法开箱即用**。

---

## 目录结构

```
int8-lm-head/
├── setup.py        vLLM 插件打包（entry point）—— 不是 CUDA 构建脚本
├── int8_vllm_plugin.py  插件本体（必须与 setup.py 同级）
├── src/            CUDA kernel + pybind11 绑定
├── scripts/        量化、benchmark、质量评测、分析
├── artifacts/
│   ├── profiling/  ABBA / BAAB 的 results.csv + summary.json
│   │               microbenchmark：1 份完整日志 + 3 次复测日志 + summary.json
│   └── quality_ab/ 客观题 + 开放题评测 JSON
├── docs/           方法、质量评测、局限、代码变更记录
├── environment.json 服务器软件版本
├── README.md       英文版
└── README.zh-CN.md 本文件
```

## 许可与出处

见仓库根目录 [`LICENSE_NOTE.md`](../LICENSE_NOTE.md)。
`artifacts/` 下的数字均复制自原始服务器运行结果;发布过程中唯一改动是**绝对宿主路径替换**,
逐条记录在 [`docs/public-code-changelog.md`](docs/public-code-changelog.md)。
**测量、调度、统计逻辑一字未改。**
