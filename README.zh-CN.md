# LLM Serving Performance Lab（中文入口）

本仓库包含两组相互独立的推理性能研究：

1. **vLLM Scheduler / Prefix Cache 实验**：研究 `max_num_batched_tokens` 与并发、前缀复用之间的交互。
2. **INT8 LM Head CUDA 优化**：权重 INT8 + 手写 CUDA GEMV，并分别测算子级与服务级。

完整英文说明见 [`README.md`](README.md)。

---

## 最新：INT8 LM Head vec4

在 `Qwen3-VL-8B-Instruct` 的单 token decode 路径上，对 LM Head `[151936, 4096]` 做
**weight-only INT8** 与 **char4+float4 vectorized GEMV**：

| 层级 | 结果 | 口径 |
|---|---|---|
| 算子 | BF16 **2.245 ms** → V2 **1.459 ms** → vec4 **1.040 ms**；BF16→vec4 **2.159×** | 单算子，RTX 3090 GPU0 |
| NCU | global-load 指令 **−74.7%**；LSU **94.9%→27.2%**；naïve char4 sectors **3.39× ideal**；final 恢复 ideal；DRAM **≈882 GB/s**（**96.8%** sustained peak） | 单 kernel probe |
| 服务 | BF16 **25.5922 ms** → vec4 **24.7129 ms**；客户端近似 TPOT **−3.44%** | 32-request ABBA，描述性 |

**边界：**

- `2.159×` 是 LM Head operator 级，不是整模型或 serving 加速。
- `−3.44%` 是 32-request ABBA 的客户端近似 TPOT；这是描述性观测，未做显著性检验，
  不宣称显著收益。
- BF16 A1→A2 漂移约 3.61%；CUDA Graph + `torch.compile` 是测量路径的一部分。
- BF16 LM Head 权重仍驻留；INT8 是额外副本，不宣称显存节省。
- 不是 W8A8 / Tensor-Core INT8；不宣称量化无损；不是 vLLM upstream。

详见：

- [`int8-lm-head/vec4/README.md`](int8-lm-head/vec4/README.md)
- [`operator/final_bench_results.json`](int8-lm-head/vec4/operator/final_bench_results.json)
- [`ncu/ncu_key_metrics.json`](int8-lm-head/vec4/ncu/ncu_key_metrics.json)
- [`serving/results.csv`](int8-lm-head/vec4/serving/results.csv)

---

## 历史 V2

`int8-lm-head/` 中的 V2 是较早的实验，保留为历史基线：

- 算子级四次运行：**1.229×–1.978×**，三次复测中位 **≈1.303×**；
- 96-request ABBA/BAAB 客户端近似 TPOT：**≈−4.14%**。

V2 与 vec4 是不同实验，不能直接比较、相加，也不能把 V2 的数字替换成 vec4 的数字。

---

## Scheduler 实验

顶层 `experiments/`、`results/`、`figures/` 与 `docs/` 保留 Exp08 / Exp12 / Exp11 的
Scheduler 与 Prefix Cache 实验；这些数据与 INT8 LM Head 的结论相互独立。

---

## 许可

本仓库未添加统一的 `LICENSE` 文件。没有 `LICENSE` **不构成 GitHub 公开仓库的技术阻塞**；
默认版权仍然适用，此时不授予复用许可。作者如未来希望授予复用权，可另行选择
MIT、Apache-2.0 或其它许可证。当前状态与第三方组件说明见 [`LICENSE_NOTE.md`](LICENSE_NOTE.md)。
