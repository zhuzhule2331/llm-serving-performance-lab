import os
import sys
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "build"))
import int8_gemv_sm86_ext

SHARD = Path(os.environ["QWEN_LM_HEAD_SHARD"])

torch.cuda.set_device(0)

with safe_open(str(SHARD), framework="pt", device="cpu") as f:
    weight = f.get_slice("lm_head.weight")
    shape = weight.get_shape()
    dtype = weight.get_dtype()

    print("========== REAL QWEN LM HEAD ==========", flush=True)
    print("Weight shape:", shape, flush=True)
    print("Weight dtype:", dtype, flush=True)

    assert len(shape) == 2
    assert shape == [151936, 4096], f"Unexpected shape: {shape}"
    assert dtype == "BF16", f"Expected BF16, got {dtype}"

    M, N = shape

    free, total = torch.cuda.mem_get_info()
    print(f"Free GPU memory: {free / 1024**3:.2f} GiB", flush=True)

    if free < 4 * 1024**3:
        raise RuntimeError("空闲显存不足 4 GiB，停止实验")

    # GPU 上保存原始 BF16 与量化 INT8 权重
    Wbf = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    Wq = torch.empty((M, N), device="cuda", dtype=torch.int8)
    scales = torch.empty(M, device="cuda", dtype=torch.float32)

    print("Quantizing real LM Head...", flush=True)

    # 每次只从 safetensors 读取 1024 行
    for start in range(0, M, 1024):
        end = min(start + 1024, M)

        part = weight[start:end, :].to(
            device="cuda", dtype=torch.bfloat16
        )

        Wbf[start:end].copy_(part)

        w32 = part.float()
        max_abs = w32.abs().amax(dim=1)

        scale = torch.where(
            max_abs > 0,
            max_abs / 127.0,
            torch.ones_like(max_abs)
        )

        scales[start:end] = scale

        Wq[start:end] = (
            (w32 / scale[:, None])
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )

        if start % (1024 * 32) == 0:
            print(f"  Quantized {end}/{M} rows", flush=True)

    torch.cuda.synchronize()

print("Quantization COMPLETE", flush=True)

def run_bf16(x_bf):
    return torch.mm(
        x_bf.reshape(1, N),
        Wbf.t(),
        out_dtype=torch.float32
    ).reshape(-1)

def run_int8(x_float):
    return int8_gemv_sm86_ext.forward(
        x_float, Wq, scales
    )

print("\n========== OUTPUT ACCURACY ==========")

# 尚未接入真实 Qwen hidden states；
# 这里使用三个不同的合成 BF16 激活向量
torch.manual_seed(42)

for trial in range(3):
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    x_bf = (x * 0.03).to(torch.bfloat16)
    x_float = x_bf.float()

    y_bf = run_bf16(x_bf)
    y_int8 = run_int8(x_float)

    torch.cuda.synchronize()

    if not torch.isfinite(y_bf).all() or not torch.isfinite(y_int8).all():
        raise RuntimeError("发现非有限输出")

    diff = y_int8 - y_bf

    nrmse = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(y_bf).clamp_min(1e-12)
    ).item() * 100

    max_err = diff.abs().max().item()

    print(
        f"Trial {trial + 1}: "
        f"NRMSE={nrmse:.6f}% "
        f"MaxError={max_err:.6f} "
        f"Top1Equal={y_bf.argmax().item() == y_int8.argmax().item()}",
        flush=True
    )

    if trial == 0:
        # 抽检自定义 GPU Kernel 和同权重 PyTorch 参考计算
        rows = torch.linspace(
            0, M - 1, steps=33, device="cuda"
        ).long()

        cpu_ref = (
            Wq[rows].cpu().float()
            @ x_float.cpu()
        ) * scales[rows].cpu()

        gpu_sample = y_int8[rows].cpu()

        torch.testing.assert_close(
            gpu_sample, cpu_ref,
            atol=1e-3, rtol=1e-4
        )

        print("INT8 GPU vs CPU reference: PASS (33 rows)")

print("\n========== GPU KERNEL BENCHMARK ==========")

# 使用最后一组输入；转换和离线量化不计入 Kernel 时间
for _ in range(10):
    run_bf16(x_bf)
    run_int8(x_float)

torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

def measure(fn):
    start_event.record()
    fn()
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event)

bf_times = []
int8_times = []

# 交错测量，减少固定顺序造成的偏差
for i in range(20):
    if i % 2 == 0:
        bf_times.append(measure(lambda: run_bf16(x_bf)))
        int8_times.append(measure(lambda: run_int8(x_float)))
    else:
        int8_times.append(measure(lambda: run_int8(x_float)))
        bf_times.append(measure(lambda: run_bf16(x_bf)))

bf_ms = statistics.median(bf_times)
int8_ms = statistics.median(int8_times)

print(f"BF16 torch.mm median: {bf_ms:.4f} ms")
print(f"INT8 V2 median:       {int8_ms:.4f} ms")
print(f"Latency ratio:        {bf_ms / int8_ms:.3f}x")
print("REAL-WEIGHT MICROBENCHMARK COMPLETE")
