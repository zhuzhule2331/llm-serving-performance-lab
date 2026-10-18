import sys
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "build")
)

import int8_gemv_sm86_ext

print("Extension loaded:", int8_gemv_sm86_ext.__file__)
print("GPU:", torch.cuda.get_device_name(0))

def test(M, N):
    # INT8 权重
    Wq = (
        torch.arange(M * N, device="cuda", dtype=torch.int32)
        .remainder(27)
        .sub(13)
        .to(torch.int8)
        .reshape(M, N)
    )

    # FP32 输入
    x = (
        torch.arange(N, device="cuda", dtype=torch.float32)
        .remainder(11)
        .sub(5)
        .div(8)
    )

    scales = torch.linspace(
        0.01, 0.05, M, device="cuda", dtype=torch.float32
    )

    # 你的自定义 CUDA Kernel
    y_gpu = int8_gemv_sm86_ext.forward(x, Wq, scales)

    # PyTorch 参考计算
    y_ref = torch.sum(
        Wq.float() * x.unsqueeze(0), dim=1
    ) * scales

    torch.cuda.synchronize()

    error = (y_gpu - y_ref).abs().max().item()

    torch.testing.assert_close(
        y_gpu, y_ref,
        atol=1e-4,
        rtol=1e-5
    )

    print(
        f"M={M}, N={N} | "
        f"Max Error={error:.8f} | PASS"
    )

test(5, 37)
test(8, 4096)

print("PYTORCH CUDA EXTENSION: PASS")
