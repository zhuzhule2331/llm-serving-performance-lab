import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Path to the safetensors shard that contains lm_head.weight.
# Set QWEN_LM_HEAD_SHARD when running; no host path is baked into this repo.
SOURCE = Path(os.environ["QWEN_LM_HEAD_SHARD"])

OUT = Path(__file__).resolve().parents[1] / "artifacts"
OUT.mkdir(exist_ok=True)
DEST = OUT / "lm_head_int8_rowwise.safetensors"

if DEST.exists():
    raise SystemExit(f"文件已存在，未覆盖：{DEST}")

if shutil.disk_usage(OUT).free < 1024**3:
    raise SystemExit("磁盘剩余空间不足 1 GiB")

with safe_open(str(SOURCE), framework="pt", device="cpu") as f:
    weight = f.get_slice("lm_head.weight")
    M, N = weight.get_shape()

    assert (M, N) == (151936, 4096)
    assert weight.get_dtype() == "BF16"

    Wq = torch.empty((M, N), dtype=torch.int8)
    scales = torch.empty(M, dtype=torch.float32)

    print(f"Quantizing: {M} x {N}", flush=True)

    for start in range(0, M, 1024):
        end = min(start + 1024, M)

        w = weight[start:end, :].to("cuda").float()
        max_abs = w.abs().amax(dim=1)

        s = torch.where(
            max_abs > 0,
            max_abs / 127.0,
            torch.ones_like(max_abs)
        )

        q = (
            (w / s[:, None])
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )

        Wq[start:end].copy_(q.cpu())
        scales[start:end].copy_(s.cpu())

        if start % 32768 == 0:
            print(f"  {end}/{M} rows", flush=True)

assert torch.isfinite(scales).all()
assert (scales > 0).all()

tmp = DEST.with_name(DEST.name + ".tmp")

save_file(
    {"weight_q": Wq, "scales": scales},
    str(tmp),
    metadata={
        "source": "Qwen3-VL-8B-Instruct/lm_head.weight",
        "scheme": "symmetric_int8_per_output_row",
        "source_dtype": "BF16",
    },
)

tmp.replace(DEST)

print("EXPORT PASS")
print("File:", DEST)
print("Size MiB:", round(DEST.stat().st_size / 1024**2, 2))
print("Wq:", tuple(Wq.shape), Wq.dtype)
print("Scales:", tuple(scales.shape), scales.dtype)
