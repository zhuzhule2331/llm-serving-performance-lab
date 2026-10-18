import os
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from safetensors.torch import save_file

ROOT = Path(os.environ["QWEN_MODEL_DIR"])

DEST = Path(__file__).resolve().parents[1] / "artifacts"
DEST.mkdir(exist_ok=True)
OUT = DEST / "qwen_decode_hidden_hf.safetensors"

if OUT.exists():
    raise SystemExit(f"输出文件已存在，未覆盖：{OUT}")

free, total = torch.cuda.mem_get_info(0)
print(f"GPU 0 free: {free / 1024**3:.2f} GiB", flush=True)

if free < 20 * 1024**3:
    raise RuntimeError("GPU 0 空闲显存不足 20 GiB，停止加载模型")

print("Loading local Qwen3-VL model...", flush=True)

processor = AutoProcessor.from_pretrained(
    str(ROOT), local_files_only=True
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    str(ROOT),
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).eval()

# 找到 Transformers 模型中的 LM Head
heads = [
    (name, module)
    for name, module in model.named_modules()
    if name.endswith("lm_head")
]

if len(heads) != 1:
    raise RuntimeError(f"LM Head 数量异常：{[x[0] for x in heads]}")

name, head = heads[0]
print("Hook target:", name, flush=True)

captured = []
calls = 0

def capture_hook(module, inputs):
    global calls
    calls += 1

    hidden = inputs[0]

    # 第一次调用通常是 Prefill；
    # 后续增量 forward 才属于 Decode。
    if calls > 1:
        vector = hidden.detach().reshape(
            -1, hidden.shape[-1]
        )[-1]

        if vector.numel() != 4096:
            raise RuntimeError("Unexpected hidden size")

        captured.append(
            vector.to("cpu", dtype=torch.bfloat16).clone()
        )

handle = head.register_forward_pre_hook(capture_hook)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请解释 GPU Warp 和 Block 的区别，并举两个简单例子。"
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to("cuda:0")

print("Generating...", flush=True)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=12,
        min_new_tokens=12,
        do_sample=False,
        use_cache=True,
    )

handle.remove()

new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]

print(
    "Generated:",
    processor.tokenizer.decode(
        new_tokens, skip_special_tokens=True
    )[:250],
    flush=True
)

print("LM Head forward calls:", calls, flush=True)
print("Captured decode vectors:", len(captured), flush=True)

if not captured:
    raise RuntimeError("没有捕获到 Decode Hidden States")

hidden_states = torch.stack(captured).contiguous()

print("Shape:", tuple(hidden_states.shape), flush=True)
print("Dtype:", hidden_states.dtype, flush=True)

save_file(
    {"hidden_states": hidden_states},
    str(OUT),
    metadata={
        "model": "Qwen3-VL-8B-Instruct",
        "source": "Transformers LM Head pre-hook",
        "stage": "incremental decode",
    },
)

print("CAPTURE PASS", flush=True)
print("Saved:", OUT, flush=True)
