import os
import sys
from pathlib import Path

ARTIFACT = (
    Path(__file__).resolve().parent
    / "artifacts/lm_head_int8_rowwise.safetensors"
)

def register():
    if os.environ.get("INT8_LM_HEAD_ENABLE") != "1":
        return

    import torch
    from safetensors.torch import load_file
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    build_dir = Path(__file__).resolve().parent / "build"
    sys.path.insert(0, str(build_dir))
    import int8_gemv_sm86_ext

    original = LogitsProcessor._apply_head

    if getattr(original, "_int8_experimental", False):
        return

    trace_enabled = os.environ.get("INT8_LM_HEAD_TRACE") == "1"
    trace_stats = {"custom": 0, "fallback": 0}
    profile_enabled = os.environ.get("INT8_LM_HEAD_PROFILE") == "1"
    profile_samples = []
    nsys_enabled = os.environ.get("INT8_LM_HEAD_NSYS") == "1"
    nsys_state = {"calls": 0}

    def patched(self, lm_head, hidden_states, embedding_bias):
        eligible = (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[-1] == 4096
            and hidden_states.numel() == 4096
            and tuple(lm_head.weight.shape) == (151936, 4096)
            and lm_head.tp_size == 1
            and embedding_bias is None
            and isinstance(
                lm_head.quant_method, UnquantizedEmbeddingMethod
            )
            and (
                self.head_dtype is None
                or self.head_dtype == hidden_states.dtype
            )
        )

        if not eligible:
            if trace_enabled:
                trace_stats["fallback"] += 1
                if trace_stats["fallback"] <= 3:
                    print(
                        "[INT8 TRACE] fallback "
                        f"shape={tuple(hidden_states.shape)} "
                        f"dtype={hidden_states.dtype}",
                        flush=True,
                    )
            return original(
                self, lm_head, hidden_states, embedding_bias
            )

        cache = getattr(lm_head, "_experimental_int8_cache", None)

        if cache is None:
            tensors = load_file(str(ARTIFACT), device="cpu")
            Wq = tensors["weight_q"].to(hidden_states.device)
            scales = tensors["scales"].to(hidden_states.device)

            assert Wq.shape == lm_head.weight.shape
            assert scales.shape == (151936,)

            cache = (Wq.contiguous(), scales.contiguous())
            lm_head._experimental_int8_cache = cache

            print(
                "[INT8 LM HEAD] Quantized weights loaded; "
                "custom CUDA V2 enabled",
                flush=True,
            )

        Wq, scales = cache

        if nsys_enabled:
            nsys_state["calls"] += 1
            n = nsys_state["calls"]

            if n == 8:
                torch.cuda.cudart().cudaProfilerStart()
                print("[INT8 NSYS] CAPTURE START", flush=True)

        if profile_enabled:
            ev = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(4)
            ]
            ev[0].record()

        x = hidden_states.reshape(-1).float().contiguous()

        if profile_enabled:
            ev[1].record()

        y = int8_gemv_sm86_ext.forward(x, Wq, scales)

        if profile_enabled:
            ev[2].record()

        if trace_enabled:
            trace_stats["custom"] += 1
            count = trace_stats["custom"]
            if count == 1 or count % 32 == 0:
                print(
                    f"[INT8 TRACE] custom_forward={count} "
                    f"fallback={trace_stats['fallback']} "
                    f"shape={tuple(hidden_states.shape)}",
                    flush=True,
                )

        out = y.reshape(
            *hidden_states.shape[:-1], 151936
        ).to(hidden_states.dtype)

        if profile_enabled:
            ev[3].record()
            profile_samples.append(ev)

            if len(profile_samples) >= 64:
                from statistics import median

                # 诊断模式：每 64 次同步一次
                ev[3].synchronize()

                med = [
                    median(
                        events[i].elapsed_time(events[i + 1])
                        for events in profile_samples
                    )
                    for i in range(3)
                ]

                print(
                    "[INT8 EVENT PROFILE] "
                    f"n={len(profile_samples)} "
                    f"input_fp32_ms={med[0]:.4f} "
                    f"gemv_interval_ms={med[1]:.4f} "
                    f"output_bf16_ms={med[2]:.4f}",
                    flush=True,
                )

                profile_samples.clear()

        if nsys_enabled and nsys_state["calls"] == 71:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print("[INT8 NSYS] CAPTURE STOP", flush=True)

        return out

    patched._int8_experimental = True
    LogitsProcessor._apply_head = patched

    print("[INT8 LM HEAD] Plugin registered", flush=True)
