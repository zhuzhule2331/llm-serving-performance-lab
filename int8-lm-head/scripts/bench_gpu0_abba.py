#!/usr/bin/env python3
"""Single-GPU hot-state vLLM ABBA stage benchmark, no third-party Python deps.

Stages: A1/BF16 -> B1/INT8 -> B2/INT8 -> A2/BF16.
GPU0 service is always 127.0.0.1:8002; restart it externally when switching.
This is a client-side streaming benchmark, not a CUDA event profiler.
"""
import argparse
import csv
import datetime as dt
import json
import random
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

STAGES = {"A1": "BF16", "B1": "INT8", "B2": "INT8", "A2": "BF16"}
MODELS = {"BF16": "qwen-baseline", "INT8": "qwen-int8"}
PROMPTS = [
    "详细说明CUDA Warp与Block的区别，以及GPU访存优化原则。",
    "请分步骤解释vLLM连续批处理、KV Cache与Decode阶段性能瓶颈。",
    "比较GPU上矩阵向量乘法和矩阵矩阵乘法在低Batch推理场景的性能差异。",
]
TOKENS = 128
ROUNDS = 12
URL = "http://127.0.0.1:8002"
TEMP_LIMIT = 80.0


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def gpu_temps():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )
    result = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            result[int(parts[0])] = float(parts[1])
    if not {0, 1}.issubset(result):
        raise RuntimeError("Cannot read both GPU temperatures")
    if max(result.values()) >= TEMP_LIMIT:
        raise RuntimeError(f"Temperature safeguard: {result}, limit={TEMP_LIMIT} C")
    return result


def stream_request(model, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": TOKENS,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    start_at = now()
    first = last = None
    n_chunks = 0
    usage_tokens = None
    finish_reason = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            line = raw[6:].strip()
            if line == b"[DONE]":
                break
            event = json.loads(line)
            usage = event.get("usage")
            if usage:
                usage_tokens = usage.get("completion_tokens")
            for choice in event.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                frag = (choice.get("delta") or {}).get("content")
                if frag:
                    instant = time.perf_counter()
                    n_chunks += 1
                    first = instant if first is None else first
                    last = instant
    end = time.perf_counter()
    if usage_tokens != TOKENS or first is None or last is None:
        raise RuntimeError(f"Unexpected response: tokens={usage_tokens}, chunks={n_chunks}")
    return {
        "request_started": start_at,
        "request_ended": now(),
        "ttft_ms": round((first - start) * 1000, 3),
        "tpot_ms": round((last - first) * 1000 / (usage_tokens - 1), 3),
        "total_ms": round((end - start) * 1000, 3),
        "completion_tokens": usage_tokens,
        "content_events": n_chunks,
        "finish_reason": finish_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES))
    args = parser.parse_args()
    tag = STAGES[args.stage]
    model = MODELS[tag]
    with urllib.request.urlopen(URL + "/v1/models", timeout=15) as f:
        model_ids = {v["id"] for v in json.load(f)["data"]}
    if model not in model_ids:
        raise SystemExit(f"Wrong model on port 8002: expected {model}, got {model_ids}")

    out = Path("artifacts/abba_gpu0") / (args.stage + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    meta = {
        "stage": args.stage, "kind": tag, "model": model, "gpu": 0,
        "port": 8002, "tokens": TOKENS, "rounds": ROUNDS,
        "prompts": PROMPTS, "temp_limit_c": TEMP_LIMIT,
        "started": now(), "gpu_initial_temps": gpu_temps(),
        "client_metric_warning": "TPOT=(last nonempty SSE content-first nonempty SSE content)/(128-1); not GPU kernel timing",
    }
    (out / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STAGE:", args.stage, tag, "GPU0, port8002")
    print("OUTPUT:", out)
    print("GPU initial temp:", meta["gpu_initial_temps"])
    print("WARMUP: 2 requests per prompt", flush=True)
    for i in range(6):
        stream_request(model, PROMPTS[i % len(PROMPTS)])
    print("WARMUP COMPLETE", flush=True)

    f_telemetry = (out / "nvidia_smi.csv").open("w", encoding="utf-8")
    f_stderr = (out / "nvidia_smi.stderr.log").open("w", encoding="utf-8")
    monitor = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,"
         "clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
         "--format=csv", "-lms", "500"],
        stdout=f_telemetry, stderr=f_stderr,
    )
    rows = []
    try:
        sequence = [p for _ in range(ROUNDS) for p in range(len(PROMPTS))]
        random.Random(26016).shuffle(sequence)
        with (out / "requests.jsonl").open("w", encoding="utf-8") as log:
            for i, prompt_idx in enumerate(sequence, 1):
                if monitor.poll() is not None:
                    raise RuntimeError("GPU telemetry stopped unexpectedly")
                if (i - 1) % 3 == 0:
                    gpu_temps()
                r = stream_request(model, PROMPTS[prompt_idx])
                r.update({"stage": args.stage, "model_kind": tag, "index": i, "prompt_id": prompt_idx})
                log.write(json.dumps(r, ensure_ascii=False) + "\n")
                log.flush()
                rows.append(r)
                if i % 6 == 0:
                    med = statistics.median(v["tpot_ms"] for v in rows)
                    print(f"{i:02d}/{len(sequence)} TPOT median={med:.3f}ms lastTTFT={r['ttft_ms']:.1f}ms", flush=True)
        gpu_temps()
        summary = {
            "stage": args.stage, "kind": tag, "requests": len(rows),
            "ttft_median_ms": statistics.median(r["ttft_ms"] for r in rows),
            "tpot_median_ms": statistics.median(r["tpot_ms"] for r in rows),
            "tpot_mean_ms": statistics.mean(r["tpot_ms"] for r in rows),
            "total_median_ms": statistics.median(r["total_ms"] for r in rows),
            "content_events_median": statistics.median(r["content_events"] for r in rows),
            "gpu_final_temps": gpu_temps(),
            "finished": now(),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nABBA STAGE COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=5)
        f_telemetry.close()
        f_stderr.close()
        print("Saved:", out, flush=True)


if __name__ == "__main__":
    main()
