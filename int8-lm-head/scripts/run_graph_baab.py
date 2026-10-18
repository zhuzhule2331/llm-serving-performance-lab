import os
import csv
import json
import time
import signal
import socket
import subprocess
import statistics
from pathlib import Path
from datetime import datetime

import requests

# Project root = parent of this scripts/ directory.
ROOT = Path(__file__).resolve().parents[1]

# Local path to the Qwen3-VL-8B-Instruct snapshot directory.
MODEL = os.environ["QWEN_MODEL_DIR"]

OUT = ROOT / "artifacts/profiling" / (
    "graph_baab_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
OUT.mkdir(parents=True, exist_ok=True)

PORT = 8002
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

STAGES = [
    ("B1_INT8", True),
    ("A1_BF16", False),
    ("A2_BF16", False),
    ("B2_INT8", True),
]

PROMPTS = [
    "详细解释 CUDA Warp、Block 和 GPU 内存优化。",
    "介绍 CUDA 全局内存合并访问和共享内存 bank conflict。",
    "解释 LLM Decode 中 KV Cache 的访存与带宽瓶颈。",
]

REQUESTS_PER_STAGE = 12
WARMUP_PER_STAGE = 3
MAX_TEMPERATURE = 79

CMD = [
    "vllm", "serve", MODEL,
    "--served-model-name", "qwen-int8",
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--tensor-parallel-size", "1",
    "--dtype", "bfloat16",
    "--max-model-len", "2048",
    "--max-num-batched-tokens", "2048",
    "--max-num-seqs", "8",
    "--gpu-memory-utilization", "0.85",
]

def port_busy():
    s = socket.socket()
    s.settimeout(1)
    try:
        return s.connect_ex(("127.0.0.1", PORT)) == 0
    finally:
        s.close()

def gpu_status():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "-i", "0",
            "--query-gpu=temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()

    parts = [x.strip() for x in raw.split(",")]
    return int(parts[0]), raw

def check_temperature():
    temp, status = gpu_status()
    if temp >= MAX_TEMPERATURE:
        raise RuntimeError(
            f"GPU 0 temperature {temp}C >= {MAX_TEMPERATURE}C"
        )
    return status

def stop_server(proc):
    if proc is None:
        return

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)

    for _ in range(30):
        if not port_busy():
            return
        time.sleep(1)

    raise RuntimeError("8002 port was not released")

def start_server(stage, int8):
    if port_busy():
        raise RuntimeError(
            "Port 8002 is occupied. Stop the existing vLLM first."
        )

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_PLUGINS": "qwen_int8_head",
        "INT8_LM_HEAD_ENABLE": "1" if int8 else "0",
        "INT8_LM_HEAD_TRACE": "0",
        "INT8_LM_HEAD_PROFILE": "0",
        "INT8_LM_HEAD_NSYS": "0",
        "HF_HUB_OFFLINE": "1",
    })

    log_path = OUT / f"{stage}.log"
    log_file = open(log_path, "w", buffering=1)

    print(f"\n========== START {stage} ==========", flush=True)

    proc = subprocess.Popen(
        CMD,
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        ready = False

        for _ in range(420):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{stage}: vLLM exited; inspect {log_path}"
                )

            try:
                r = requests.get(
                    f"http://127.0.0.1:{PORT}/health",
                    timeout=2,
                )
                if r.status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass

            time.sleep(1)

        if not ready:
            raise RuntimeError(
                f"{stage}: startup timeout; inspect {log_path}"
            )

        text = log_path.read_text(errors="replace")

        if "Graph capturing finished" not in text:
            raise RuntimeError(
                f"{stage}: CUDA Graph capture not confirmed"
            )

        loaded = "Quantized weights loaded" in text

        if int8 and not loaded:
            raise RuntimeError(
                f"{stage}: INT8 weights not confirmed"
            )

        if not int8 and loaded:
            raise RuntimeError(
                f"{stage}: unexpected INT8 weights in BF16 baseline"
            )

        print(f"{stage}: server ready, CUDA Graph confirmed")
        print("GPU:", check_temperature(), flush=True)

        return proc, log_file

    except BaseException:
        stop_server(proc)
        log_file.close()
        raise

def request_once(prompt):
    payload = {
        "model": "qwen-int8",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 128,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    first = None
    last = None
    tokens = None
    chunks = 0

    with requests.post(
        URL,
        json=payload,
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue

            raw = line[6:]
            if raw == b"[DONE]":
                break

            data = json.loads(raw)
            now = time.perf_counter()

            usage = data.get("usage")
            if usage:
                tokens = usage.get("completion_tokens")

            for choice in data.get("choices", []):
                content = choice.get("delta", {}).get("content")

                if content:
                    chunks += 1
                    if first is None:
                        first = now
                    last = now

    if first is None or last is None or tokens != 128:
        raise RuntimeError(
            f"Invalid response: tokens={tokens}, chunks={chunks}"
        )

    return {
        "tokens": tokens,
        "chunks": chunks,
        "ttft_ms": (first - t0) * 1000,
        "tpot_ms": (last - first) * 1000 / (tokens - 1),
    }

if port_busy():
    raise SystemExit(
        "ERROR: Port 8002 is busy. Press Ctrl+C in old vLLM terminal."
    )

if not Path(MODEL).exists():
    raise SystemExit("ERROR: Model snapshot not found")

print("========== GRAPH BAAB ==========")
print("Output directory:", OUT)
print("Stage order:", [x[0] for x in STAGES])
print("Requests per stage:", REQUESTS_PER_STAGE)
print("Warmup per stage:", WARMUP_PER_STAGE)
print("Temperature limit:", MAX_TEMPERATURE)
print("GPU:", check_temperature(), flush=True)

results = []

csv_path = OUT / "results.csv"

with open(csv_path, "w", newline="") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "stage", "mode", "request",
            "prompt_id", "tokens", "chunks",
            "ttft_ms", "tpot_ms", "gpu_status"
        ],
    )
    writer.writeheader()

    for stage, int8 in STAGES:
        proc = None
        log_file = None

        try:
            proc, log_file = start_server(stage, int8)

            print(f"\n========== WARMUP {stage} ==========")

            for i in range(WARMUP_PER_STAGE):
                check_temperature()
                request_once(PROMPTS[i % len(PROMPTS)])

            print("Warmup complete", flush=True)

            stage_values = []

            for i in range(REQUESTS_PER_STAGE):
                check_temperature()

                prompt_id = i % len(PROMPTS)
                sample = request_once(PROMPTS[prompt_id])

                status = check_temperature()
                value = sample["tpot_ms"]
                stage_values.append(value)

                row = {
                    "stage": stage,
                    "mode": "INT8" if int8 else "BF16",
                    "request": i + 1,
                    "prompt_id": prompt_id,
                    "tokens": sample["tokens"],
                    "chunks": sample["chunks"],
                    "ttft_ms": round(sample["ttft_ms"], 4),
                    "tpot_ms": round(value, 4),
                    "gpu_status": status,
                }

                writer.writerow(row)
                csv_file.flush()
                results.append(row)

                print(
                    f"{stage} [{i+1:02d}/{REQUESTS_PER_STAGE}] "
                    f"TPOT={value:.3f}ms "
                    f"chunks={sample['chunks']} "
                    f"GPU={status}",
                    flush=True,
                )

            print(
                f"\n{stage} "
                f"median={statistics.median(stage_values):.4f}ms "
                f"mean={statistics.mean(stage_values):.4f}ms",
                flush=True,
            )

        finally:
            stop_server(proc)
            if log_file:
                log_file.close()

if len(results) != len(STAGES) * REQUESTS_PER_STAGE:
    raise RuntimeError("Incomplete benchmark")

print("\n========== FINAL ABBA SUMMARY ==========")

summary = {}

for stage, int8 in STAGES:
    vals = [
        r["tpot_ms"] for r in results
        if r["stage"] == stage
    ]

    summary[stage] = {
        "median_ms": statistics.median(vals),
        "mean_ms": statistics.mean(vals),
        "count": len(vals),
    }

    print(
        f"{stage}: "
        f"median={summary[stage]['median_ms']:.4f}ms "
        f"mean={summary[stage]['mean_ms']:.4f}ms"
    )

bf16 = [
    r["tpot_ms"] for r in results
    if r["mode"] == "BF16"
]
int8 = [
    r["tpot_ms"] for r in results
    if r["mode"] == "INT8"
]

bf16_mean = statistics.mean(bf16)
int8_mean = statistics.mean(int8)

diff = int8_mean - bf16_mean
speedup = bf16_mean / int8_mean

print("\n========== BF16 VS INT8 ==========")
print(f"BF16 mean TPOT: {bf16_mean:.4f} ms")
print(f"INT8 mean TPOT: {int8_mean:.4f} ms")
print(f"INT8 - BF16:    {diff:+.4f} ms")
print(f"Speed ratio:    {speedup:.4f}x")
print(
    f"TPOT change:    "
    f"{100 * (int8_mean / bf16_mean - 1):+.3f}%"
)

print("\nNOTE: Descriptive comparison only.")
print("No statistical significance claim is made.")
print("Streaming chunk timing is an approximation of TPOT.")

report = {
    "stages": summary,
    "bf16_mean_ms": bf16_mean,
    "int8_mean_ms": int8_mean,
    "int8_minus_bf16_ms": diff,
    "speed_ratio": speedup,
    "results_csv": str(csv_path),
}

(OUT / "summary.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
)

print("\n========== FILES ==========")
print("Results:", csv_path)
print("Summary:", OUT / "summary.json")
print("Server logs:", OUT)
print("\nGRAPH BAAB COMPLETE")
