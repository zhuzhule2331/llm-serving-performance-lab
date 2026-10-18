import json
import time
import statistics
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8002/v1/chat/completions"

PAYLOAD = {
    "model": "qwen-int8",
    "messages": [{
        "role": "user",
        "content": "详细说明CUDA Warp与Block的区别，以及GPU访存优化原则。"
    }],
    "temperature": 0,
    "max_tokens": 96,
    "ignore_eos": True,
    "stream": True,
    "stream_options": {"include_usage": True},
}

def request():
    req = urllib.request.Request(
        URL,
        data=json.dumps(PAYLOAD, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )

    first = None
    last = None
    tokens = None
    start = time.perf_counter()

    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue

            body = raw[6:].strip()

            if body == b"[DONE]":
                break

            event = json.loads(body)

            if event.get("usage"):
                tokens = event["usage"]["completion_tokens"]

            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")

                if content:
                    now = time.perf_counter()

                    if first is None:
                        first = now

                    last = now

    end = time.perf_counter()

    if first is None or tokens != 96:
        raise RuntimeError(
            f"Invalid response: first={first}, tokens={tokens}"
        )

    return {
        "ttft_ms": round((first - start) * 1000, 2),
        "tpot_ms": round(
            (last - first) * 1000 / (tokens - 1), 2
        ),
        "total_ms": round((end - start) * 1000, 2),
    }

print("========== WARMUP ==========", flush=True)
print(request(), flush=True)

results = []

for i in range(6):
    print(f"\n========== ROUND {i+1} ==========", flush=True)

    time.sleep(8)

    cold = request()
    print("COLD:", cold, flush=True)

    warm = request()
    print("WARM:", warm, flush=True)

    results.append({
        "round": i + 1,
        "cold": cold,
        "warm": warm,
    })

cold_ttft = [r["cold"]["ttft_ms"] for r in results]
warm_ttft = [r["warm"]["ttft_ms"] for r in results]

print("\n========== SUMMARY ==========")

print("Cold TTFT median:", statistics.median(cold_ttft), "ms")
print("Warm TTFT median:", statistics.median(warm_ttft), "ms")

print(
    "Cold/Warm ratio:",
    round(
        statistics.median(cold_ttft) /
        statistics.median(warm_ttft), 3
    ),
    "x"
)

out = Path("artifacts/telemetry/cold_warm_ttft.json")
out.parent.mkdir(exist_ok=True)

out.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Saved:", out)
print("COLD WARM TEST COMPLETE")
