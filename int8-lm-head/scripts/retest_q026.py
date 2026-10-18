import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path("artifacts/quality_ab")
SOURCE = ROOT / "full_results.json"
DEST = ROOT / "q026_retest_768.json"

if DEST.exists():
    raise SystemExit(f"已有结果文件，未覆盖：{DEST}")

tasks = json.loads(SOURCE.read_text(encoding="utf-8"))
task = next(t for t in tasks if t["id"] == "Q026")
prompt = task["actual_prompt"]

servers = {
    "BF16": (8001, "qwen-baseline"),
    "INT8": (8002, "qwen-int8"),
}

results = []

def generate(tag):
    port, model = servers[tag]

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": 768,
        "stream": False,
    }

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.load(response)

    choice = data["choices"][0]
    text = choice["message"].get("content") or ""

    answers = re.findall(
        r"答案\s*[:：]\s*([^\r\n]+)",
        text,
    )

    return {
        "model": tag,
        "finish_reason": choice["finish_reason"],
        "output_tokens": data["usage"]["completion_tokens"],
        "elapsed_s": round(time.perf_counter() - start, 3),
        "extracted_answer": answers[-1] if answers else None,
        "text": text,
    }

print("Question:", task["question"], flush=True)

for round_id in range(1, 4):
    order = (
        ["BF16", "INT8"]
        if round_id % 2
        else ["INT8", "BF16"]
    )

    for tag in order:
        result = generate(tag)
        result["round"] = round_id
        results.append(result)

        DEST.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\n===== ROUND {round_id} {tag} =====", flush=True)
        print("Finish:", result["finish_reason"], flush=True)
        print("Tokens:", result["output_tokens"], flush=True)
        print("Extracted:", result["extracted_answer"], flush=True)
        print("Last 700 chars:", flush=True)
        print(result["text"][-700:], flush=True)

print("\nSAVED:", DEST)
print("Q026 RETEST COMPLETE")
