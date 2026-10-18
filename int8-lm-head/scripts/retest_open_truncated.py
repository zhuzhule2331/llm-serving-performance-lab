import json
import time
import hashlib
import urllib.request
from pathlib import Path

BASE = Path("artifacts/quality_ab")
SOURCE = BASE / "full_results.json"
FULL_OUT = BASE / "retest_open_long_full.json"
BLIND_OUT = BASE / "retest_open_long_blind.json"

SERVERS = {
    "BF16": (8001, "qwen-baseline"),
    "INT8": (8002, "qwen-int8"),
}

MAX_TOKENS = 1200

original = json.loads(
    SOURCE.read_text(encoding="utf-8")
)

targets = [
    r for r in original
    if r["kind"] == "open"
    and any(
        r["answers"][tag]["finish_reason"] == "length"
        for tag in SERVERS
    )
]

print("========== RETEST TARGETS ==========")
print("Total:", len(targets))
print("IDs:", [r["id"] for r in targets])
print("Max tokens:", MAX_TOKENS)

# 支持中断后继续，不覆盖已完成的请求
if FULL_OUT.exists():
    saved = json.loads(
        FULL_OUT.read_text(encoding="utf-8")
    )
    records = {r["id"]: r for r in saved}
else:
    records = {}

def request_model(tag, prompt):
    port, model = SERVERS[tag]

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            payload, ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        req, timeout=300
    ) as response:
        data = json.load(response)

    choice = data["choices"][0]

    return {
        "text": choice["message"].get("content") or "",
        "finish_reason": choice["finish_reason"],
        "output_tokens": data["usage"]["completion_tokens"],
        "elapsed_s": round(
            time.perf_counter() - start, 3
        ),
    }

for index, task in enumerate(targets):
    tid = task["id"]

    if tid not in records:
        records[tid] = {
            "id": tid,
            "category": task["category"],
            "question": task["question"],
            "criteria": task["criteria"],
            "actual_prompt": task["actual_prompt"],
            "original_finish": {
                tag: task["answers"][tag]["finish_reason"]
                for tag in SERVERS
            },
            "answers": {},
        }

    record = records[tid]

    order = (
        ["BF16", "INT8"]
        if index % 2 == 0
        else ["INT8", "BF16"]
    )

    for tag in order:
        if tag in record["answers"]:
            print(f"SKIP {tid} {tag}", flush=True)
            continue

        print(
            f"RUN {tid} {tag}...",
            flush=True
        )

        result = request_model(
            tag, task["actual_prompt"]
        )

        record["answers"][tag] = result

        # 每次请求完成后立即保存
        tmp = FULL_OUT.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                list(records.values()),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(FULL_OUT)

        print(
            f"  finish={result['finish_reason']} "
            f"tokens={result['output_tokens']} "
            f"time={result['elapsed_s']}s",
            flush=True,
        )

print("\n========== RETEST SUMMARY ==========")

for task in targets:
    tid = task["id"]
    record = records[tid]

    b = record["answers"]["BF16"]
    q = record["answers"]["INT8"]

    print(
        f"{tid}: "
        f"BF16={b['finish_reason']}"
        f"({b['output_tokens']} tokens) "
        f"INT8={q['finish_reason']}"
        f"({q['output_tokens']} tokens)"
    )

# 按原来 X/Y 的映射生成盲评文件
# 不在盲评文件内暴露版本
blind = []

for task in targets:
    record = records[task["id"]]

    flip = int(
        hashlib.sha256(
            task["id"].encode()
        ).hexdigest(), 16
    ) % 2

    tags = ["BF16", "INT8"]

    if flip:
        tags.reverse()

    blind.append({
        "id": task["id"],
        "category": task["category"],
        "question": task["question"],
        "criteria": task["criteria"],
        "X": record["answers"][tags[0]]["text"],
        "Y": record["answers"][tags[1]]["text"],
        "X_finish_reason":
            record["answers"][tags[0]]["finish_reason"],
        "Y_finish_reason":
            record["answers"][tags[1]]["finish_reason"],
        "score_X_0_to_4": None,
        "score_Y_0_to_4": None,
        "notes": "",
    })

BLIND_OUT.write_text(
    json.dumps(
        blind, ensure_ascii=False, indent=2
    ),
    encoding="utf-8",
)

print("\nSaved full:", FULL_OUT)
print("Saved blind:", BLIND_OUT)
print("OPEN RETEST COMPLETE")
