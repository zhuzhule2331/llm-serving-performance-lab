import sqlite3
from collections import defaultdict

db = sqlite3.connect(
    "artifacts/profiling/qwen_int8_decode.sqlite"
)

kernels = db.execute("""
SELECT k.start, k.end, k.deviceId,
       COALESCE(s.value, '')
FROM CUPTI_ACTIVITY_KIND_KERNEL k
LEFT JOIN StringIds s ON s.id = k.demangledName
ORDER BY k.start
""").fetchall()

gemvs = [
    r for r in kernels
    if "int8_gemv_v2" in r[3]
]

device = gemvs[0][2]

def category(name):
    if "int8_gemv_v2" in name:
        return "INT8_GEMV"
    if "sliced1x2" in name:
        return "BF16_GEMM_SLICED"
    if "ampere_bf16" in name:
        return "BF16_GEMM"
    if "cutlass::Kernel2" in name:
        return "CUTLASS_GEMM"
    if "flash_fwd" in name:
        return "FLASH_ATTN"
    if "fused_add_rms_norm" in name:
        return "FUSED_RMS_NORM"
    if "rms_norm_kernel" in name:
        return "RMS_NORM"
    if "act_and_mul" in name:
        return "SILU"
    if "reshape_and_cache" in name:
        return "KV_CACHE"
    if "rope" in name.lower():
        return "ROPE"
    if "index" in name.lower():
        return "INDEX"
    return name.split("<")[0][:45] or "UNKNOWN"

events = [
    (a, b, category(n))
    for a, b, dev, n in kernels
    if dev == device
]

for table, label in [
    ("CUPTI_ACTIVITY_KIND_MEMCPY", "MEMCPY"),
    ("CUPTI_ACTIVITY_KIND_MEMSET", "MEMSET"),
]:
    exists = db.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name=?",
        (table,)
    ).fetchone()

    if exists:
        events += [
            (a, b, label)
            for a, b, dev in db.execute(
                f'SELECT start,end,deviceId FROM "{table}"'
            )
            if dev == device
        ]

events.sort()

stats = defaultdict(lambda: [0, 0.0])
examples = []

for cycle, (left, right) in enumerate(
    zip(gemvs, gemvs[1:]), 1
):
    begin, end = left[1], right[1]

    cursor = begin
    previous = "INT8_GEMV"

    for a, b, name in events:
        if a >= end:
            break
        if b <= begin:
            continue

        a = max(a, begin)
        b = min(b, end)

        if a > cursor:
            gap_ms = (a - cursor) / 1e6

            if gap_ms >= 0.1:
                key = (previous, name)
                stats[key][0] += 1
                stats[key][1] += gap_ms

                examples.append(
                    (gap_ms, cycle, previous, name)
                )

        if b > cursor:
            cursor = b
            previous = name

print("========== TOP GAP NEIGHBORS ==========")
print("Rank | Count | Total ms | Avg ms | Before -> After")

ranked = sorted(
    stats.items(),
    key=lambda item: item[1][1],
    reverse=True
)

for i, (pair, (count, total)) in enumerate(
    ranked[:20], 1
):
    print(
        f"{i:2d} | {count:5d} | "
        f"{total:8.2f} | "
        f"{total/count:.3f} | "
        f"{pair[0]} -> {pair[1]}"
    )

print("\n========== TOP LONGEST GAPS ==========")

for gap, cycle, before, after in sorted(
    examples, reverse=True
)[:15]:
    print(
        f"cycle={cycle:02d} "
        f"gap={gap:.3f}ms "
        f"{before} -> {after}"
    )

print("\nGAP NEIGHBOR ANALYSIS COMPLETE")
db.close()
