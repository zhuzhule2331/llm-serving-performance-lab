import sqlite3
import statistics as st
from pathlib import Path

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

assert len(gemvs) >= 10, "Too few GEMV records"

device = gemvs[0][2]

events = [
    (a, b) for a, b, dev, _ in kernels
    if dev == device
]

for table in (
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
):
    exists = db.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name=?",
        (table,)
    ).fetchone()

    if exists:
        rows = db.execute(
            f'SELECT start,end,deviceId FROM "{table}"'
        )
        events += [
            (a, b) for a, b, dev in rows
            if dev == device
        ]

events.sort()

gaps = []
per_cycle = []

for i, (left, right) in enumerate(
    zip(gemvs, gemvs[1:]), 1
):
    begin, end = left[1], right[1]
    cursor = begin
    cycle_gaps = []

    for a, b in events:
        if a >= end:
            break
        if b <= begin:
            continue

        a = max(a, begin)
        b = min(b, end)

        if a > cursor:
            gap_ms = (a - cursor) / 1e6
            gaps.append((gap_ms, i))
            cycle_gaps.append(gap_ms)

        cursor = max(cursor, b)

    if cursor < end:
        gap_ms = (end - cursor) / 1e6
        gaps.append((gap_ms, i))
        cycle_gaps.append(gap_ms)

    per_cycle.append(sum(cycle_gaps))

values = sorted(g for g, _ in gaps)
total_idle = sum(values)

def percentile(p):
    idx = round((len(values) - 1) * p / 100)
    return values[idx]

print("========== GPU GAP ANALYSIS ==========")
print("Decode cycles:", len(per_cycle))
print("Number of gaps:", len(gaps))
print("Total gap time:", round(total_idle, 2), "ms")
print("Median gap:", round(percentile(50)*1000, 2), "us")
print("P90 gap:", round(percentile(90)*1000, 2), "us")
print("P99 gap:", round(percentile(99)*1000, 2), "us")

print("\n========== GAP DISTRIBUTION ==========")

bins = [
    (0, 0.01),
    (0.01, 0.05),
    (0.05, 0.1),
    (0.1, 0.5),
    (0.5, 1.0),
    (1.0, float("inf")),
]

for lo, hi in bins:
    subset = [
        v for v in values
        if lo <= v < hi
    ]
    duration = sum(subset)

    print(
        f"{lo:.2f} - {hi:.2f} ms: "
        f"count={len(subset):6d}, "
        f"total={duration:9.2f} ms, "
        f"idle_share={100*duration/total_idle:5.1f}%"
    )

print("\n========== TOP 10 LONGEST GAPS ==========")

for gap, cycle in sorted(gaps, reverse=True)[:10]:
    print(f"cycle={cycle:02d} gap={gap:.4f} ms")

print("\nGPU GAP ANALYSIS COMPLETE")
db.close()
