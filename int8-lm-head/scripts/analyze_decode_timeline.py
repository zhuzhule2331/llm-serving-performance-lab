import sqlite3
import statistics as st
from pathlib import Path

db_path = Path("artifacts/profiling/qwen_int8_decode.sqlite")
db = sqlite3.connect(str(db_path))

tables = {
    r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
}

kernels = db.execute("""
    SELECT k.start, k.end, k.deviceId, COALESCE(s.value, '')
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    LEFT JOIN StringIds s ON s.id = k.demangledName
    ORDER BY k.start
""").fetchall()

gemvs = [
    r for r in kernels
    if "int8_gemv_v2" in r[3]
]

print("Total kernels:", len(kernels))
print("INT8 GEMV kernels:", len(gemvs))

if len(gemvs) < 10:
    raise RuntimeError("Too few INT8 GEMV kernels")

gpu_id = gemvs[0][2]

kernels = [
    r for r in kernels if r[2] == gpu_id
]

activity = [(r[0], r[1]) for r in kernels]

for table in (
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
):
    if table not in tables:
        continue

    extra = db.execute(
        f'SELECT start, end, deviceId FROM "{table}"'
    ).fetchall()

    activity.extend(
        (a, b) for a, b, dev in extra
        if dev == gpu_id
    )

activity.sort()

def union_ms(spans, begin, end):
    busy = 0
    cursor = begin
    longest_gap = 0

    for a, b in spans:
        if a >= end:
            break
        if b <= begin:
            continue

        a = max(a, begin)
        b = min(b, end)

        if a > cursor:
            longest_gap = max(longest_gap, a - cursor)

        if b > cursor:
            busy += b - max(a, cursor)
            cursor = b

    longest_gap = max(longest_gap, end - cursor)

    return busy / 1e6, longest_gap / 1e6

windows = []

for left, right in zip(gemvs, gemvs[1:]):
    begin = left[1]
    end = right[1]

    if end <= begin:
        continue

    wall_ms = (end - begin) / 1e6

    kernel_busy, _ = union_ms(
        [(r[0], r[1]) for r in kernels],
        begin, end
    )

    active_ms, longest_gap = union_ms(
        activity, begin, end
    )

    kernel_count = sum(
        begin <= r[0] < end
        for r in kernels
    )

    windows.append({
        "wall_ms": wall_ms,
        "kernel_busy_ms": kernel_busy,
        "gpu_activity_ms": active_ms,
        "no_recorded_activity_ms": max(
            0, wall_ms - active_ms
        ),
        "longest_gap_ms": longest_gap,
        "kernel_count": kernel_count,
    })

print("\n========== DECODE TIMELINE ==========")
print("GPU deviceId:", gpu_id)
print("Decode intervals:", len(windows))

for key in [
    "wall_ms",
    "kernel_busy_ms",
    "gpu_activity_ms",
    "no_recorded_activity_ms",
    "longest_gap_ms",
    "kernel_count",
]:
    values = [w[key] for w in windows]

    print(
        f"{key}: "
        f"median={st.median(values):.3f} "
        f"mean={st.mean(values):.3f} "
        f"min={min(values):.3f} "
        f"max={max(values):.3f}"
    )

total_wall = sum(w["wall_ms"] for w in windows)
total_active = sum(w["gpu_activity_ms"] for w in windows)

print(
    "\nRecorded GPU activity share:",
    f"{100 * total_active / total_wall:.2f}%"
)

print(
    "No recorded activity share:",
    f"{100 * (total_wall-total_active)/total_wall:.2f}%"
)

db.close()
print("\nDECODE TIMELINE ANALYSIS COMPLETE")
