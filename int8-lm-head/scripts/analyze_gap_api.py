import sqlite3
from collections import defaultdict

db = sqlite3.connect(
    "artifacts/profiling/qwen_int8_decode.sqlite"
)
db.row_factory = sqlite3.Row

tables = {
    r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
}

def columns(table):
    return {
        r[1] for r in db.execute(
            f'PRAGMA table_info("{table}")'
        )
    }

# GPU Kernel + Memcpy + Memset
events = []

for table, kind in [
    ("CUPTI_ACTIVITY_KIND_KERNEL", "KERNEL"),
    ("CUPTI_ACTIVITY_KIND_MEMCPY", "MEMCPY"),
    ("CUPTI_ACTIVITY_KIND_MEMSET", "MEMSET"),
]:
    if table not in tables:
        continue

    cols = columns(table)
    corr = "correlationId" if "correlationId" in cols else "NULL"

    for r in db.execute(
        f'SELECT start,end,deviceId,{corr} AS cid '
        f'FROM "{table}"'
    ):
        if r["deviceId"] == 0:
            events.append((
                r["start"], r["end"], kind, r["cid"]
            ))

events.sort()

gemvs = list(db.execute("""
SELECT k.start,k.end
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id=k.demangledName
WHERE k.deviceId=0 AND s.value LIKE '%int8_gemv_v2%'
ORDER BY k.start
"""))

# CPU CUDA Runtime / Driver API
apis = defaultdict(list)

for table in [
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_DRIVER",
]:
    if table not in tables:
        continue

    if not {"start", "end", "correlationId", "nameId"} <= columns(table):
        continue

    for r in db.execute(f"""
        SELECT a.start,a.end,a.correlationId,
               COALESCE(s.value,'UNKNOWN') AS name
        FROM "{table}" a
        LEFT JOIN StringIds s ON s.id=a.nameId
    """):
        apis[r["correlationId"]].append((
            r["start"], r["end"], r["name"]
        ))

# 提取相邻 GEMV 之间 >=0.5ms 的空隙
gaps = []

for cycle, (left, right) in enumerate(
    zip(gemvs, gemvs[1:]), 1
):
    begin, end = left["end"], right["end"]
    cursor = begin

    for a, b, kind, cid in events:
        if a >= end:
            break
        if b <= begin:
            continue

        a = max(a, begin)
        b = min(b, end)

        if a > cursor and (a - cursor) >= 500_000:
            gaps.append((cycle, cursor, a, kind, cid))

        cursor = max(cursor, b)

stats = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
matched = 0

for cycle, begin, end, kind, cid in gaps:
    candidates = [
        a for a in apis.get(cid, [])
        if a[0] <= end
    ]

    if not candidates:
        continue

    # 选择与下一次 GPU 活动最接近的 API
    api_start, api_end, name = min(
        candidates,
        key=lambda x: abs(end - x[1])
    )

    before = max(
        0, min(end, api_start) - begin
    ) / 1e6

    during = max(
        0,
        min(end, api_end) - max(begin, api_start)
    ) / 1e6

    after = max(
        0, end - max(begin, api_end)
    ) / 1e6

    duration = (end - begin) / 1e6
    matched += 1

    key = f"{kind} / {name}"
    row = stats[key]

    row[0] += 1
    row[1] += duration
    row[2] += before
    row[3] += during
    row[4] += after

print("========== GPU GAP / CUDA API ==========")
print("GEMV kernels:", len(gemvs))
print("Long gaps >=0.5ms:", len(gaps))
print("Gaps matched to CUDA API:", matched)

if matched:
    totals = [
        sum(v[i] for v in stats.values())
        for i in range(1, 5)
    ]

    print("\nMatched long gap time: %.2f ms" % totals[0])
    print("Before API start:       %.2f ms" % totals[1])
    print("During API call:        %.2f ms" % totals[2])
    print("After API return:       %.2f ms" % totals[3])

print("\n========== TOP API GROUPS ==========")
print("Count | Gap ms | Before | During | After | API")

for name, values in sorted(
    stats.items(),
    key=lambda item: item[1][1],
    reverse=True
)[:15]:
    n, total, before, during, after = values
    print(
        f"{n:5d} | {total:7.1f} | "
        f"{before:6.1f} | {during:6.1f} | "
        f"{after:6.1f} | {name}"
    )

db.close()
print("\nCUDA API CORRELATION COMPLETE")
