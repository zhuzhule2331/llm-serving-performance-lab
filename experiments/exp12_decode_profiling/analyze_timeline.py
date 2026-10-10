# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp12_decode_profiling/exp12_timeline_analysis.py
# Copied  : verbatim except for the changes listed in ../../PUBLIC_CODE_CHANGELOG.md
# Frozen  : this file is the FINAL public-facing implementation used in the
#           formal campaign; it is a copy, the original is unmodified.
# =============================================================================
import sqlite3
import statistics
from pathlib import Path


CONDITIONS = {
    "A_C1_MBT2048":
        "exp12_A_c1_mbt2048.sqlite",

    "B_C8_MBT2048":
        "exp12_B_c8_mbt2048.sqlite",

    "C_C8_MBT4096":
        "exp12_C_c8_mbt4096.sqlite",
}


GEMV_PATTERN = "%gemv2T_kernel_val%"

GEMM64_PATTERN = (
    "%ampere_bf16_s16816gemm_bf16_"
    "64x64_ldg8_f2f_stages_64x6_tn%"
)

PREPARE_PATTERN = (
    "%_prepare_prefill_inputs_kernel%"
)


def percentile(xs, p):
    xs = sorted(xs)

    if not xs:
        return None

    if len(xs) == 1:
        return xs[0]

    pos = (
        (len(xs) - 1)
        * p
    )

    lo = int(pos)
    hi = min(
        lo + 1,
        len(xs) - 1,
    )

    frac = pos - lo

    return (
        xs[lo] * (1 - frac)
        + xs[hi] * frac
    )


def fetch_events(conn, pattern):
    q = """
    SELECT
        k.start,
        k.end,
        s.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s
      ON k.demangledName = s.id
    WHERE s.value LIKE ?
    ORDER BY k.start
    """

    return conn.execute(
        q,
        (pattern,),
    ).fetchall()


def interval_stats(starts_ns):
    if len(starts_ns) < 2:
        return None

    intervals_ms = [
        (
            starts_ns[i + 1]
            - starts_ns[i]
        ) / 1e6
        for i in range(
            len(starts_ns) - 1
        )
    ]

    return {
        "n_intervals":
            len(intervals_ms),

        "mean_ms":
            statistics.mean(
                intervals_ms
            ),

        "median_ms":
            statistics.median(
                intervals_ms
            ),

        "p10_ms":
            percentile(
                intervals_ms,
                0.10,
            ),

        "p90_ms":
            percentile(
                intervals_ms,
                0.90,
            ),

        "min_ms":
            min(intervals_ms),

        "max_ms":
            max(intervals_ms),

        "stdev_ms":
            (
                statistics.stdev(
                    intervals_ms
                )
                if len(intervals_ms) > 1
                else 0.0
            ),

        "intervals":
            intervals_ms,
    }


def print_stats(label, rows):
    starts = [
        r[0]
        for r in rows
    ]

    print(
        f"{label}: "
        f"instances={len(rows)}"
    )

    s = interval_stats(
        starts
    )

    if s is None:
        return

    print(
        "  intervals="
        f"{s['n_intervals']}"
    )

    print(
        "  mean="
        f"{s['mean_ms']:.3f} ms"
    )

    print(
        "  median="
        f"{s['median_ms']:.3f} ms"
    )

    print(
        "  p10="
        f"{s['p10_ms']:.3f} ms"
    )

    print(
        "  p90="
        f"{s['p90_ms']:.3f} ms"
    )

    print(
        "  min="
        f"{s['min_ms']:.3f} ms"
    )

    print(
        "  max="
        f"{s['max_ms']:.3f} ms"
    )

    print(
        "  stdev="
        f"{s['stdev_ms']:.3f} ms"
    )

    largest = sorted(
        enumerate(
            s["intervals"],
            start=1,
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    print(
        "  10 largest intervals:"
    )

    for idx, value in largest:
        print(
            f"    #{idx}: "
            f"{value:.3f} ms"
        )


for name, db in CONDITIONS.items():
    print()
    print("=" * 76)
    print(name)
    print("=" * 76)

    if not Path(db).exists():
        print(
            "MISSING:",
            db,
        )
        continue

    conn = sqlite3.connect(
        db
    )

    prepare = fetch_events(
        conn,
        PREPARE_PATTERN,
    )

    gemv = fetch_events(
        conn,
        GEMV_PATTERN,
    )

    gemm64 = fetch_events(
        conn,
        GEMM64_PATTERN,
    )

    print(
        "prepare_prefill instances:",
        len(prepare),
    )

    if prepare:
        first_start = prepare[0][0]

        print(
            "prepare relative starts (ms):"
        )

        for i, row in enumerate(
            prepare,
            start=1,
        ):
            print(
                f"  {i:2d}: "
                f"{(row[0]-first_start)/1e6:.3f}"
            )

        last_prepare_end = (
            prepare[-1][1]
        )

    else:
        last_prepare_end = None

    print()
    print_stats(
        "ALL gemv2T",
        gemv,
    )

    print()
    print_stats(
        "ALL 64x64 GEMM",
        gemm64,
    )

    if last_prepare_end is not None:
        gemv_after = [
            r for r in gemv
            if r[0] > last_prepare_end
        ]

        gemm64_after = [
            r for r in gemm64
            if r[0] > last_prepare_end
        ]

        print()
        print(
            "--- after final "
            "prepare_prefill ---"
        )

        print_stats(
            "gemv2T after-final",
            gemv_after,
        )

        print()

        print_stats(
            "64x64 after-final",
            gemm64_after,
        )

    conn.close()
