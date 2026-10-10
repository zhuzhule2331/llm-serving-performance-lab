# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp11_prefix_mbt/exp11_analyze_formal.py
# Copied  : verbatim except for the changes listed in ../../PUBLIC_CODE_CHANGELOG.md
# Frozen  : this file is the FINAL public-facing implementation used in the
#           formal campaign; it is a copy, the original is unmodified.
# =============================================================================
#!/usr/bin/env python3

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Public-repo portability shim (added during public-repo preparation).
# The original frozen scripts hard-coded absolute paths on the experiment host.
# Those literals are replaced by environment variables below.
# NO measurement or scheduling logic is altered.
# ---------------------------------------------------------------------------
import os as _os

VLLM_BIN = _os.environ.get("VLLM_BIN", "vllm")
MODEL_PATH_STR = _os.environ.get("MODEL_PATH", "models/Qwen3-VL-8B-Instruct")
VLLM_HOME = _os.environ.get("VLLM_HOME", "./")
# ---------------------------------------------------------------------------



ROOT = Path(_os.environ.get("RESULTS_DIR", "results"))

BURSTS = ROOT / "formal_bursts.csv"
HASHES = ROOT / "artifact_hashes.json"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def mean(xs):
    return statistics.mean(xs)


def pct_change(a, b):
    return (
        (b / a) - 1.0
    ) * 100.0


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"empty output: {path}"
        )

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with path.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------
# 1. Verify frozen final evidence
# ------------------------------------------------------------

expected = json.loads(
    HASHES.read_text()
)

for name, digest in expected.items():

    actual = sha256_file(
        ROOT / name
    )

    if actual != digest:
        raise RuntimeError(
            f"HASH MISMATCH: {name}"
        )

print("FINAL ARTIFACT HASHES: PASS")


# ------------------------------------------------------------
# 2. Load
# ------------------------------------------------------------

with BURSTS.open(
    newline="",
) as f:
    rows = list(
        csv.DictReader(f)
    )

if len(rows) != 54:
    raise RuntimeError(
        f"expected 54 bursts, got {len(rows)}"
    )


# Exact columns used by the formal runner.
METRICS = {
    "ttft_ms":
        "mean_ttft_ms",

    "tpot_ms":
        "mean_tpot_ms",

    "out_tok_s":
        "output_throughput_tok_s",

    "raw_hit_ratio":
        "raw_group_hit_ratio",

    "compute_per_request":
        "compute_per_request",
}


# Optional metrics: include if present.
headers = set(
    rows[0].keys()
)

optional_candidates = {
    "makespan_s": [
        "makespan_s",
        "makespan_sec",
        "makespan",
    ],

    "req_s": [
        "request_throughput_req_s",
        "request_throughput_rps",
        "request_throughput",
    ],
}

for short, candidates in (
    optional_candidates.items()
):
    for c in candidates:
        if c in headers:
            METRICS[short] = c
            break


for r in rows:
    r["_block"] = int(
        r["block_id"]
    )
    r["_mbt"] = int(
        r["mbt"]
    )
    r["_reuse"] = (
        r["reuse_condition"]
    )

    for short, col in (
        METRICS.items()
    ):
        r[
            "_" + short
        ] = float(
            r[col]
        )


# ------------------------------------------------------------
# 3. Structural balance
# ------------------------------------------------------------

cell_groups = defaultdict(list)
block_cell_groups = defaultdict(list)

for r in rows:

    cell_groups[
        (
            r["_reuse"],
            r["_mbt"],
        )
    ].append(r)

    block_cell_groups[
        (
            r["_block"],
            r["_reuse"],
            r["_mbt"],
        )
    ].append(r)


expected_cells = {
    (reuse, mbt)
    for reuse in [
        "P0",
        "P50",
        "PHIGH",
    ]
    for mbt in [
        512,
        2048,
        4096,
    ]
}

if set(
    cell_groups.keys()
) != expected_cells:
    raise RuntimeError(
        "3x3 cells incomplete"
    )

for key, xs in cell_groups.items():
    if len(xs) != 6:
        raise RuntimeError(
            f"{key} does not have 6 bursts"
        )

for key, xs in (
    block_cell_groups.items()
):
    if len(xs) != 2:
        raise RuntimeError(
            f"{key} does not have 2 replicas"
        )

print("3x3 BALANCE: PASS")
print("BLOCK-CELL REPLICATION: PASS")


# ------------------------------------------------------------
# 4. Block-cell means
# ------------------------------------------------------------

block_cells = {}

block_cell_rows = []

for key, xs in sorted(
    block_cell_groups.items()
):

    block, reuse, mbt = key

    out = {
        "block":
            block,
        "reuse":
            reuse,
        "mbt":
            mbt,
        "n":
            len(xs),
    }

    for metric in METRICS:
        value = mean([
            x["_" + metric]
            for x in xs
        ])

        out[metric] = value

    block_cells[key] = out
    block_cell_rows.append(out)


# ------------------------------------------------------------
# 5. 3x3 descriptive cell means
# ------------------------------------------------------------

cell_rows = []

cell_means = {}

for reuse in [
    "P0",
    "P50",
    "PHIGH",
]:
    for mbt in [
        512,
        2048,
        4096,
    ]:

        xs = cell_groups[
            (reuse, mbt)
        ]

        out = {
            "reuse":
                reuse,
            "mbt":
                mbt,
            "n_bursts":
                len(xs),
            "n_blocks":
                3,
        }

        for metric in METRICS:

            vals = [
                x["_" + metric]
                for x in xs
            ]

            block_vals = [
                block_cells[
                    (
                        b,
                        reuse,
                        mbt,
                    )
                ][metric]
                for b in [
                    1,
                    2,
                    3,
                ]
            ]

            out[metric] = mean(
                vals
            )

            out[
                metric
                + "_block_min"
            ] = min(
                block_vals
            )

            out[
                metric
                + "_block_max"
            ] = max(
                block_vals
            )

        cell_means[
            (reuse, mbt)
        ] = out

        cell_rows.append(out)


# ------------------------------------------------------------
# 6. MBT contrasts within each reuse condition
#
# Important:
# pooled_pct = ratio of overall cell means
# block_mean_pct = mean of 3 within-block ratios
#
# They are NOT mathematically identical estimands.
# ------------------------------------------------------------

mbt_contrast_rows = []

comparisons = [
    (512, 2048),
    (512, 4096),
    (2048, 4096),
]

contrast_metrics = [
    x
    for x in [
        "ttft_ms",
        "tpot_ms",
        "makespan_s",
        "req_s",
        "out_tok_s",
    ]
    if x in METRICS
]

for reuse in [
    "P0",
    "P50",
    "PHIGH",
]:

    for mbt_a, mbt_b in comparisons:

        for metric in (
            contrast_metrics
        ):

            a = cell_means[
                (reuse, mbt_a)
            ][metric]

            b = cell_means[
                (reuse, mbt_b)
            ][metric]

            block_pcts = []

            block_deltas = []

            for block in [
                1,
                2,
                3,
            ]:

                ba = block_cells[
                    (
                        block,
                        reuse,
                        mbt_a,
                    )
                ][metric]

                bb = block_cells[
                    (
                        block,
                        reuse,
                        mbt_b,
                    )
                ][metric]

                block_deltas.append(
                    bb - ba
                )

                block_pcts.append(
                    pct_change(
                        ba,
                        bb,
                    )
                )

            mbt_contrast_rows.append({
                "reuse":
                    reuse,

                "metric":
                    metric,

                "from_mbt":
                    mbt_a,

                "to_mbt":
                    mbt_b,

                "from_cell_mean":
                    a,

                "to_cell_mean":
                    b,

                "pooled_pct_change":
                    pct_change(
                        a,
                        b,
                    ),

                "mean_block_delta":
                    mean(
                        block_deltas
                    ),

                "mean_block_pct_change":
                    mean(
                        block_pcts
                    ),

                "min_block_pct_change":
                    min(
                        block_pcts
                    ),

                "max_block_pct_change":
                    max(
                        block_pcts
                    ),

                "all_blocks_same_sign":
                    (
                        all(
                            x > 0
                            for x
                            in block_pcts
                        )
                        or
                        all(
                            x < 0
                            for x
                            in block_pcts
                        )
                    ),
            })


# ------------------------------------------------------------
# 7. Prefix-reuse contrasts within each MBT
# ------------------------------------------------------------

reuse_contrast_rows = []

reuse_comparisons = [
    ("P0", "P50"),
    ("P0", "PHIGH"),
    ("P50", "PHIGH"),
]

for mbt in [
    512,
    2048,
    4096,
]:

    for reuse_a, reuse_b in (
        reuse_comparisons
    ):

        for metric in (
            contrast_metrics
        ):

            a = cell_means[
                (
                    reuse_a,
                    mbt,
                )
            ][metric]

            b = cell_means[
                (
                    reuse_b,
                    mbt,
                )
            ][metric]

            block_pcts = []

            for block in [
                1,
                2,
                3,
            ]:

                ba = block_cells[
                    (
                        block,
                        reuse_a,
                        mbt,
                    )
                ][metric]

                bb = block_cells[
                    (
                        block,
                        reuse_b,
                        mbt,
                    )
                ][metric]

                block_pcts.append(
                    pct_change(
                        ba,
                        bb,
                    )
                )

            reuse_contrast_rows.append({
                "mbt":
                    mbt,

                "metric":
                    metric,

                "from_reuse":
                    reuse_a,

                "to_reuse":
                    reuse_b,

                "from_cell_mean":
                    a,

                "to_cell_mean":
                    b,

                "pooled_pct_change":
                    pct_change(
                        a,
                        b,
                    ),

                "mean_block_pct_change":
                    mean(
                        block_pcts
                    ),

                "min_block_pct_change":
                    min(
                        block_pcts
                    ),

                "max_block_pct_change":
                    max(
                        block_pcts
                    ),

                "all_blocks_same_sign":
                    (
                        all(
                            x > 0
                            for x
                            in block_pcts
                        )
                        or
                        all(
                            x < 0
                            for x
                            in block_pcts
                        )
                    ),
            })


# ------------------------------------------------------------
# 8. Emit
# ------------------------------------------------------------

write_csv(
    ROOT / "block_cell_summary.csv",
    block_cell_rows,
)

write_csv(
    ROOT / "cell_summary.csv",
    cell_rows,
)

write_csv(
    ROOT
    / "mbt_within_reuse_contrasts.csv",
    mbt_contrast_rows,
)

write_csv(
    ROOT
    / "reuse_within_mbt_contrasts.csv",
    reuse_contrast_rows,
)


# ------------------------------------------------------------
# 9. Human-readable core table
# ------------------------------------------------------------

print()
print("=" * 92)
print("EXP11 3x3 FORMAL CELL MEANS")
print("=" * 92)

print(
    f"{'reuse':<7}"
    f"{'MBT':>6}"
    f"{'TTFT(ms)':>13}"
    f"{'TPOT(ms)':>13}"
    f"{'out tok/s':>13}"
    f"{'hit%':>10}"
    f"{'compute':>11}"
)

for reuse in [
    "P0",
    "P50",
    "PHIGH",
]:

    for mbt in [
        512,
        2048,
        4096,
    ]:

        x = cell_means[
            (reuse, mbt)
        ]

        print(
            f"{reuse:<7}"
            f"{mbt:>6}"
            f"{x['ttft_ms']:>13.3f}"
            f"{x['tpot_ms']:>13.3f}"
            f"{x['out_tok_s']:>13.3f}"
            f"{x['raw_hit_ratio']*100:>10.2f}"
            f"{x['compute_per_request']:>11.1f}"
        )


print()
print("=" * 92)
print(
    "KEY INTERACTION: MBT4096 vs MBT512"
)
print(
    "negative latency % = lower/better; "
    "positive throughput % = higher/better"
)
print("=" * 92)

for reuse in [
    "P0",
    "P50",
    "PHIGH",
]:

    print()
    print(reuse)

    for metric in [
        "ttft_ms",
        "tpot_ms",
        "out_tok_s",
    ]:

        r = next(
            x
            for x in mbt_contrast_rows
            if (
                x["reuse"] == reuse
                and
                x["metric"] == metric
                and
                x["from_mbt"] == 512
                and
                x["to_mbt"] == 4096
            )
        )

        print(
            f"  {metric:<10} "
            f"pooled="
            f"{r['pooled_pct_change']:+7.2f}% "
            f"block-mean="
            f"{r['mean_block_pct_change']:+7.2f}% "
            f"range=["
            f"{r['min_block_pct_change']:+.2f}, "
            f"{r['max_block_pct_change']:+.2f}] "
            f"same_sign="
            f"{r['all_blocks_same_sign']}"
        )


print()
print("=" * 92)
print("PREFIX EFFECT: PHIGH vs P0")
print("=" * 92)

for mbt in [
    512,
    2048,
    4096,
]:

    print()
    print(
        f"MBT={mbt}"
    )

    for metric in [
        "ttft_ms",
        "tpot_ms",
        "out_tok_s",
    ]:

        r = next(
            x
            for x
            in reuse_contrast_rows
            if (
                x["mbt"] == mbt
                and
                x["metric"]
                == metric
                and
                x["from_reuse"]
                == "P0"
                and
                x["to_reuse"]
                == "PHIGH"
            )
        )

        print(
            f"  {metric:<10} "
            f"pooled="
            f"{r['pooled_pct_change']:+7.2f}% "
            f"block-mean="
            f"{r['mean_block_pct_change']:+7.2f}% "
            f"same_sign="
            f"{r['all_blocks_same_sign']}"
        )


print()
print(
    "FORMAL ANALYSIS EXTRACTION: PASS"
)
