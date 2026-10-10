# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp11_prefix_mbt/exp11_recover_block2.py
# Copied  : verbatim except for the changes listed in ../../PUBLIC_CODE_CHANGELOG.md
# Frozen  : this file is the FINAL public-facing implementation used in the
#           formal campaign; it is a copy, the original is unmodified.
# =============================================================================

# --- public-repo module resolution (added during public-repo preparation) ---
# The frozen scripts assumed every module lived flat in one directory.
# The public layout splits shared modules into common/, so add both roots.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (
    _os.path.abspath(_os.path.join(_HERE, "..", "..")),   # repo root  -> common.*
    _os.path.abspath(_os.path.join(_HERE, "..")),         # experiments/ -> sibling pkg
    _HERE,                                                # same dir
    _os.path.abspath(_os.path.join(_HERE, "..", "..", "common")),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

#!/usr/bin/env python3

import argparse
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

import formal_runner as runner
import formal_execute as formal

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



ROOT = Path(__file__).resolve().parent

POLICY = (
    ROOT
    / "exp11_block2_recovery_policy_v1.json"
)

POLICY_HASH = (
    ROOT
    / "exp11_block2_recovery_policy_v1_SHA256.txt"
)

EXPECTED_TOTAL_BURSTS = 27
EXPECTED_MEASURED_BURSTS = 18
EXPECTED_TOTAL_REQUESTS = 216
EXPECTED_MEASURED_REQUESTS = 144


def truth(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() == "true"


async def main(args):

    plan = (
        runner.load_and_validate_plan()
    )

    formal.validate_namespace_map()

    block2_rows = [
        r for r in plan
        if r["block_id"] == 2
    ]

    if len(block2_rows) != 27:
        raise RuntimeError(
            "Expected exactly 27 frozen "
            f"Block-2 rows, got {len(block2_rows)}"
        )

    ids = [
        r["formal_row_id"]
        for r in block2_rows
    ]

    if ids != list(range(28, 55)):
        raise RuntimeError(
            f"Unexpected Block-2 row IDs: {ids}"
        )

    print("PLAN HASH/STRUCTURE: PASS")
    print("NAMESPACE MAP HASH: PASS")
    print("BLOCK-2 ROW RANGE: PASS (28..54)")
    print(
        "RECOVERY POLICY:",
        POLICY.name,
    )

    print()
    print(
        "Frozen replacement order:"
    )

    for position in [1, 2, 3]:

        rows = [
            r for r in block2_rows
            if (
                r["whole_plot_position"]
                == position
            )
        ]

        mbt = rows[0]["mbt"]

        reuse_order = []

        for r in rows:
            reuse = r[
                "reuse_condition"
            ]

            if (
                not reuse_order
                or reuse_order[-1] != reuse
            ):
                reuse_order.append(
                    reuse
                )

        print(
            f"whole-plot {position}: "
            f"MBT={mbt} | "
            + " -> ".join(reuse_order)
        )

    if args.dry_run:

        print()
        print(
            "BLOCK-2 REPLACEMENT DRY RUN: PASS"
        )
        print(
            "No server was started."
        )

        return

    if not args.execute:

        raise RuntimeError(
            "Use --execute only after "
            "dry-run validation."
        )

    timestamp = (
        datetime.now()
        .astimezone()
        .strftime("%Y%m%d_%H%M%S")
    )

    out_dir = (
        ROOT
        / (
            "exp11_block2_replacement_"
            + timestamp
        )
    )

    out_dir.mkdir()

    shutil.copy2(
        POLICY,
        out_dir
        / POLICY.name,
    )

    shutil.copy2(
        POLICY_HASH,
        out_dir
        / POLICY_HASH.name,
    )

    shutil.copy2(
        runner.PLAN_PATH,
        out_dir
        / "exp11_preregistered_plan.csv",
    )

    shutil.copy2(
        runner.PLAN_HASH_PATH,
        out_dir
        / "exp11_preregistered_plan_SHA256.txt",
    )

    shutil.copy2(
        formal.NAMESPACE_MAP,
        out_dir
        / "exp11_formal_namespace_map.csv",
    )

    manifest = {
        "experiment":
            "Exp11 Block-2 Replacement",

        "created_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "logical_block":
            2,

        "row_range":
            [28, 54],

        "reason":
            (
                "Original Block 2 invalidated "
                "by formal_row_id=48 "
                "launch_skew hard-validity failure."
            ),

        "protocol_changes":
            "NONE",

        "fixed_rendered_lengths":
            formal.FIXED_RENDERED_LENGTHS,

        "replacement_rule":
            (
                "Whole affected block rerun once; "
                "no individual-row retry."
            ),
    }

    (
        out_dir
        / "replacement_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    # Reuse the already validated formal executor,
    # but execute only logical Block 2.
    original_blocks = list(
        formal.BLOCKS
    )

    formal.BLOCKS = [2]

    try:

        (
            bursts,
            requests,
            block_status,
        ) = await formal.execute_formal(
            out_dir,
            plan,
        )

    finally:

        formal.BLOCKS = (
            original_blocks
        )

    measured_bursts = [
        r for r in bursts
        if truth(
            r["include_in_formal"]
        )
    ]

    measured_requests = [
        r for r in requests
        if truth(
            r["include_in_formal"]
        )
    ]

    print()
    print("=" * 88)
    print(
        "BLOCK-2 REPLACEMENT COLLECTION COMPLETE"
    )
    print("=" * 88)

    print(
        "executed bursts:",
        len(bursts),
    )

    print(
        "measured bursts:",
        len(measured_bursts),
    )

    print(
        "executed requests:",
        len(requests),
    )

    print(
        "measured requests:",
        len(measured_requests),
    )

    print(
        "block status:",
        block_status,
    )

    replacement_valid = (
        len(block_status) == 1
        and
        block_status[0]["block_id"] == 2
        and
        truth(
            block_status[0]["valid"]
        )
    )

    if not replacement_valid:

        print()
        print(
            "BLOCK-2 REPLACEMENT STATUS: INVALID"
        )

        raise RuntimeError(
            "Replacement block failed "
            "hard validity."
        )

    if len(bursts) != EXPECTED_TOTAL_BURSTS:
        raise RuntimeError(
            "Expected 27 executions, "
            f"got {len(bursts)}"
        )

    if (
        len(measured_bursts)
        != EXPECTED_MEASURED_BURSTS
    ):
        raise RuntimeError(
            "Expected 18 measured bursts, "
            f"got {len(measured_bursts)}"
        )

    if (
        len(requests)
        != EXPECTED_TOTAL_REQUESTS
    ):
        raise RuntimeError(
            "Expected 216 requests, "
            f"got {len(requests)}"
        )

    if (
        len(measured_requests)
        != EXPECTED_MEASURED_REQUESTS
    ):
        raise RuntimeError(
            "Expected 144 measured requests, "
            f"got {len(measured_requests)}"
        )

    if not all(
        truth(
            r["measurement_valid"]
        )
        for r in bursts
    ):
        raise RuntimeError(
            "At least one execution "
            "measurement_valid != True"
        )

    if not all(
        float(
            r["preemption_delta"]
        ) == 0.0
        for r in bursts
    ):
        raise RuntimeError(
            "At least one replacement "
            "execution has preemption != 0"
        )

    runner.write_csv(
        out_dir
        / "replacement_all_executions.csv",
        bursts,
    )

    runner.write_csv(
        out_dir
        / "replacement_formal_bursts.csv",
        measured_bursts,
    )

    runner.write_csv(
        out_dir
        / "replacement_formal_requests.csv",
        measured_requests,
    )

    hashes = {}

    for name in [
        "replacement_all_executions.csv",
        "replacement_formal_bursts.csv",
        "replacement_formal_requests.csv",
        "block_status.csv",
        "replacement_manifest.json",
    ]:

        hashes[name] = (
            formal.sha256_file(
                out_dir / name
            )
        )

    (
        out_dir
        / "replacement_artifact_hashes.json"
    ).write_text(
        json.dumps(
            hashes,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "BLOCK-2 REPLACEMENT STATUS: PASS"
    )

    print(
        "All 27 executions valid."
    )

    print(
        "Measured bursts: 18"
    )

    print(
        "Measured requests: 144"
    )

    print(
        "RESULT_DIR:",
        out_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    group = (
        parser
        .add_mutually_exclusive_group()
    )

    group.add_argument(
        "--dry-run",
        action="store_true",
    )

    group.add_argument(
        "--execute",
        action="store_true",
    )

    asyncio.run(
        main(
            parser.parse_args()
        )
    )
