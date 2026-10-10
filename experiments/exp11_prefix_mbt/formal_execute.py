# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp11_prefix_mbt/exp11_formal_execute.py
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
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from openai import (
    AsyncOpenAI,
    DefaultAioHttpClient,
)

import formal_runner as runner
from common import server_lifecycle as infra
import prefix_core as core

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

NAMESPACE_MAP = (
    ROOT
    / "configs"
    / "exp11_formal_namespace_map.csv"
)

EXPECTED_NAMESPACE_MAP_SHA256 = (
    "a0c97a991652089e9ee4a68bd529fb705da7565c59f1fd78d3ecbdbb19f71502"
)

FIXED_RENDERED_LENGTHS = {
    "P0": 0,
    "P50": 995,
    "PHIGH": 1953,
}

BLOCKS = [1, 2, 3]


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def validate_namespace_map():

    actual = sha256_file(
        NAMESPACE_MAP
    )

    if (
        actual
        != EXPECTED_NAMESPACE_MAP_SHA256
    ):
        raise RuntimeError(
            "NAMESPACE MAP HASH MISMATCH: "
            f"expected="
            f"{EXPECTED_NAMESPACE_MAP_SHA256}, "
            f"actual={actual}"
        )

    return actual


def augment_rows(
    plan_row,
    burst_row,
    request_rows,
):

    provenance = {
        "formal_row_id":
            plan_row["formal_row_id"],

        "whole_plot_position":
            plan_row[
                "whole_plot_position"
            ],

        "subplot_position":
            plan_row[
                "subplot_position"
            ],

        "include_in_formal":
            plan_row[
                "include_in_formal"
            ],
    }

    burst_row.update(
        provenance
    )

    for row in request_rows:
        row.update(
            provenance
        )


def write_progress(
    out_dir,
    burst_rows,
    request_rows,
    block_status,
):

    runner.write_csv(
        out_dir
        / "formal_bursts_progress.csv",
        burst_rows,
    )

    runner.write_csv(
        out_dir
        / "formal_requests_progress.csv",
        request_rows,
    )

    runner.write_csv(
        out_dir
        / "block_status.csv",
        block_status,
    )


def validate_whole_plot(
    rows,
    block,
    position,
):

    if len(rows) != 9:
        raise RuntimeError(
            "Whole plot must contain exactly "
            f"9 executions, got {len(rows)}: "
            f"block={block}, "
            f"position={position}"
        )

    mbts = {
        r["mbt"]
        for r in rows
    }

    if len(mbts) != 1:
        raise RuntimeError(
            "Whole plot contains multiple MBTs: "
            f"{mbts}"
        )

    reuse_counts = {}

    for row in rows:
        reuse_counts.setdefault(
            row["reuse_condition"],
            0,
        )

        reuse_counts[
            row["reuse_condition"]
        ] += 1

    if reuse_counts != {
        "P0": 3,
        "P50": 3,
        "PHIGH": 3,
    }:
        raise RuntimeError(
            "Invalid reuse composition in "
            f"whole plot: {reuse_counts}"
        )

    return next(
        iter(mbts)
    )


async def execute_formal(
    out_dir,
    plan,
):

    all_bursts = []
    all_requests = []
    block_status = []

    infra.kill_existing_vllm()

    try:

        for block in BLOCKS:

            print()
            print("=" * 88)
            print(
                f"FORMAL BLOCK {block}"
            )
            print("=" * 88)

            block_dir = (
                out_dir
                / f"block_{block}"
            )

            block_dir.mkdir()

            block_valid = True
            failure_reason = ""

            block_rows = [
                r for r in plan
                if r["block_id"] == block
            ]

            block_rows.sort(
                key=lambda x:
                    x["formal_row_id"]
            )

            try:

                positions = sorted({
                    r[
                        "whole_plot_position"
                    ]
                    for r in block_rows
                })

                if positions != [1, 2, 3]:
                    raise RuntimeError(
                        "Expected whole-plot "
                        f"positions [1,2,3], "
                        f"got {positions}"
                    )

                for position in positions:

                    wp_rows = [
                        r
                        for r in block_rows
                        if (
                            r[
                                "whole_plot_position"
                            ]
                            == position
                        )
                    ]

                    wp_rows.sort(
                        key=lambda x:
                            x["formal_row_id"]
                    )

                    mbt = validate_whole_plot(
                        wp_rows,
                        block,
                        position,
                    )

                    print()
                    print("#" * 88)
                    print(
                        f"BLOCK={block} "
                        f"WHOLE_PLOT={position} "
                        f"MBT={mbt}"
                    )
                    print("#" * 88)

                    server_dir = (
                        block_dir
                        / (
                            f"whole_plot_"
                            f"{position}_"
                            f"mbt_{mbt}"
                        )
                    )

                    server_dir.mkdir()

                    proc = None
                    log_handle = None

                    try:

                        # ------------------------------------
                        # Contemporaneous hardware boundary
                        # ------------------------------------

                        power_before = (
                            runner.query_power()
                        )

                        (
                            server_dir
                            / "power_boundary_before.json"
                        ).write_text(
                            json.dumps(
                                power_before,
                                indent=2,
                            )
                            + "\n"
                        )

                        (
                            proc,
                            log_handle,
                            cmd,
                        ) = runner.start_server(
                            mbt,
                            server_dir,
                        )

                        (
                            server_dir
                            / "server_command.txt"
                        ).write_text(
                            " ".join(cmd)
                            + "\n"
                        )

                        infra.wait_for_server(
                            proc
                        )

                        cmdline = (
                            runner
                            .verify_server_cmdline(
                                mbt
                            )
                        )

                        (
                            server_dir
                            / "server_cmdline.txt"
                        ).write_text(
                            cmdline + "\n"
                        )

                        power_ready = (
                            runner.query_power()
                        )

                        (
                            server_dir
                            / "power_boundary_ready.json"
                        ).write_text(
                            json.dumps(
                                power_ready,
                                indent=2,
                            )
                            + "\n"
                        )

                        preempt_start = (
                            infra.query_preemption()
                        )

                        (
                            server_dir
                            / "preemption_start.txt"
                        ).write_text(
                            f"{preempt_start}\n"
                        )

                        if preempt_start != 0:
                            raise RuntimeError(
                                "Fresh server preemption "
                                "counter not zero: "
                                f"{preempt_start}"
                            )

                        async with AsyncOpenAI(
                            base_url=
                                core.API_BASE,
                            api_key=
                                core.API_KEY,
                            http_client=
                                DefaultAioHttpClient(),
                        ) as client:

                            model_name = (
                                await
                                core.exp03.exp02
                                .get_model_name(
                                    client
                                )
                            )

                            for row in wp_rows:

                                formal_row_id = (
                                    row[
                                        "formal_row_id"
                                    ]
                                )

                                reuse = (
                                    row[
                                        "reuse_condition"
                                    ]
                                )

                                phase = (
                                    row["phase"]
                                )

                                rep = (
                                    row["rep"]
                                )

                                print()
                                print(
                                    f"ROW="
                                    f"{formal_row_id:02d} "
                                    f"B{block} "
                                    f"MBT={mbt} "
                                    f"{reuse:<5} "
                                    f"{phase:<8} "
                                    f"R{rep}"
                                )

                                workload = (
                                    await
                                    core
                                    .build_formal_workload(
                                        client,
                                        model_name,
                                        block,
                                        mbt,
                                        reuse,
                                        phase,
                                        rep,
                                        FIXED_RENDERED_LENGTHS,
                                    )
                                )

                                # Cached conditions must map
                                # to the exact preregistered row.
                                if reuse != "P0":

                                    if (
                                        int(
                                            workload[
                                                "formal_row_id"
                                            ]
                                        )
                                        != formal_row_id
                                    ):
                                        raise RuntimeError(
                                            "Frozen workload "
                                            "formal_row_id "
                                            "mismatch: "
                                            f"plan="
                                            f"{formal_row_id}, "
                                            f"workload="
                                            f"{workload['formal_row_id']}"
                                        )

                                (
                                    burst_row,
                                    request_rows,
                                ) = await (
                                    core
                                    .run_formal_condition(
                                        client,
                                        model_name,
                                        block,
                                        formal_row_id,
                                        mbt,
                                        reuse,
                                        phase,
                                        rep,
                                        workload,
                                        FIXED_RENDERED_LENGTHS,
                                    )
                                )

                                augment_rows(
                                    row,
                                    burst_row,
                                    request_rows,
                                )

                                all_bursts.append(
                                    burst_row
                                )

                                all_requests.extend(
                                    request_rows
                                )

                                write_progress(
                                    out_dir,
                                    all_bursts,
                                    all_requests,
                                    block_status,
                                )

                                print(
                                    f"  hit="
                                    f"{burst_row['raw_group_hit_ratio'] * 100:6.2f}% "
                                    f"compute="
                                    f"{burst_row['compute_per_request']:7.1f} "
                                    f"TTFT="
                                    f"{burst_row['mean_ttft_ms']:8.2f} "
                                    f"TPOT="
                                    f"{burst_row['mean_tpot_ms']:7.3f} "
                                    f"out="
                                    f"{burst_row['output_throughput_tok_s']:7.2f} "
                                    f"skew="
                                    f"{burst_row['launch_skew_ms']:6.2f} "
                                    f"preempt="
                                    f"{burst_row['preemption_delta']:.0f} "
                                    f"valid="
                                    f"{burst_row['measurement_valid']}"
                                )

                                # Fail closed for warmup and
                                # measured execution alike.
                                if not burst_row[
                                    "measurement_valid"
                                ]:

                                    raise RuntimeError(
                                        "FORMAL VALIDITY "
                                        "FAILED: "
                                        f"formal_row_id="
                                        f"{formal_row_id}, "
                                        f"block={block}, "
                                        f"MBT={mbt}, "
                                        f"reuse={reuse}, "
                                        f"phase={phase}, "
                                        f"rep={rep}"
                                    )

                        preempt_end = (
                            infra.query_preemption()
                        )

                        (
                            server_dir
                            / "preemption_end.txt"
                        ).write_text(
                            f"{preempt_end}\n"
                        )

                        if preempt_end != 0:
                            raise RuntimeError(
                                "Whole-plot server "
                                "accumulated "
                                "preemptions: "
                                f"{preempt_end}"
                            )

                    finally:

                        if (
                            proc is not None
                            and
                            log_handle is not None
                        ):
                            runner.stop_server(
                                proc,
                                log_handle,
                            )

                block_status.append(
                    {
                        "block_id":
                            block,

                        "valid":
                            True,

                        "failure_reason":
                            "",
                    }
                )

                print()
                print(
                    f"BLOCK {block}: PASS"
                )

            except Exception as exc:

                block_valid = False
                failure_reason = repr(
                    exc
                )

                block_status.append(
                    {
                        "block_id":
                            block,

                        "valid":
                            False,

                        "failure_reason":
                            failure_reason,
                    }
                )

                print()
                print(
                    f"BLOCK {block}: INVALID"
                )

                print(
                    failure_reason
                )

            write_progress(
                out_dir,
                all_bursts,
                all_requests,
                block_status,
            )

    finally:

        infra.kill_existing_vllm()

    return (
        all_bursts,
        all_requests,
        block_status,
    )


async def main(args):

    plan = (
        runner.load_and_validate_plan()
    )

    namespace_hash = (
        validate_namespace_map()
    )

    print(
        "PLAN HASH/STRUCTURE: PASS"
    )

    print(
        "NAMESPACE MAP HASH: PASS"
    )

    runner.print_plan(
        plan
    )

    if args.dry_run:

        print()
        print(
            "FORMAL EXECUTION DRY RUN: PASS"
        )

        print(
            "No server was started."
        )

        return

    if not args.execute:

        raise RuntimeError(
            "Refusing to execute formal run "
            "without --execute"
        )

    timestamp = (
        datetime.now()
        .astimezone()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    out_dir = (
        ROOT
        / (
            f"exp11_formal_"
            f"{timestamp}"
        )
    )

    out_dir.mkdir()

    # --------------------------------------------
    # Preserve preregistration artifacts
    # --------------------------------------------

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
        NAMESPACE_MAP,
        out_dir
        / "exp11_formal_namespace_map.csv",
    )

    manifest = {
        "experiment":
            "Exp11 Prefix Reuse x Scheduler Token Budget",

        "created_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "plan_sha256":
            runner.EXPECTED_PLAN_SHA256,

        "namespace_map_sha256":
            namespace_hash,

        "factors": {
            "mbt":
                [512, 2048, 4096],

            "reuse":
                [
                    "P0",
                    "P50",
                    "PHIGH",
                ],
        },

        "fixed_config": {
            "concurrency":
                8,

            "prompt_tokens":
                2048,

            "completion_tokens":
                128,

            "max_num_seqs":
                16,

            "max_model_len":
                8192,

            "gpu_memory_utilization":
                0.90,

            "chunked_prefill":
                True,

            "prefix_caching":
                True,

            "power_target_w":
                220.0,

            "fixed_rendered_lengths":
                FIXED_RENDERED_LENGTHS,
        },

        "design": {
            "blocks":
                3,

            "whole_plot_factor":
                "mbt",

            "subplot_factor":
                "reuse",

            "warmups":
                27,

            "measured_bursts":
                54,

            "measured_requests":
                432,
        },

        "validity": {
            "max_launch_skew_ms":
                25.0,

            "preemption_hard_validity":
                True,

            "runtime_namespace_search":
                False,

            "block_failure_rule":
                (
                    "Any validity failure "
                    "invalidates the affected block. "
                    "No partial block may enter "
                    "formal aggregation."
                ),
        },
    }

    (
        out_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "RESULT_DIR:",
        out_dir,
    )

    (
        burst_rows,
        request_rows,
        block_status,
    ) = await execute_formal(
        out_dir,
        plan,
    )

    all_blocks_valid = (
        len(block_status) == 3
        and
        all(
            r["valid"]
            for r in block_status
        )
    )

    measured_bursts = [
        r
        for r in burst_rows
        if r[
            "include_in_formal"
        ]
    ]

    measured_requests = [
        r
        for r in request_rows
        if r[
            "include_in_formal"
        ]
    ]

    print()
    print("=" * 88)
    print("EXP11 FORMAL COLLECTION COMPLETE")
    print("=" * 88)

    print(
        "executed bursts:",
        len(burst_rows),
    )

    print(
        "measured bursts:",
        len(measured_bursts),
    )

    print(
        "executed requests:",
        len(request_rows),
    )

    print(
        "measured requests:",
        len(measured_requests),
    )

    print(
        "block status:",
        block_status,
    )

    if not all_blocks_valid:

        print()
        print(
            "FORMAL CAMPAIGN STATUS: INVALID"
        )

        print(
            "No final formal dataset was emitted. "
            "Use progress files only for diagnosis."
        )

        raise RuntimeError(
            "At least one formal block "
            "was invalid."
        )

    # --------------------------------------------
    # Full expected-count validation
    # --------------------------------------------

    if len(burst_rows) != 81:
        raise RuntimeError(
            "Expected 81 total bursts, "
            f"got {len(burst_rows)}"
        )

    if len(measured_bursts) != 54:
        raise RuntimeError(
            "Expected 54 measured bursts, "
            f"got {len(measured_bursts)}"
        )

    if len(request_rows) != 648:
        raise RuntimeError(
            "Expected 648 total requests, "
            f"got {len(request_rows)}"
        )

    if len(measured_requests) != 432:
        raise RuntimeError(
            "Expected 432 measured requests, "
            f"got {len(measured_requests)}"
        )

    # Only after all three blocks pass do we create
    # the final formal evidence files.
    runner.write_csv(
        out_dir
        / "formal_bursts.csv",
        measured_bursts,
    )

    runner.write_csv(
        out_dir
        / "formal_requests.csv",
        measured_requests,
    )

    runner.write_csv(
        out_dir
        / "all_executions.csv",
        burst_rows,
    )

    formal_hashes = {}

    for name in [
        "formal_bursts.csv",
        "formal_requests.csv",
        "all_executions.csv",
        "block_status.csv",
    ]:

        formal_hashes[
            name
        ] = sha256_file(
            out_dir / name
        )

    (
        out_dir
        / "formal_artifact_hashes.json"
    ).write_text(
        json.dumps(
            formal_hashes,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "FORMAL CAMPAIGN STATUS: PASS"
    )

    print(
        "All 3 blocks valid."
    )

    print(
        "Final measured bursts: 54"
    )

    print(
        "Final measured requests: 432"
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

    args = parser.parse_args()

    asyncio.run(
        main(args)
    )
