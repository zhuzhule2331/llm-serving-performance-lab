# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp11_prefix_mbt/exp11_formal_runner.py
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
import os
import shutil
import signal
import subprocess
import time

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import (
    AsyncOpenAI,
    DefaultAioHttpClient,
)

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

PLAN_PATH = (
    ROOT
    / "exp11_preregistered_plan.csv"
)

PLAN_HASH_PATH = (
    ROOT
    / "exp11_preregistered_plan_SHA256.txt"
)

EXPECTED_PLAN_SHA256 = (
    "635f4a8f2bd95983ea08efbed575691e89dede5889446e71aa78289612789266"
)

BLOCKS = [1, 2, 3]
MBTS = [512, 2048, 4096]

REUSE_CONDITIONS = [
    "P0",
    "P50",
    "PHIGH",
]

CONCURRENCY = 8

POWER_TARGET_W = 220.0
POWER_TOLERANCE_W = 0.2

MAX_LAUNCH_SKEW_MS = 25.0


# ============================================================
# Basic I/O
# ============================================================

def write_csv(path, rows):

    if not rows:
        path.write_text("")
        return

    fields = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# Frozen plan validation
# ============================================================

def load_and_validate_plan():

    actual_hash = sha256_file(
        PLAN_PATH
    )

    if actual_hash != EXPECTED_PLAN_SHA256:

        raise RuntimeError(
            "PREREGISTERED PLAN HASH MISMATCH: "
            f"expected={EXPECTED_PLAN_SHA256}, "
            f"actual={actual_hash}"
        )

    with PLAN_PATH.open(
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 81:
        raise RuntimeError(
            f"Expected 81 plan rows, got {len(rows)}"
        )

    normalized = []

    for r in rows:

        row = {
            "formal_row_id":
                int(r["formal_row_id"]),

            "block_id":
                int(r["block_id"]),

            "whole_plot_position":
                int(r[
                    "whole_plot_position"
                ]),

            "mbt":
                int(r["mbt"]),

            "subplot_position":
                int(r[
                    "subplot_position"
                ]),

            "reuse_condition":
                r["reuse_condition"],

            "phase":
                r["phase"],

            "rep":
                int(r["rep"]),

            "include_in_formal":
                r[
                    "include_in_formal"
                ].lower()
                == "true",

            "concurrency":
                int(r["concurrency"]),
        }

        normalized.append(row)

    normalized.sort(
        key=lambda x:
            x["formal_row_id"]
    )

    if [
        r["formal_row_id"]
        for r in normalized
    ] != list(
        range(1, 82)
    ):

        raise RuntimeError(
            "formal_row_id is not exactly 1..81"
        )

    warmups = [
        r for r in normalized
        if r["phase"] == "WARMUP"
    ]

    measured = [
        r for r in normalized
        if r["phase"] == "MEASURED"
    ]

    if len(warmups) != 27:
        raise RuntimeError(
            f"Expected 27 warmups, got {len(warmups)}"
        )

    if len(measured) != 54:
        raise RuntimeError(
            f"Expected 54 measured, got {len(measured)}"
        )

    if {
        r["concurrency"]
        for r in normalized
    } != {CONCURRENCY}:

        raise RuntimeError(
            "Plan concurrency is not fixed C8"
        )

    cells = defaultdict(list)

    for r in normalized:

        key = (
            r["block_id"],
            r["mbt"],
            r["reuse_condition"],
        )

        cells[key].append(r)

    if len(cells) != 27:
        raise RuntimeError(
            f"Expected 27 cells, got {len(cells)}"
        )

    expected_triplet = sorted([
        ("WARMUP", 0),
        ("MEASURED", 1),
        ("MEASURED", 2),
    ])

    for key, rs in cells.items():

        observed = sorted(
            (
                r["phase"],
                r["rep"],
            )
            for r in rs
        )

        if observed != expected_triplet:

            raise RuntimeError(
                "Cell execution triplet invalid: "
                f"{key}: {observed}"
            )

    return normalized


# ============================================================
# Server lifecycle
# ============================================================

def server_cmd(mbt):

    return [
        str(infra.VLLM_BIN),
        "serve",
        str(infra.MODEL_PATH),

        "--port",
        str(infra.PORT),

        "--max-model-len",
        str(infra.MAX_MODEL_LEN),

        "--max-num-seqs",
        str(infra.MAX_NUM_SEQS),

        "--gpu-memory-utilization",
        str(
            infra.GPU_MEMORY_UTILIZATION
        ),

        "--enable-chunked-prefill",

        "--max-num-batched-tokens",
        str(mbt),

        # EXP11 DIFFERENCE FROM EXP08:
        # APC MUST BE ON.
        "--enable-prefix-caching",
    ]


def query_power():

    state = (
        infra.query_power_state()
    )

    power = float(
        state["power_limit_w"]
    )

    if abs(
        power - POWER_TARGET_W
    ) > POWER_TOLERANCE_W:

        raise RuntimeError(
            "POWER VALIDITY FAILED: "
            f"{power} W"
        )

    return state


def start_server(
    mbt,
    server_dir,
):

    log_path = (
        server_dir
        / "server.log"
    )

    log_handle = log_path.open(
        "w",
        buffering=1,
    )

    env = os.environ.copy()

    env[
        "CUDA_VISIBLE_DEVICES"
    ] = "0"

    env[
        "VLLM_WSL2_ENABLE_PIN_MEMORY"
    ] = "1"

    cmd = server_cmd(
        mbt
    )

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )

    return (
        proc,
        log_handle,
        cmd,
    )


def stop_server(
    proc,
    log_handle,
):

    if (
        proc is not None
        and
        proc.poll() is None
    ):

        try:

            os.killpg(
                proc.pid,
                signal.SIGTERM,
            )

        except ProcessLookupError:

            pass

        try:

            proc.wait(
                timeout=20
            )

        except subprocess.TimeoutExpired:

            try:

                os.killpg(
                    proc.pid,
                    signal.SIGKILL,
                )

            except ProcessLookupError:

                pass

            proc.wait(
                timeout=10
            )

    if log_handle is not None:

        log_handle.close()

    time.sleep(3)


def verify_server_cmdline(
    mbt,
):

    proc = subprocess.run(
        [
            "ps",
            "-ef",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    matching = [
        line
        for line
        in proc.stdout.splitlines()
        if (
            f"{VLLM_BIN} serve"
            in line
        )
    ]

    if len(matching) != 1:

        raise RuntimeError(
            "Expected exactly one vllm serve process, "
            f"found {len(matching)}"
        )

    line = matching[0]

    required = [
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--max-num-seqs 16",
        "--max-model-len 8192",
        "--gpu-memory-utilization 0.9",
        (
            "--max-num-batched-tokens "
            f"{mbt}"
        ),
    ]

    forbidden = [
        "--no-enable-prefix-caching",
    ]

    missing = [
        x
        for x in required
        if x not in line
    ]

    present_forbidden = [
        x
        for x in forbidden
        if x in line
    ]

    if missing:

        raise RuntimeError(
            "SERVER CMDLINE VALIDITY FAILED. "
            f"Missing: {missing}"
        )

    if present_forbidden:

        raise RuntimeError(
            "SERVER CMDLINE VALIDITY FAILED. "
            "Forbidden flags present: "
            f"{present_forbidden}"
        )

    return line


# ============================================================
# Prefix construction preflight
#
# This freezes rendered prefix LENGTHS only.
# Formal prefix CONTENT is regenerated independently for every
# warmup/measured replica.
# ============================================================

async def run_prefix_preflight(
    client,
    model_name,
    out_dir,
):

    fixed = {
        "P0": 0,
    }

    rows = []

    for reuse in [
        "P50",
        "PHIGH",
    ]:

        canonical = (
            await core
            .determine_canonical_rendered_length(
                client,
                model_name,
                reuse,
            )
        )

        desired = int(
            canonical[
                "rendered_tokens"
            ]
        )

        fixed[
            reuse
        ] = desired

        # Verify the frozen length across several
        # independent namespaces.
        for namespace_index in range(
            1,
            10,
        ):

            namespace = (
                "EXP11_PREFLIGHT_"
                f"{reuse}_"
                f"N{namespace_index:02d}"
            )

            result = await (
                core.find_exact_rendered_prefix(
                    client,
                    model_name,
                    namespace,
                    reuse,
                    desired,
                    core.BASE_NOMINAL_TARGETS[
                        reuse
                    ],
                )
            )

            if result is None:

                raise RuntimeError(
                    "PREFIX PREFLIGHT FAILED: "
                    f"reuse={reuse}, "
                    f"namespace={namespace}"
                )

            observed = int(
                result[
                    "rendered_tokens"
                ]
            )

            if observed != desired:

                raise RuntimeError(
                    "PREFIX PREFLIGHT LENGTH MISMATCH: "
                    f"reuse={reuse}, "
                    f"expected={desired}, "
                    f"observed={observed}"
                )

            rows.append(
                {
                    "reuse_condition":
                        reuse,

                    "namespace_index":
                        namespace_index,

                    "namespace":
                        namespace,

                    "rendered_shared_tokens":
                        observed,

                    "nominal_target_used":
                        int(
                            result[
                                "nominal_target"
                            ]
                        ),
                }
            )

    write_csv(
        out_dir
        / "prefix_preflight.csv",
        rows,
    )

    (
        out_dir
        / "fixed_rendered_lengths.json"
    ).write_text(
        json.dumps(
            fixed,
            indent=2,
        )
        + "\n"
    )

    return fixed


# ============================================================
# Dry-run display
# ============================================================

def print_plan(
    rows,
):

    print()
    print("=" * 80)
    print("EXP11 FROZEN PLAN")
    print("=" * 80)

    for block in BLOCKS:

        print()
        print(
            f"BLOCK {block}"
        )

        block_rows = [
            r for r in rows
            if r["block_id"]
            == block
        ]

        whole_plots = sorted(
            {
                (
                    r[
                        "whole_plot_position"
                    ],
                    r["mbt"],
                )
                for r in block_rows
            }
        )

        for pos, mbt in whole_plots:

            warmup_rows = [
                r
                for r in block_rows
                if (
                    r[
                        "whole_plot_position"
                    ]
                    == pos
                    and
                    r["mbt"]
                    == mbt
                    and
                    r["phase"]
                    == "WARMUP"
                )
            ]

            warmup_rows.sort(
                key=lambda x:
                    x["subplot_position"]
            )

            reuse_order = [
                r["reuse_condition"]
                for r in warmup_rows
            ]

            print(
                f"  whole-plot {pos}: "
                f"MBT={mbt} | "
                + " -> ".join(
                    reuse_order
                )
            )


# ============================================================
# Manifest
# ============================================================

def write_manifest(
    out_dir,
    fixed_rendered_lengths=None,
):

    manifest = {
        "experiment":
            "Exp11 Prefix Reuse x Scheduler Token Budget",

        "created_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "plan_sha256":
            EXPECTED_PLAN_SHA256,

        "factors": {
            "mbt":
                MBTS,

            "reuse_condition":
                REUSE_CONDITIONS,
        },

        "fixed_config": {
            "concurrency":
                CONCURRENCY,

            "prompt_tokens":
                core.TOTAL_PROMPT_TOKENS,

            "completion_tokens":
                core.COMPLETION_TOKENS,

            "max_num_seqs":
                infra.MAX_NUM_SEQS,

            "max_model_len":
                infra.MAX_MODEL_LEN,

            "gpu_memory_utilization":
                infra.GPU_MEMORY_UTILIZATION,

            "prefix_caching":
                True,

            "chunked_prefill":
                True,

            "power_target_w":
                POWER_TARGET_W,

            "max_launch_skew_ms":
                MAX_LAUNCH_SKEW_MS,
        },

        "design":
            (
                "3-block position-balanced split-plot; "
                "MBT whole-plot factor; "
                "prefix-reuse randomized subplot factor"
            ),

        "protocol": {
            "warmups_per_cell":
                1,

            "measured_per_cell":
                2,

            "block_failure_rule":
                (
                    "Any formal validity failure "
                    "invalidates the affected block."
                ),

            "preemption_hard_validity":
                True,

            "formal_prefix_content":
                (
                    "Regenerated with a unique namespace "
                    "for every warmup/measured replica."
                ),
        },

        "fixed_rendered_lengths":
            fixed_rendered_lengths,
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


# ============================================================
# Main
# ============================================================

async def async_main(args):

    plan = (
        load_and_validate_plan()
    )

    print(
        "PLAN HASH: PASS"
    )

    print(
        "PLAN STRUCTURE: PASS"
    )

    print_plan(
        plan
    )

    if args.dry_run:

        print()
        print(
            "DRY RUN COMPLETE"
        )

        print(
            "No server was started."
        )

        return

    raise RuntimeError(
        "Formal execution is intentionally disabled "
        "in this skeleton. "
        "Run the dedicated Exp11 diagnostic/preflight "
        "before enabling formal collection."
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    asyncio.run(
        async_main(args)
    )


if __name__ == "__main__":
    main()
