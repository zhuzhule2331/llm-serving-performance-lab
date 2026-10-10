# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp08_mbt_concurrency/exp08_formal.py
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

import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

from common import server_lifecycle as infra

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



BURST_SCRIPT = ROOT / "exp08_burst_FROZEN.py"

FORMAL_SEED = 2026100801

BLOCKS = [1, 2, 3]

MBTS = [512, 2048, 4096]

WORKLOADS = [
    {
        "name": "C1",
        "prompt_tokens": 2048,
        "completion_tokens": 128,
        "concurrency": 1,
    },
    {
        "name": "C2",
        "prompt_tokens": 2048,
        "completion_tokens": 128,
        "concurrency": 2,
    },
    {
        "name": "C4",
        "prompt_tokens": 2048,
        "completion_tokens": 128,
        "concurrency": 4,
    },
    {
        "name": "C8",
        "prompt_tokens": 2048,
        "completion_tokens": 128,
        "concurrency": 8,
    },
]

WARMUPS_PER_CONDITION = 1
MEASURED_PER_CONDITION = 2

POWER_TARGET_W = 220.0
POWER_TOLERANCE_W = 0.2

MAX_LAUNCH_SKEW_MS = 25.0


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return

    fields = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_plan():
    """
    Position-balanced MBT ordering:
    every MBT occupies each block position exactly once.

    A random permutation chooses the base row.
    The following blocks are cyclic rotations.
    """

    rng = random.Random(FORMAL_SEED)

    base_order = MBTS.copy()
    rng.shuffle(base_order)

    mbt_orders = {}

    for i, block in enumerate(BLOCKS):
        rotated = (
            base_order[i:]
            + base_order[:i]
        )
        mbt_orders[str(block)] = rotated

    workload_orders = {}

    for block in BLOCKS:
        workload_orders[str(block)] = {}

        for mbt in mbt_orders[str(block)]:
            items = [
                dict(x)
                for x in WORKLOADS
            ]

            rng.shuffle(items)

            workload_orders[str(block)][str(mbt)] = items

    return {
        "base_mbt_order": base_order,
        "mbt_orders": mbt_orders,
        "workload_orders": workload_orders,
    }


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
        str(infra.GPU_MEMORY_UTILIZATION),

        "--enable-chunked-prefill",

        "--max-num-batched-tokens",
        str(mbt),

        "--no-enable-prefix-caching",
    ]


def query_power():
    state = infra.query_power_state()

    power = state["power_limit_w"]

    if abs(
        power - POWER_TARGET_W
    ) > POWER_TOLERANCE_W:
        raise RuntimeError(
            "POWER VALIDITY FAILED: "
            f"{power} W"
        )

    return state


def start_server(mbt, server_dir):
    log_path = (
        server_dir / "server.log"
    )

    log_handle = log_path.open(
        "w",
        buffering=1,
    )

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

    cmd = server_cmd(mbt)

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )

    return proc, log_handle, cmd


def stop_server(proc, log_handle):
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(
                proc.pid,
                signal.SIGTERM,
            )
        except ProcessLookupError:
            pass

        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(
                    proc.pid,
                    signal.SIGKILL,
                )
            except ProcessLookupError:
                pass

            proc.wait(timeout=10)

    if log_handle is not None:
        log_handle.close()

    time.sleep(3)


def verify_server_cmdline(mbt):
    proc = subprocess.run(
        ["ps", "-ef"],
        check=True,
        capture_output=True,
        text=True,
    )

    matching = [
        line
        for line in proc.stdout.splitlines()
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
        "--no-enable-prefix-caching",
        "--max-num-seqs 16",
        "--max-model-len 8192",
        "--gpu-memory-utilization 0.9",
        f"--max-num-batched-tokens {mbt}",
    ]

    missing = [
        x
        for x in required
        if x not in line
    ]

    if missing:
        raise RuntimeError(
            "SERVER CMDLINE VALIDITY FAILED. "
            f"Missing: {missing}"
        )

    return line


def parse_burst_output(text):
    return infra.parse_burst_output(
        text
    )


def run_burst(
    prompt_tokens,
    completion_tokens,
    concurrency,
    log_path,
):
    pre_before = (
        infra.query_preemption()
    )

    cmd = [
        sys.executable,
        str(BURST_SCRIPT),

        "--prompt-tokens",
        str(prompt_tokens),

        "--completion-tokens",
        str(completion_tokens),

        "--concurrency",
        str(concurrency),

        "--max-model-len",
        "8192",

        "--max-launch-skew-ms",
        "25",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_path.write_text(
        proc.stdout
    )

    print(
        proc.stdout,
        end="",
        flush=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Burst client exited non-zero: "
            f"{proc.returncode}"
        )

    parsed = parse_burst_output(
        proc.stdout
    )

    pre_after = (
        infra.query_preemption()
    )

    pre_delta = (
        pre_after - pre_before
    )

    parsed["preemption_before"] = (
        pre_before
    )

    parsed["preemption_after"] = (
        pre_after
    )

    parsed["preemption_delta"] = (
        pre_delta
    )

    parsed["formal_validity"] = (
        parsed["validity"]
        and parsed["concurrency"] == concurrency
        and parsed["launch_skew_ms"]
            <= MAX_LAUNCH_SKEW_MS
        and pre_delta == 0
    )

    print(
        "Formal preemption delta : "
        f"{pre_delta}"
    )

    print(
        "Formal burst validity   : "
        f"{parsed['formal_validity']}"
    )

    return parsed


def summarize(rows):
    measured = [
        r
        for r in rows
        if not r["is_warmup"]
    ]

    groups = {}

    for row in measured:
        key = (
            row["mbt"],
            row["workload"],
            row["target_concurrency"],
            row["prompt_tokens"],
            row["completion_tokens"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    summary = []

    metrics = [
        "mean_ttft_ms",
        "mean_tpot_ms",
        "makespan_s",
        "request_throughput",
        "output_throughput",
        "launch_skew_ms",
    ]

    for key, items in sorted(
        groups.items()
    ):
        (
            mbt,
            workload,
            target_concurrency,
            prompt_tokens,
            completion_tokens,
        ) = key

        row = {
            "mbt": mbt,
            "workload": workload,
            "target_concurrency":
                target_concurrency,
            "prompt_tokens":
                prompt_tokens,
            "completion_tokens":
                completion_tokens,
            "n_measured":
                len(items),
            "all_valid":
                all(
                    x["formal_validity"]
                    for x in items
                ),
            "total_preemptions":
                sum(
                    x["preemption_delta"]
                    for x in items
                ),
        }

        for metric in metrics:
            values = [
                float(x[metric])
                for x in items
            ]

            row[f"{metric}_mean"] = (
                mean(values)
            )

            if len(values) >= 2:
                row[f"{metric}_sd"] = (
                    stdev(values)
                )
            else:
                row[f"{metric}_sd"] = ""

        summary.append(row)

    return summary


def relative_to_2048(summary):
    by_workload = {}

    for row in summary:
        by_workload.setdefault(
            row["workload"],
            {},
        )[row["mbt"]] = row

    rows = []

    metrics = [
        "mean_ttft_ms_mean",
        "mean_tpot_ms_mean",
        "makespan_s_mean",
        "request_throughput_mean",
        "output_throughput_mean",
    ]

    for workload, data in sorted(
        by_workload.items()
    ):
        if 2048 not in data:
            continue

        baseline = data[2048]

        for mbt in [512, 4096]:
            if mbt not in data:
                continue

            out = {
                "workload":
                    workload,
                "target_concurrency":
                    int(
                        data[mbt][
                            "target_concurrency"
                        ]
                    ),
                "comparison_mbt":
                    mbt,
                "baseline_mbt":
                    2048,
            }

            for metric in metrics:
                b = float(
                    baseline[metric]
                )

                x = float(
                    data[mbt][metric]
                )

                out[
                    metric + "_pct_vs_2048"
                ] = (
                    (x / b - 1.0)
                    * 100.0
                )

            rows.append(out)

    return rows


def concurrency_scaling(summary):
    """
    Compare C2/C4/C8 against C1 within each MBT.
    """

    by_mbt = {}

    for row in summary:
        mbt = int(row["mbt"])
        c = int(row["target_concurrency"])

        by_mbt.setdefault(
            mbt,
            {},
        )[c] = row

    rows = []

    metrics = [
        "mean_ttft_ms_mean",
        "mean_tpot_ms_mean",
        "makespan_s_mean",
        "request_throughput_mean",
        "output_throughput_mean",
    ]

    for mbt, data in sorted(
        by_mbt.items()
    ):
        if 1 not in data:
            continue

        baseline = data[1]

        for concurrency in [2, 4, 8]:
            if concurrency not in data:
                continue

            out = {
                "mbt":
                    mbt,
                "comparison_concurrency":
                    concurrency,
                "baseline_concurrency":
                    1,
            }

            for metric in metrics:
                b = float(
                    baseline[metric]
                )

                x = float(
                    data[concurrency][metric]
                )

                out[
                    metric + "_pct_vs_C1"
                ] = (
                    (x / b - 1.0)
                    * 100.0
                )

            rows.append(out)

    return rows



def main():
    timestamp = (
        datetime.now()
        .strftime("%Y%m%d_%H%M%S")
    )

    out_dir = (
        ROOT
        / f"exp08_formal_{timestamp}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    plan = build_plan()

    manifest = {
        "experiment":
            "Exp08 Formal MBT x Concurrency",

        "created_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "formal_seed":
            FORMAL_SEED,

        "blocks":
            BLOCKS,

        "workloads":
            WORKLOADS,

        "mbts":
            MBTS,

        "design":
            "position-balanced split-plot randomized blocks",

        "plan":
            plan,

        "factors": {
            "mbt":
                MBTS,
            "concurrency":
                [1, 2, 4, 8],
        },

        "fixed_config": {
            "max_num_seqs":
                infra.MAX_NUM_SEQS,
            "max_model_len":
                infra.MAX_MODEL_LEN,
            "gpu_memory_utilization":
                infra.GPU_MEMORY_UTILIZATION,
            "prefix_caching":
                False,
            "chunked_prefill":
                True,
            "power_target_w":
                POWER_TARGET_W,
        },

        "protocol": {
            "warmups_per_condition":
                WARMUPS_PER_CONDITION,

            "measured_per_condition":
                MEASURED_PER_CONDITION,

            "max_launch_skew_ms":
                MAX_LAUNCH_SKEW_MS,

            "block_failure_rule":
                (
                    "Any formal validity failure "
                    "invalidates the affected block."
                ),
        },
    }

    (
        out_dir / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    # ---------------------------------------------
    # Save preregistered execution order
    # ---------------------------------------------
    execution_rows = []

    global_order = 0

    for block in BLOCKS:
        mbt_order = (
            plan["mbt_orders"][
                str(block)
            ]
        )

        for mbt_position, mbt in enumerate(
            mbt_order,
            start=1,
        ):
            workload_order = (
                plan[
                    "workload_orders"
                ][str(block)][str(mbt)]
            )

            for workload_position, w in enumerate(
                workload_order,
                start=1,
            ):
                global_order += 1

                execution_rows.append(
                    {
                        "global_condition_order":
                            global_order,

                        "block":
                            block,

                        "mbt_position":
                            mbt_position,

                        "mbt":
                            mbt,

                        "workload_position":
                            workload_position,

                        "workload":
                            w["name"],

                        "prompt_tokens":
                            w[
                                "prompt_tokens"
                            ],

                        "completion_tokens":
                            w[
                                "completion_tokens"
                            ],

                        "target_concurrency":
                            w[
                                "concurrency"
                            ],
                    }
                )

    write_csv(
        out_dir
        / "execution_order.csv",
        execution_rows,
    )

    all_rows = []

    block_status = []

    infra.kill_existing_vllm()

    try:
        for block in BLOCKS:
            print()
            print("=" * 80)
            print(
                f"FORMAL BLOCK {block}"
            )
            print("=" * 80)

            block_dir = (
                out_dir
                / f"block_{block}"
            )

            block_dir.mkdir()

            block_valid = True
            failure_reason = ""

            mbt_order = (
                plan["mbt_orders"][
                    str(block)
                ]
            )

            try:
                for mbt_position, mbt in enumerate(
                    mbt_order,
                    start=1,
                ):
                    print()
                    print("#" * 80)
                    print(
                        f"BLOCK={block} "
                        f"MBT={mbt} "
                        f"POSITION={mbt_position}"
                    )
                    print("#" * 80)

                    server_dir = (
                        block_dir
                        / f"mbt_{mbt}"
                    )

                    server_dir.mkdir()

                    # -------------------------
                    # Power boundary evidence
                    # -------------------------
                    power_state = (
                        query_power()
                    )

                    (
                        server_dir
                        / "power_boundary.json"
                    ).write_text(
                        json.dumps(
                            power_state,
                            indent=2,
                        )
                        + "\n"
                    )

                    print(
                        "Power boundary: "
                        f"{power_state['power_limit_w']:.2f} W"
                    )

                    proc = None
                    log_handle = None

                    try:
                        (
                            proc,
                            log_handle,
                            cmd,
                        ) = start_server(
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
                            verify_server_cmdline(
                                mbt
                            )
                        )

                        (
                            server_dir
                            / "server_cmdline.txt"
                        ).write_text(
                            cmdline + "\n"
                        )

                        server_pre_start = (
                            infra.query_preemption()
                        )

                        (
                            server_dir
                            / "preemption_start.txt"
                        ).write_text(
                            f"{server_pre_start}\n"
                        )

                        if server_pre_start != 0:
                            raise RuntimeError(
                                "Fresh server preemption "
                                f"counter not zero: "
                                f"{server_pre_start}"
                            )

                        workload_order = (
                            plan[
                                "workload_orders"
                            ][str(block)][str(mbt)]
                        )

                        for w_position, workload in enumerate(
                            workload_order,
                            start=1,
                        ):
                            name = (
                                workload["name"]
                            )

                            p = workload[
                                "prompt_tokens"
                            ]

                            c = workload[
                                "completion_tokens"
                            ]

                            concurrency = workload[
                                "concurrency"
                            ]

                            cond_dir = (
                                server_dir
                                / name
                            )

                            cond_dir.mkdir()

                            print()
                            print(
                                f"--- BLOCK={block} "
                                f"MBT={mbt} "
                                f"{name} "
                                f"{p}->{c} ---"
                            )

                            # -----------------
                            # Warmup
                            # -----------------
                            warmup = run_burst(
                                p,
                                c,
                                concurrency,
                                cond_dir
                                / "warmup.log",
                            )

                            warmup_row = {
                                "block":
                                    block,

                                "mbt_position":
                                    mbt_position,

                                "mbt":
                                    mbt,

                                "workload_position":
                                    w_position,

                                "workload":
                                    name,

                                "prompt_tokens":
                                    p,

                                "completion_tokens":
                                    c,

                                "target_concurrency":
                                    concurrency,

                                "rep":
                                    0,

                                "is_warmup":
                                    True,

                                **warmup,
                            }

                            all_rows.append(
                                warmup_row
                            )

                            if not warmup[
                                "formal_validity"
                            ]:
                                raise RuntimeError(
                                    "Warmup formal "
                                    "validity failed: "
                                    f"block={block}, "
                                    f"MBT={mbt}, "
                                    f"{name}"
                                )

                            # -----------------
                            # Measured
                            # -----------------
                            for rep in range(
                                1,
                                MEASURED_PER_CONDITION
                                + 1,
                            ):
                                result = run_burst(
                                    p,
                                    c,
                                    concurrency,
                                    cond_dir
                                    / (
                                        f"measured_"
                                        f"{rep}.log"
                                    ),
                                )

                                row = {
                                    "block":
                                        block,

                                    "mbt_position":
                                        mbt_position,

                                    "mbt":
                                        mbt,

                                    "workload_position":
                                        w_position,

                                    "workload":
                                        name,

                                    "prompt_tokens":
                                        p,

                                    "completion_tokens":
                                        c,

                                    "target_concurrency":
                                        concurrency,

                                    "rep":
                                        rep,

                                    "is_warmup":
                                        False,

                                    **result,
                                }

                                all_rows.append(
                                    row
                                )

                                if not result[
                                    "formal_validity"
                                ]:
                                    raise RuntimeError(
                                        "Measured formal "
                                        "validity failed: "
                                        f"block={block}, "
                                        f"MBT={mbt}, "
                                        f"{name}, "
                                        f"rep={rep}"
                                    )

                        server_pre_end = (
                            infra.query_preemption()
                        )

                        (
                            server_dir
                            / "preemption_end.txt"
                        ).write_text(
                            f"{server_pre_end}\n"
                        )

                        if server_pre_end != 0:
                            raise RuntimeError(
                                "Server accumulated "
                                "preemptions: "
                                f"{server_pre_end}"
                            )

                    finally:
                        if (
                            proc is not None
                            and log_handle
                            is not None
                        ):
                            stop_server(
                                proc,
                                log_handle,
                            )

                block_status.append(
                    {
                        "block":
                            block,
                        "valid":
                            True,
                        "failure_reason":
                            "",
                    }
                )

            except Exception as exc:
                block_valid = False
                failure_reason = repr(exc)

                block_status.append(
                    {
                        "block":
                            block,
                        "valid":
                            False,
                        "failure_reason":
                            failure_reason,
                    }
                )

                write_csv(
                    out_dir
                    / "formal_runs_PARTIAL.csv",
                    all_rows,
                )

                write_csv(
                    out_dir
                    / "block_status.csv",
                    block_status,
                )

                raise

        # -----------------------------------------
        # Completed formal
        # -----------------------------------------
        write_csv(
            out_dir
            / "formal_runs.csv",
            all_rows,
        )

        write_csv(
            out_dir
            / "block_status.csv",
            block_status,
        )

        summary = summarize(
            all_rows
        )

        write_csv(
            out_dir
            / "formal_condition_summary.csv",
            summary,
        )

        relative = relative_to_2048(
            summary
        )

        write_csv(
            out_dir
            / "formal_relative_to_mbt2048.csv",
            relative,
        )

        scaling = concurrency_scaling(
            summary
        )

        write_csv(
            out_dir
            / "formal_concurrency_scaling_vs_C1.csv",
            scaling,
        )

        print()
        print("=" * 80)
        print(
            "EXP07-A FORMAL COMPLETE"
        )
        print("=" * 80)

        for row in summary:
            print(
                f"MBT={row['mbt']:4d} "
                f"{row['workload']:13s} "
                f"N={row['n_measured']} "
                f"TTFT="
                f"{row['mean_ttft_ms_mean']:.1f} "
                f"TPOT="
                f"{row['mean_tpot_ms_mean']:.3f} "
                f"makespan="
                f"{row['makespan_s_mean']:.3f} "
                f"req/s="
                f"{row['request_throughput_mean']:.3f} "
                f"out="
                f"{row['output_throughput_mean']:.2f}"
            )

    finally:
        infra.kill_existing_vllm()


if __name__ == "__main__":
    main()
