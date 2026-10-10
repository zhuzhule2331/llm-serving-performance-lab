# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp08_mbt_concurrency/exp08_infra_FROZEN.py
# Copied  : verbatim except for the changes listed in ../../PUBLIC_CODE_CHANGELOG.md
# Frozen  : this file is the FINAL public-facing implementation used in the
#           formal campaign; it is a copy, the original is unmodified.
# =============================================================================
#!/usr/bin/env python3

import csv
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

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

BURST_SCRIPT = ROOT / "exp07_burst.py"

VLLM_BIN = Path(VLLM_BIN)

MODEL_PATH = Path(MODEL_PATH_STR)

PORT = 8000

MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 16
GPU_MEMORY_UTILIZATION = 0.90

CONCURRENCY = 8
MAX_LAUNCH_SKEW_MS = 25.0

POWER_TARGET_W = 220.0
POWER_TOLERANCE_W = 0.2

RANDOM_SEED = 20261007

MBTS = [
    512,
    2048,
    4096,
]

WORKLOADS = [
    {
        "name": "W1_512_128",
        "prompt_tokens": 512,
        "completion_tokens": 128,
    },
    {
        "name": "W2_2048_128",
        "prompt_tokens": 2048,
        "completion_tokens": 128,
    },
    {
        "name": "W3_2048_512",
        "prompt_tokens": 2048,
        "completion_tokens": 512,
    },
    {
        "name": "W4_512_512",
        "prompt_tokens": 512,
        "completion_tokens": 512,
    },
]

WARMUPS_PER_CONDITION = 1
MEASURED_PER_CONDITION = 2

SERVER_READY_TIMEOUT_S = 180
SERVER_READY_POLL_S = 2


def run_cmd(
    args,
    *,
    check=True,
    capture_output=True,
    text=True,
    env=None,
):
    return subprocess.run(
        args,
        check=check,
        capture_output=capture_output,
        text=text,
        env=env,
    )


def kill_existing_vllm():
    subprocess.run(
        [
            "pkill",
            "-f",
            f"{VLLM_BIN} serve",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(3)


def query_power_state():
    proc = run_cmd(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu="
            "power.limit,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )

    line = proc.stdout.strip()

    parts = [
        x.strip()
        for x in line.split(",")
    ]

    if len(parts) != 3:
        raise RuntimeError(
            f"Unexpected nvidia-smi output: {line!r}"
        )

    power_w = float(parts[0])
    memory_mib = float(parts[1])
    temperature_c = float(parts[2])

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "power_limit_w": power_w,
        "memory_used_mib": memory_mib,
        "temperature_c": temperature_c,
    }


def validate_power(state):
    power_w = state["power_limit_w"]

    if abs(
        power_w - POWER_TARGET_W
    ) > POWER_TOLERANCE_W:
        raise RuntimeError(
            "POWER VALIDITY FAILED: "
            f"GPU0 power.limit={power_w:.2f} W, "
            f"expected {POWER_TARGET_W:.2f} W"
        )


def save_power_state(path, state):
    path.write_text(
        json.dumps(
            state,
            indent=2,
        )
        + "\n"
    )


def server_command(mbt):
    return [
        str(VLLM_BIN),
        "serve",
        str(MODEL_PATH),

        "--port",
        str(PORT),

        "--max-model-len",
        str(MAX_MODEL_LEN),

        "--max-num-seqs",
        str(MAX_NUM_SEQS),

        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),

        "--enable-chunked-prefill",

        "--max-num-batched-tokens",
        str(mbt),

        "--no-enable-prefix-caching",
    ]


def start_server(mbt, condition_dir):
    server_log = condition_dir / "server.log"

    log_handle = server_log.open(
        "w",
        buffering=1,
    )

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

    cmd = server_command(mbt)

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
    if proc.poll() is None:
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

    log_handle.close()

    time.sleep(3)


def server_ready():
    proc = subprocess.run(
        [
            "curl",
            "-fsS",
            f"http://127.0.0.1:{PORT}/v1/models",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return proc.returncode == 0


def wait_for_server(proc):
    deadline = (
        time.time()
        + SERVER_READY_TIMEOUT_S
    )

    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "vLLM server exited before becoming ready. "
                f"returncode={proc.returncode}"
            )

        if server_ready():
            return

        time.sleep(
            SERVER_READY_POLL_S
        )

    raise TimeoutError(
        "Timed out waiting for vLLM /v1/models"
    )


def save_server_cmdline(path):
    proc = run_cmd(
        [
            "ps",
            "-ef",
        ],
        check=True,
    )

    lines = []

    for line in proc.stdout.splitlines():
        if (
            f"{VLLM_BIN} serve"
            in line
        ):
            lines.append(line)

    path.write_text(
        "\n".join(lines) + "\n"
    )

    if not lines:
        raise RuntimeError(
            "Could not find vllm serve process in ps output"
        )


def query_preemption():
    proc = run_cmd(
        [
            "curl",
            "-fsS",
            f"http://127.0.0.1:{PORT}/metrics",
        ]
    )

    values = []

    for line in proc.stdout.splitlines():
        if line.startswith(
            "vllm:num_preemptions_total"
        ):
            try:
                values.append(
                    float(
                        line.rsplit(
                            " ",
                            1,
                        )[1]
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed parsing preemption metric: "
                    f"{line!r}"
                ) from exc

    if not values:
        raise RuntimeError(
            "Could not find "
            "vllm:num_preemptions_total"
        )

    return sum(values)


SUMMARY_PATTERNS = {
    "concurrency": re.compile(
        r"^Concurrency\s*:\s*(\d+)"
    ),
    "launch_skew_ms": re.compile(
        r"^Launch skew\s*:\s*"
        r"([0-9.]+)\s*ms"
    ),
    "makespan_s": re.compile(
        r"^Makespan\s*:\s*"
        r"([0-9.]+)\s*s"
    ),
    "mean_ttft_ms": re.compile(
        r"^Mean TTFT\s*:\s*"
        r"([0-9.]+)\s*ms"
    ),
    "median_ttft_ms": re.compile(
        r"^Median TTFT\s*:\s*"
        r"([0-9.]+)\s*ms"
    ),
    "mean_tpot_ms": re.compile(
        r"^Mean TPOT\s*:\s*"
        r"([0-9.]+)\s*ms/token"
    ),
    "median_tpot_ms": re.compile(
        r"^Median TPOT\s*:\s*"
        r"([0-9.]+)\s*ms/token"
    ),
    "mean_e2e_ms": re.compile(
        r"^Mean E2E\s*:\s*"
        r"([0-9.]+)\s*ms"
    ),
    "request_throughput": re.compile(
        r"^Request throughput\s*:\s*"
        r"([0-9.]+)\s*req/s"
    ),
    "output_throughput": re.compile(
        r"^Output throughput\s*:\s*"
        r"([0-9.]+)\s*tok/s"
    ),
}


def parse_burst_output(text):
    result = {
        "validity": None,
    }

    for line in text.splitlines():
        stripped = line.strip()

        if stripped == "VALIDITY: PASS":
            result["validity"] = True

        elif stripped == "VALIDITY: FAIL":
            result["validity"] = False

        for key, pattern in SUMMARY_PATTERNS.items():
            match = pattern.match(stripped)

            if not match:
                continue

            value = match.group(1)

            if key == "concurrency":
                result[key] = int(value)
            else:
                result[key] = float(value)

    required = [
        "concurrency",
        "launch_skew_ms",
        "makespan_s",
        "mean_ttft_ms",
        "median_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "mean_e2e_ms",
        "request_throughput",
        "output_throughput",
        "validity",
    ]

    missing = [
        key
        for key in required
        if key not in result
    ]

    if missing:
        raise RuntimeError(
            "Failed parsing burst output. "
            f"Missing: {missing}"
        )

    return result


def run_burst(
    *,
    prompt_tokens,
    completion_tokens,
    log_path,
):
    cmd = [
        sys.executable,
        str(BURST_SCRIPT),

        "--prompt-tokens",
        str(prompt_tokens),

        "--completion-tokens",
        str(completion_tokens),

        "--concurrency",
        str(CONCURRENCY),

        "--max-model-len",
        str(MAX_MODEL_LEN),

        "--max-launch-skew-ms",
        str(MAX_LAUNCH_SKEW_MS),
    ]

    preemption_before = query_preemption()

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
            "Burst client failed: "
            f"returncode={proc.returncode}"
        )

    parsed = parse_burst_output(
        proc.stdout
    )

    preemption_after = query_preemption()

    preemption_delta = (
        preemption_after
        - preemption_before
    )

    parsed["preemption_before"] = (
        preemption_before
    )

    parsed["preemption_after"] = (
        preemption_after
    )

    parsed["preemption_delta"] = (
        preemption_delta
    )

    parsed["clean_validity"] = (
        parsed["validity"]
        and parsed["launch_skew_ms"]
            <= MAX_LAUNCH_SKEW_MS
        and preemption_delta == 0
    )

    parsed["operating_regime"] = (
        "CLEAN"
        if preemption_delta == 0
        else "STRESS_PREEMPTION"
    )

    print(
        "Preemption delta   : "
        f"{preemption_delta}"
    )

    print(
        "Operating regime   : "
        f"{parsed['operating_regime']}"
    )

    return parsed


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def build_execution_plan():
    rng = random.Random(
        RANDOM_SEED
    )

    mbt_order = MBTS.copy()
    rng.shuffle(mbt_order)

    workload_orders = {}

    for mbt in mbt_order:
        items = [
            dict(x)
            for x in WORKLOADS
        ]

        rng.shuffle(items)

        workload_orders[str(mbt)] = items

    return (
        mbt_order,
        workload_orders,
    )


def aggregate_measured_rows(rows):
    groups = {}

    for row in rows:
        if row["is_warmup"]:
            continue

        key = (
            row["mbt"],
            row["workload"],
            row["prompt_tokens"],
            row["completion_tokens"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    out = []

    metric_fields = [
        "mean_ttft_ms",
        "median_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "mean_e2e_ms",
        "makespan_s",
        "request_throughput",
        "output_throughput",
        "launch_skew_ms",
    ]

    for key, items in sorted(groups.items()):
        (
            mbt,
            workload,
            prompt_tokens,
            completion_tokens,
        ) = key

        row = {
            "mbt": mbt,
            "workload": workload,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "measured_runs": len(items),
            "all_client_valid": all(
                x["validity"]
                for x in items
            ),

            "all_clean_no_preemption": all(
                x["clean_validity"]
                for x in items
            ),

            "total_preemptions": sum(
                x["preemption_delta"]
                for x in items
            ),
        }

        for field in metric_fields:
            row[field] = mean(
                float(x[field])
                for x in items
            )

        out.append(row)

    return out


def main():
    if not BURST_SCRIPT.exists():
        raise RuntimeError(
            f"Missing {BURST_SCRIPT}"
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    out_dir = (
        ROOT
        / f"exp07_pilot_v2_{timestamp}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        mbt_order,
        workload_orders,
    ) = build_execution_plan()

    manifest = {
        "experiment": "Exp07 pilot v2 - per-burst preemption classification",
        "created_at": datetime.now()
        .astimezone()
        .isoformat(),

        "random_seed": RANDOM_SEED,

        "mbt_order": mbt_order,

        "workload_orders": workload_orders,

        "fixed_config": {
            "concurrency": CONCURRENCY,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization":
                GPU_MEMORY_UTILIZATION,

            "prefix_caching": False,
            "chunked_prefill": True,

            "gpu_index": 0,
            "power_limit_target_w":
                POWER_TARGET_W,
        },

        "protocol": {
            "warmups_per_condition":
                WARMUPS_PER_CONDITION,

            "measured_per_condition":
                MEASURED_PER_CONDITION,

            "max_launch_skew_ms":
                MAX_LAUNCH_SKEW_MS,
        },

        "workloads": WORKLOADS,
        "mbts": MBTS,
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

    execution_rows = []

    order_index = 0

    for mbt in mbt_order:
        for workload in workload_orders[str(mbt)]:
            order_index += 1

            execution_rows.append(
                {
                    "order": order_index,
                    "mbt": mbt,
                    "workload":
                        workload["name"],

                    "prompt_tokens":
                        workload[
                            "prompt_tokens"
                        ],

                    "completion_tokens":
                        workload[
                            "completion_tokens"
                        ],
                }
            )

    write_csv(
        out_dir / "execution_order.csv",
        execution_rows,
    )

    all_rows = []

    print("=" * 80)
    print("EXP07 PILOT")
    print("=" * 80)
    print("Output directory:")
    print(out_dir)
    print()
    print("MBT order:")
    print(mbt_order)
    print()

    kill_existing_vllm()

    try:
        for mbt in mbt_order:
            print()
            print("#" * 80)
            print(f"MBT={mbt}")
            print("#" * 80)

            mbt_dir = (
                out_dir
                / f"mbt{mbt}"
            )

            mbt_dir.mkdir()

            power_state = (
                query_power_state()
            )

            save_power_state(
                mbt_dir
                / "power_boundary.json",
                power_state,
            )

            validate_power(
                power_state
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
                    mbt_dir,
                )

                (
                    mbt_dir
                    / "server_command.txt"
                ).write_text(
                    " ".join(cmd)
                    + "\n"
                )

                wait_for_server(
                    proc
                )

                save_server_cmdline(
                    mbt_dir
                    / "server_cmdline.txt"
                )

                pre_before = (
                    query_preemption()
                )

                (
                    mbt_dir
                    / "preemption_before.txt"
                ).write_text(
                    f"{pre_before}\n"
                )

                print(
                    "Preemption before: "
                    f"{pre_before}"
                )

                workloads = (
                    workload_orders[
                        str(mbt)
                    ]
                )

                for workload in workloads:
                    name = workload["name"]
                    p_tokens = (
                        workload[
                            "prompt_tokens"
                        ]
                    )
                    c_tokens = (
                        workload[
                            "completion_tokens"
                        ]
                    )

                    condition_dir = (
                        mbt_dir
                        / name
                    )

                    condition_dir.mkdir()

                    print()
                    print(
                        f"--- MBT={mbt} "
                        f"{name} "
                        f"{p_tokens}->{c_tokens} ---"
                    )

                    # --------------------------------
                    # Excluded warmup
                    # --------------------------------
                    warmup_log = (
                        condition_dir
                        / "warmup.log"
                    )

                    warmup = run_burst(
                        prompt_tokens=p_tokens,
                        completion_tokens=c_tokens,
                        log_path=warmup_log,
                    )

                    warmup_row = {
                        "mbt": mbt,
                        "workload": name,
                        "prompt_tokens":
                            p_tokens,
                        "completion_tokens":
                            c_tokens,

                        "rep": 0,
                        "is_warmup": True,

                        **warmup,
                    }

                    all_rows.append(
                        warmup_row
                    )

                    if not warmup["validity"]:
                        raise RuntimeError(
                            f"Warmup validity failed: "
                            f"MBT={mbt}, {name}"
                        )

                    # --------------------------------
                    # Measured
                    # --------------------------------
                    for rep in range(
                        1,
                        MEASURED_PER_CONDITION
                        + 1,
                    ):
                        log_path = (
                            condition_dir
                            / f"measured_{rep}.log"
                        )

                        result = run_burst(
                            prompt_tokens=p_tokens,
                            completion_tokens=c_tokens,
                            log_path=log_path,
                        )

                        row = {
                            "mbt": mbt,
                            "workload": name,

                            "prompt_tokens":
                                p_tokens,

                            "completion_tokens":
                                c_tokens,

                            "rep": rep,
                            "is_warmup": False,

                            **result,
                        }

                        all_rows.append(
                            row
                        )

                        if not result["validity"]:
                            raise RuntimeError(
                                "Measured validity failed: "
                                f"MBT={mbt}, "
                                f"{name}, rep={rep}"
                            )

                pre_after = (
                    query_preemption()
                )

                (
                    mbt_dir
                    / "preemption_after.txt"
                ).write_text(
                    f"{pre_after}\n"
                )

                pre_delta = (
                    pre_after
                    - pre_before
                )

                (
                    mbt_dir
                    / "preemption_delta.txt"
                ).write_text(
                    f"{pre_delta}\n"
                )

                print()
                print(
                    "Preemption after : "
                    f"{pre_after}"
                )

                print(
                    "Preemption delta : "
                    f"{pre_delta}"
                )

                if pre_delta != 0:
                    print(
                        "NOTE: this MBT block entered a "
                        "preemption regime. Per-burst "
                        "classification is preserved; "
                        "pilot v2 continues."
                    )

            finally:
                if (
                    proc is not None
                    and log_handle is not None
                ):
                    stop_server(
                        proc,
                        log_handle,
                    )

        # --------------------------------------------
        # Save all run-level data
        # --------------------------------------------
        write_csv(
            out_dir
            / "exp07_pilot_runs.csv",
            all_rows,
        )

        measured_means = (
            aggregate_measured_rows(
                all_rows
            )
        )

        write_csv(
            out_dir
            / "exp07_pilot_condition_means.csv",
            measured_means,
        )

        validity_rows = []

        for row in all_rows:
            validity_rows.append(
                {
                    "mbt":
                        row["mbt"],

                    "workload":
                        row["workload"],

                    "rep":
                        row["rep"],

                    "is_warmup":
                        row["is_warmup"],

                    "launch_skew_ms":
                        row[
                            "launch_skew_ms"
                        ],

                    "launch_skew_valid":
                        (
                            row[
                                "launch_skew_ms"
                            ]
                            <=
                            MAX_LAUNCH_SKEW_MS
                        ),

                    "client_validity":
                        row["validity"],

                    "preemption_before":
                        row["preemption_before"],

                    "preemption_after":
                        row["preemption_after"],

                    "preemption_delta":
                        row["preemption_delta"],

                    "clean_validity":
                        row["clean_validity"],

                    "operating_regime":
                        row["operating_regime"],
                }
            )

        write_csv(
            out_dir
            / "exp07_pilot_validity.csv",
            validity_rows,
        )

        print()
        print("=" * 80)
        print("EXP07 PILOT COMPLETE")
        print("=" * 80)
        print(out_dir)

        print()
        print(
            "Condition means:"
        )

        for row in measured_means:
            print(
                f"MBT={row['mbt']:4d} "
                f"{row['workload']:12s} "
                f"TTFT={row['mean_ttft_ms']:.1f} "
                f"TPOT={row['mean_tpot_ms']:.3f} "
                f"req/s={row['request_throughput']:.3f} "
                f"out={row['output_throughput']:.2f}"
            )

    except Exception:
        # Save whatever has been collected so far.
        if all_rows:
            write_csv(
                out_dir
                / "exp07_pilot_runs_PARTIAL.csv",
                all_rows,
            )

        raise

    finally:
        kill_existing_vllm()


if __name__ == "__main__":
    main()
