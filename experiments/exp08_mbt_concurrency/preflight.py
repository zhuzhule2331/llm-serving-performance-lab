# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp08_mbt_concurrency/exp08_preflight.py
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
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

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

PROMPT_TOKENS = 2048
COMPLETION_TOKENS = 128
CONCURRENCY = 8

MBTS = [512, 2048, 4096]

POWER_TARGET_W = 220.0
POWER_TOLERANCE_W = 0.2
MAX_LAUNCH_SKEW_MS = 25.0


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


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


def start_server(mbt, out_dir):
    log_handle = (
        out_dir / "server.log"
    ).open(
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
        capture_output=True,
        text=True,
        check=True,
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
            f"Expected one vllm server, found {len(matching)}"
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
        item
        for item in required
        if item not in line
    ]

    if missing:
        raise RuntimeError(
            f"Server cmdline missing: {missing}"
        )

    return line


def run_burst(log_path):
    pre_before = infra.query_preemption()

    cmd = [
        sys.executable,
        str(BURST_SCRIPT),

        "--prompt-tokens",
        str(PROMPT_TOKENS),

        "--completion-tokens",
        str(COMPLETION_TOKENS),

        "--concurrency",
        str(CONCURRENCY),

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

    log_path.write_text(proc.stdout)

    print(
        proc.stdout,
        end="",
        flush=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Burst failed: returncode={proc.returncode}"
        )

    parsed = infra.parse_burst_output(
        proc.stdout
    )

    pre_after = infra.query_preemption()
    pre_delta = pre_after - pre_before

    parsed["preemption_before"] = pre_before
    parsed["preemption_after"] = pre_after
    parsed["preemption_delta"] = pre_delta

    parsed["clean"] = (
        parsed["validity"]
        and parsed["concurrency"] == CONCURRENCY
        and parsed["launch_skew_ms"]
            <= MAX_LAUNCH_SKEW_MS
        and pre_delta == 0
    )

    print(
        f"Preemption delta : {pre_delta}"
    )

    print(
        f"CLEAN            : {parsed['clean']}"
    )

    return parsed


def main():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    out_dir = (
        ROOT
        / f"exp08_preflight_{timestamp}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = {
        "experiment":
            "Exp08 MBT x Concurrency preflight",

        "created_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "fixed": {
            "prompt_tokens":
                PROMPT_TOKENS,
            "completion_tokens":
                COMPLETION_TOKENS,
            "concurrency":
                CONCURRENCY,
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

        "mbts":
            MBTS,

        "purpose":
            (
                "Validity screening only; "
                "not formal performance evidence."
            ),
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

    rows = []

    infra.kill_existing_vllm()

    try:
        for mbt in MBTS:
            print()
            print("=" * 80)
            print(
                f"EXP08 PREFLIGHT C8 MBT={mbt}"
            )
            print("=" * 80)

            mbt_dir = (
                out_dir
                / f"mbt_{mbt}"
            )

            mbt_dir.mkdir()

            power = infra.query_power_state()

            if abs(
                power["power_limit_w"]
                - POWER_TARGET_W
            ) > POWER_TOLERANCE_W:
                raise RuntimeError(
                    f"Power validity failed: {power}"
                )

            (
                mbt_dir
                / "power_boundary.json"
            ).write_text(
                json.dumps(
                    power,
                    indent=2,
                )
                + "\n"
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

                infra.wait_for_server(
                    proc
                )

                cmdline = verify_server_cmdline(
                    mbt
                )

                (
                    mbt_dir
                    / "server_cmdline.txt"
                ).write_text(
                    cmdline + "\n"
                )

                pre0 = infra.query_preemption()

                if pre0 != 0:
                    raise RuntimeError(
                        f"Fresh server preemption counter={pre0}"
                    )

                warmup = run_burst(
                    mbt_dir
                    / "warmup.log"
                )

                rows.append({
                    "mbt":
                        mbt,
                    "rep":
                        0,
                    "is_warmup":
                        True,
                    **warmup,
                })

                measured = run_burst(
                    mbt_dir
                    / "measured_1.log"
                )

                rows.append({
                    "mbt":
                        mbt,
                    "rep":
                        1,
                    "is_warmup":
                        False,
                    **measured,
                })

            finally:
                if proc is not None:
                    stop_server(
                        proc,
                        log_handle,
                    )

        write_csv(
            out_dir
            / "exp08_preflight_runs.csv",
            rows,
        )

        print()
        print("=" * 80)
        print(
            "EXP08 PREFLIGHT COMPLETE"
        )
        print("=" * 80)

        for mbt in MBTS:
            items = [
                r
                for r in rows
                if r["mbt"] == mbt
            ]

            print(
                f"MBT={mbt:4d} "
                f"C8 "
                f"clean="
                f"{all(r['clean'] for r in items)} "
                f"preemptions="
                f"{sum(r['preemption_delta'] for r in items):.0f}"
            )

    finally:
        infra.kill_existing_vllm()


if __name__ == "__main__":
    main()
