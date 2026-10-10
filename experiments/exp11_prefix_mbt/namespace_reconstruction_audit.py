# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp11_prefix_mbt/exp11_namespace_reconstruction_audit.py
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

import asyncio
import csv
import hashlib
import json
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

MAP_PATH = (
    ROOT
    / "exp11_formal_namespace_map.csv"
)

EXPECTED_MAP_SHA256 = (
    "a0c97a991652089e9ee4a68bd529fb705da7565c59f1fd78d3ecbdbb19f71502"
)

OUT_CSV = (
    ROOT
    / "exp11_namespace_reconstruction_audit.csv"
)


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def write_csv(path, rows):

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def load_map():

    actual = sha256_file(
        MAP_PATH
    )

    if actual != EXPECTED_MAP_SHA256:

        raise RuntimeError(
            "NAMESPACE MAP HASH MISMATCH: "
            f"expected={EXPECTED_MAP_SHA256}, "
            f"actual={actual}"
        )

    with MAP_PATH.open(
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 54:

        raise RuntimeError(
            f"Expected 54 map rows, got {len(rows)}"
        )

    return rows


async def main():

    rows = load_map()

    print("MAP HASH: PASS")
    print("MAP ROW COUNT: PASS")

    server_dir = (
        ROOT
        / "exp11_namespace_reconstruction_server"
    )

    if server_dir.exists():

        raise RuntimeError(
            f"{server_dir} already exists. "
            "Inspect/remove before rerun."
        )

    server_dir.mkdir()

    proc = None
    log_handle = None

    audit_rows = []

    try:

        infra.kill_existing_vllm()

        power = (
            runner.query_power()
        )

        (
            server_dir
            / "power_boundary.json"
        ).write_text(
            json.dumps(
                power,
                indent=2,
            )
            + "\n"
        )

        (
            proc,
            log_handle,
            cmd,
        ) = runner.start_server(
            2048,
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
            runner.verify_server_cmdline(
                2048
            )
        )

        (
            server_dir
            / "server_cmdline.txt"
        ).write_text(
            cmdline + "\n"
        )

        preempt_start = (
            infra.query_preemption()
        )

        if preempt_start != 0:

            raise RuntimeError(
                "Fresh reconstruction server "
                "preemption counter != 0: "
                f"{preempt_start}"
            )

        async with AsyncOpenAI(
            base_url=core.API_BASE,
            api_key=core.API_KEY,
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

            for index, row in enumerate(
                rows,
                start=1,
            ):

                formal_row_id = int(
                    row[
                        "formal_row_id"
                    ]
                )

                reuse = row[
                    "reuse_condition"
                ]

                namespace = row[
                    "frozen_namespace"
                ]

                nominal = int(
                    row[
                        "nominal_target_used"
                    ]
                )

                expected_rendered = int(
                    row[
                        "rendered_shared_tokens"
                    ]
                )

                expected_sha = row[
                    "shared_prefix_sha256"
                ]

                result = await (
                    core.render_shared_candidate(
                        client,
                        model_name,
                        namespace,
                        reuse,
                        nominal,
                    )
                )

                actual_rendered = int(
                    result[
                        "rendered_tokens"
                    ]
                )

                actual_sha = (
                    sha256_text(
                        result[
                            "shared_prefix"
                        ]
                    )
                )

                rendered_valid = (
                    actual_rendered
                    == expected_rendered
                )

                sha_valid = (
                    actual_sha
                    == expected_sha
                )

                valid = (
                    rendered_valid
                    and
                    sha_valid
                )

                audit_rows.append(
                    {
                        "formal_row_id":
                            formal_row_id,

                        "reuse_condition":
                            reuse,

                        "expected_rendered":
                            expected_rendered,

                        "actual_rendered":
                            actual_rendered,

                        "rendered_valid":
                            rendered_valid,

                        "expected_sha256":
                            expected_sha,

                        "actual_sha256":
                            actual_sha,

                        "sha_valid":
                            sha_valid,

                        "reconstruction_valid":
                            valid,
                    }
                )

                print(
                    f"[{index:02d}/54] "
                    f"formal_row="
                    f"{formal_row_id:02d} "
                    f"{reuse:<5} "
                    f"rendered="
                    f"{actual_rendered} "
                    f"sha="
                    f"{actual_sha[:12]} "
                    f"valid={valid}"
                )

                if not valid:

                    raise RuntimeError(
                        "FROZEN PREFIX "
                        "RECONSTRUCTION FAILED: "
                        f"formal_row_id="
                        f"{formal_row_id}"
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
                "Reconstruction server "
                "accumulated preemption: "
                f"{preempt_end}"
            )

        if not all(
            r[
                "reconstruction_valid"
            ]
            for r in audit_rows
        ):

            raise RuntimeError(
                "At least one reconstruction "
                "audit row failed"
            )

        write_csv(
            OUT_CSV,
            audit_rows,
        )

        print()
        print("=" * 80)
        print(
            "NAMESPACE RECONSTRUCTION AUDIT: PASS"
        )
        print("=" * 80)

        print(
            "rows:",
            len(audit_rows),
        )

        print(
            "valid:",
            sum(
                r[
                    "reconstruction_valid"
                ]
                for r in audit_rows
            ),
        )

        print(
            "preemption:",
            preempt_end,
        )

        print(
            "created_at:",
            datetime.now()
            .astimezone()
            .isoformat(),
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

        infra.kill_existing_vllm()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
