# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp11_prefix_mbt/exp11_namespace_map.py
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

OUT_CSV = (
    ROOT
    / "exp11_formal_namespace_map.csv"
)

OUT_HASH = (
    ROOT
    / "exp11_formal_namespace_map_SHA256.txt"
)

SEARCH_LOG = (
    ROOT
    / "exp11_namespace_map_search.jsonl"
)

DESIRED_RENDERED = {
    "P50": 995,
    "PHIGH": 1953,
}

MAX_PAD_COUNT = 128


def sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def write_csv(path, rows):

    if not rows:
        raise RuntimeError(
            "Refusing to write empty namespace map"
        )

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


def logical_namespace(row):

    return core.formal_namespace(
        row["block_id"],
        row["mbt"],
        row["reuse_condition"],
        row["phase"],
        row["rep"],
    )


def candidate_namespace(
    logical,
    pad_count,
):

    # logical ID guarantees replica identity.
    #
    # Repeated compatibility padding changes tokenizer
    # alignment while preserving deterministic construction.
    return (
        f"{logical}_COMPAT{pad_count:03d}"
        + (
            " compat"
            * pad_count
        )
    )


async def fast_candidate_probe(
    client,
    model_name,
    namespace,
    reuse,
    desired,
):

    base = (
        f"{namespace} "
        f"{reuse} "
        f"isolated shared namespace.\n"
        f"Shared reusable context begins.\n"
    )

    base_tokens = await (
        core.exp03.rendered_tokens(
            client,
            model_name,
            base,
        )
    )

    one_unit_tokens = await (
        core.exp03.rendered_tokens(
            client,
            model_name,
            base
            + core.exp03.SHARED_UNIT,
        )
    )

    increment = (
        one_unit_tokens
        - base_tokens
    )

    if increment <= 0:

        return {
            "possible": False,
            "base_tokens": base_tokens,
            "increment": increment,
            "tested_repeat_count": None,
        }

    approx = int(
        round(
            (
                desired
                - base_tokens
            )
            / increment
        )
    )

    repeat_candidates = sorted(
        {
            max(
                0,
                approx + delta,
            )
            for delta in [
                -2,
                -1,
                0,
                1,
                2,
            ]
        }
    )

    for repeat_count in (
        repeat_candidates
    ):

        text = (
            base
            + core.exp03.SHARED_UNIT
            * repeat_count
        )

        rendered = await (
            core.exp03.rendered_tokens(
                client,
                model_name,
                text,
            )
        )

        if rendered == desired:

            return {
                "possible": True,
                "base_tokens": base_tokens,
                "increment": increment,
                "tested_repeat_count":
                    repeat_count,
            }

    return {
        "possible": False,
        "base_tokens": base_tokens,
        "increment": increment,
        "tested_repeat_count": None,
    }


async def find_compatible_namespace(
    client,
    model_name,
    row,
):

    reuse = row[
        "reuse_condition"
    ]

    desired = (
        DESIRED_RENDERED[
            reuse
        ]
    )

    logical = (
        logical_namespace(
            row
        )
    )

    nominal_target = (
        core.BASE_NOMINAL_TARGETS[
            reuse
        ]
    )

    for pad_count in range(
        0,
        MAX_PAD_COUNT + 1,
    ):

        namespace = (
            candidate_namespace(
                logical,
                pad_count,
            )
        )

        fast = await (
            fast_candidate_probe(
                client,
                model_name,
                namespace,
                reuse,
                desired,
            )
        )

        with SEARCH_LOG.open(
            "a"
        ) as f:

            f.write(
                json.dumps(
                    {
                        "formal_row_id":
                            row[
                                "formal_row_id"
                            ],

                        "reuse_condition":
                            reuse,

                        "pad_count":
                            pad_count,

                        "base_tokens":
                            fast[
                                "base_tokens"
                            ],

                        "increment":
                            fast[
                                "increment"
                            ],

                        "fast_possible":
                            fast[
                                "possible"
                            ],

                        "tested_repeat_count":
                            fast[
                                "tested_repeat_count"
                            ],
                    }
                )
                + "\n"
            )

        if not fast[
            "possible"
        ]:

            continue

        # Final confirmation uses exactly the same
        # builder that formal execution will use.
        result = await (
            core.render_shared_candidate(
                client,
                model_name,
                namespace,
                reuse,
                nominal_target,
            )
        )

        if (
            int(
                result[
                    "rendered_tokens"
                ]
            )
            != desired
        ):

            continue

        shared_prefix = (
            result[
                "shared_prefix"
            ]
        )

        return {
            "formal_row_id":
                row[
                    "formal_row_id"
                ],

            "block_id":
                row[
                    "block_id"
                ],

            "whole_plot_position":
                row[
                    "whole_plot_position"
                ],

            "mbt":
                row[
                    "mbt"
                ],

            "subplot_position":
                row[
                    "subplot_position"
                ],

            "reuse_condition":
                reuse,

            "phase":
                row[
                    "phase"
                ],

            "rep":
                row[
                    "rep"
                ],

            "logical_namespace":
                logical,

            "frozen_namespace":
                namespace,

            "namespace_pad_count":
                pad_count,

            "nominal_target_used":
                int(
                    result[
                        "nominal_target"
                    ]
                ),

            "rendered_shared_tokens":
                int(
                    result[
                        "rendered_tokens"
                    ]
                ),

            "shared_prefix_sha256":
                sha256_text(
                    shared_prefix
                ),

            "construction_base_tokens":
                fast[
                    "base_tokens"
                ],

            "construction_increment":
                fast[
                    "increment"
                ],
        }

    raise RuntimeError(
        "NO COMPATIBLE NAMESPACE FOUND: "
        f"formal_row_id="
        f"{row['formal_row_id']}, "
        f"reuse={reuse}, "
        f"searched pad_count=0.."
        f"{MAX_PAD_COUNT}"
    )


async def main():

    if OUT_CSV.exists():
        raise RuntimeError(
            f"{OUT_CSV.name} already exists. "
            "Refusing to overwrite frozen artifact."
        )

    if OUT_HASH.exists():
        raise RuntimeError(
            f"{OUT_HASH.name} already exists. "
            "Refusing to overwrite frozen artifact."
        )

    if SEARCH_LOG.exists():
        SEARCH_LOG.unlink()

    plan = (
        runner.load_and_validate_plan()
    )

    cached_rows = [
        row
        for row in plan
        if row[
            "reuse_condition"
        ] in {
            "P50",
            "PHIGH",
        }
    ]

    if len(cached_rows) != 54:

        raise RuntimeError(
            "Expected exactly 54 cached-prefix "
            f"plan rows, got {len(cached_rows)}"
        )

    print(
        "PLAN HASH/STRUCTURE: PASS"
    )

    print(
        "cached replicas:",
        len(cached_rows),
    )

    server_dir = (
        ROOT
        / "exp11_namespace_map_server"
    )

    if server_dir.exists():

        raise RuntimeError(
            f"{server_dir} already exists; "
            "inspect/remove it before rerun."
        )

    server_dir.mkdir()

    proc = None
    log_handle = None

    try:

        infra.kill_existing_vllm()

        power = runner.query_power()

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

        if (
            infra.query_preemption()
            != 0
        ):

            raise RuntimeError(
                "Fresh construction server "
                "preemption counter is not zero"
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

            output_rows = []

            for index, row in enumerate(
                cached_rows,
                start=1,
            ):

                print(
                    f"[{index:02d}/54] "
                    f"formal_row="
                    f"{row['formal_row_id']:02d} "
                    f"B{row['block_id']} "
                    f"MBT={row['mbt']} "
                    f"{row['reuse_condition']} "
                    f"{row['phase']} "
                    f"R{row['rep']}"
                )

                result = await (
                    find_compatible_namespace(
                        client,
                        model_name,
                        row,
                    )
                )

                print(
                    "    PASS "
                    f"pad="
                    f"{result['namespace_pad_count']} "
                    f"rendered="
                    f"{result['rendered_shared_tokens']} "
                    f"sha="
                    f"{result['shared_prefix_sha256'][:12]}"
                )

                output_rows.append(
                    result
                )

        # Render-only construction should not create
        # inference preemption. Verify nevertheless.
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
                "Construction server accumulated "
                f"preemption={preempt_end}"
            )

        # Structural checks before freezing.
        if len(output_rows) != 54:

            raise RuntimeError(
                "Namespace map row count mismatch"
            )

        if len({
            r["frozen_namespace"]
            for r in output_rows
        }) != 54:

            raise RuntimeError(
                "Frozen namespaces are not unique"
            )

        for r in output_rows:

            expected = (
                DESIRED_RENDERED[
                    r[
                        "reuse_condition"
                    ]
                ]
            )

            if (
                r[
                    "rendered_shared_tokens"
                ]
                != expected
            ):

                raise RuntimeError(
                    "Rendered target mismatch "
                    f"in row "
                    f"{r['formal_row_id']}"
                )

        write_csv(
            OUT_CSV,
            output_rows,
        )

        digest = (
            sha256_file(
                OUT_CSV
            )
        )

        OUT_HASH.write_text(
            f"{digest}  "
            f"{OUT_CSV.name}\n"
        )

        print()
        print("=" * 80)
        print(
            "NAMESPACE MAP CONSTRUCTION PASS"
        )
        print("=" * 80)

        print(
            "rows:",
            len(output_rows),
        )

        print(
            "unique namespaces:",
            len({
                r["frozen_namespace"]
                for r in output_rows
            }),
        )

        print(
            "sha256:",
            digest,
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
