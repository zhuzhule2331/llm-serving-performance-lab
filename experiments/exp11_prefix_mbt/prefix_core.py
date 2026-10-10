# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp11_prefix_mbt/exp11_prefix_core.py
# Copied  : verbatim except for the changes listed in ../../PUBLIC_CODE_CHANGELOG.md
# Frozen  : this file is the FINAL public-facing implementation used in the
#           formal campaign; it is a copy, the original is unmodified.
# =============================================================================
import asyncio
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI, DefaultAioHttpClient

# Exp11 lives one directory below the historical Exp03/04
# modules that provide the validated prefix/cache helpers.
import sys
from pathlib import Path as _Path

_EXP11_DIR = _Path(__file__).resolve().parent
_BENCH_ROOT = _EXP11_DIR.parent

if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

import exp03_formal as exp03


# ============================================================
# 1. Frozen Exp04 Formal Protocol
# ============================================================

API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"

TOTAL_PROMPT_TOKENS = 2048
COMPLETION_TOKENS = 128
PRIMER_COMPLETION_TOKENS = 1

TEMPERATURE = 0.0
TOP_P = 1.0

REUSE_CONDITIONS = [
    "P0",
    "P50",
    "PHIGH",
]

CONCURRENCY_LEVELS = [
    1,
    2,
    4,
    8,
]

NUM_BLOCKS = 3
RANDOM_SEED = 2026101101

MAX_LAUNCH_SKEW_MS = 25.0

MBTS = [512, 2048, 4096]
CONCURRENCY = 8
BLOCKS = [1, 2, 3]
MEASURED_PER_CELL = 2

# ------------------------------------------------------------
# Nominal construction targets.
#
# IMPORTANT:
# These are NOT the final controlled shared-prefix lengths.
# Preflight will determine fixed rendered lengths.
# ------------------------------------------------------------

BASE_NOMINAL_TARGETS = {
    "P50": 1024,
    "PHIGH": 1984,
}

# Search around the nominal target to obtain the exact
# rendered shared-token count required by formal preflight.
PREFLIGHT_SEARCH_RADIUS = 96

# ------------------------------------------------------------
# Raw cache mechanism validity ranges.
#
# No Exp03-style 16-token subtraction is used in Exp04.
# ------------------------------------------------------------

RAW_HIT_RATIO_RANGE = {
    "P0": (
        0.00,
        0.02,
    ),
    "P50": (
        0.35,
        0.65,
    ),
    "PHIGH": (
        0.90,
        1.00,
    ),
}

FORMAL_NAMESPACE = "EXP11_FORMAL_V1"
PRECONDITION_NAMESPACE = "EXP11_PRECONDITION_V1"

FROZEN_NAMESPACE_MAP_PATH = (
    _EXP11_DIR
    / "exp11_formal_namespace_map.csv"
)

FROZEN_NAMESPACE_MAP_SHA256 = (
    "a0c97a991652089e9ee4a68bd529fb705da7565c59f1fd78d3ecbdbb19f71502"
)

_NAMESPACE_MAP_CACHE = None

PRECONDITION_ROUNDS = 2

METRIC_SETTLE_SEC = 0.10


# ============================================================
# 2. Unique suffix markers
# ============================================================

UNIQUE_SUFFIX_WORDS = [
    "Quartz",
    "Nimbus",
    "Falcon",
    "Maple",
    "Cobalt",
    "Velvet",
    "Harbor",
    "Juniper",
]

P0_WORDS = [
    "Orchid",
    "Lantern",
    "Copper",
    "Willow",
    "Kestrel",
    "Zephyr",
    "Granite",
    "Meadow",
    "Saffron",
    "Anchor",
    "Cedar",
    "Marble",
    "Topaz",
    "River",
    "Comet",
    "Basil",
    "Hazel",
    "Ivory",
    "Silver",
    "Aspen",
    "Onyx",
    "Pine",
    "Coral",
    "Linden",
]


# ============================================================
# 3. Output directory
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

RESULT_DIR = Path(
    f"exp04_formal_{timestamp}"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)


# ============================================================
# 4. Helpers
# ============================================================

def percentile(values, p):

    xs = sorted(values)

    if not xs:
        return float("nan")

    if len(xs) == 1:
        return xs[0]

    pos = (
        len(xs) - 1
    ) * p

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    frac = pos - lo

    return (
        xs[lo]
        + (
            xs[hi]
            - xs[lo]
        )
        * frac
    )


def mean_or_nan(values):

    if not values:
        return float("nan")

    return statistics.mean(values)


def std_or_zero(values):

    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


def write_csv(
    path,
    rows,
):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def search_order(
    center,
    radius,
):

    values = [center]

    for delta in range(
        1,
        radius + 1,
    ):

        values.append(
            center - delta
        )

        values.append(
            center + delta
        )

    return values


# ============================================================
# 5. Formal condition list
# ============================================================

FORMAL_CONDITIONS = [
    (
        mbt,
        reuse,
    )
    for mbt in MBTS
    for reuse in REUSE_CONDITIONS
]


# ============================================================
# 6. Namespace
# ============================================================

def preflight_namespace(
    block_id,
    mbt,
    reuse_condition,
):

    return (
        f"{FORMAL_NAMESPACE}_"
        f"PREFLIGHT_"
        f"B{block_id:02d}_"
        f"MBT{mbt}_"
        f"{reuse_condition}"
    )


def formal_namespace(
    block_id,
    mbt,
    reuse_condition,
    phase,
    rep,
):

    return (
        f"{FORMAL_NAMESPACE}_"
        f"B{block_id:02d}_"
        f"MBT{mbt}_"
        f"{reuse_condition}_"
        f"{phase}_"
        f"R{rep:02d}"
    )


# ============================================================
# 7. Shared-prefix preflight
# ============================================================

async def render_shared_candidate(
    client,
    model_name,
    namespace,
    reuse_condition,
    nominal_target,
):

    (
        shared_prefix,
        rendered_tokens,
    ) = await exp03.build_shared_prefix(
        client,
        model_name,
        namespace,
        reuse_condition,
        nominal_target,
    )

    return {
        "shared_prefix":
            shared_prefix,

        "rendered_tokens":
            int(
                rendered_tokens
            ),

        "nominal_target":
            int(
                nominal_target
            ),
    }


async def find_exact_rendered_prefix(
    client,
    model_name,
    namespace,
    reuse_condition,
    desired_rendered_tokens,
    base_nominal_target,
):

    for nominal_target in search_order(
        base_nominal_target,
        PREFLIGHT_SEARCH_RADIUS,
    ):

        result = await render_shared_candidate(
            client,
            model_name,
            namespace,
            reuse_condition,
            nominal_target,
        )

        if (
            result[
                "rendered_tokens"
            ]
            == desired_rendered_tokens
        ):

            return result

    return None


async def determine_canonical_rendered_length(
    client,
    model_name,
    reuse_condition,
):

    """
    Use a canonical namespace to determine the first candidate
    rendered shared-prefix length.

    The candidate is then enforced across every formal
    block/concurrency namespace.
    """

    namespace = (
        f"{FORMAL_NAMESPACE}_"
        f"PREFLIGHT_CANONICAL_"
        f"{reuse_condition}"
    )

    result = await render_shared_candidate(
        client,
        model_name,
        namespace,
        reuse_condition,
        BASE_NOMINAL_TARGETS[
            reuse_condition
        ],
    )

    return result


async def run_shared_prefix_preflight(
    client,
    model_name,
):

    print()
    print(
        "=" * 118
    )

    print(
        "EXP04 FORMAL SHARED-PREFIX PREFLIGHT"
    )

    print(
        "=" * 118
    )

    preflight_map = {}

    fixed_rendered_lengths = {}

    for reuse_condition in [
        "P50",
        "PHIGH",
    ]:

        canonical = (
            await determine_canonical_rendered_length(
                client,
                model_name,
                reuse_condition,
            )
        )

        desired_rendered = (
            canonical[
                "rendered_tokens"
            ]
        )

        fixed_rendered_lengths[
            reuse_condition
        ] = desired_rendered

        print()
        print(
            f"{reuse_condition}:"
        )

        print(
            f"  canonical nominal target = "
            f"{canonical['nominal_target']}"
        )

        print(
            f"  frozen rendered length   = "
            f"{desired_rendered}"
        )

        for block_id in range(
            1,
            NUM_BLOCKS + 1,
        ):

            for mbt in MBTS:

                namespace = preflight_namespace(
                    block_id,
                    mbt,
                    reuse_condition,
                )

                result = (
                    await find_exact_rendered_prefix(
                        client,
                        model_name,
                        namespace,
                        reuse_condition,
                        desired_rendered,
                        BASE_NOMINAL_TARGETS[
                            reuse_condition
                        ],
                    )
                )

                if result is None:

                    raise RuntimeError(
                        "\n"
                        "EXP04 PREFLIGHT FAILED\n"
                        f"Could not obtain fixed rendered "
                        f"shared length={desired_rendered}\n"
                        f"for reuse={reuse_condition}, "
                        f"block={block_id}, "
                        f"MBT={mbt}.\n"
                        "Formal collection has NOT started."
                    )

                preflight_map[
                    (
                        block_id,
                        reuse_condition,
                        mbt,
                    )
                ] = {
                    "namespace":
                        namespace,

                    "shared_prefix":
                        result[
                            "shared_prefix"
                        ],

                    "rendered_shared_tokens":
                        result[
                            "rendered_tokens"
                        ],

                    "nominal_target_used":
                        result[
                            "nominal_target"
                        ],
                }

        observed_lengths = sorted(
            {
                item[
                    "rendered_shared_tokens"
                ]
                for key, item
                in preflight_map.items()
                if key[1]
                == reuse_condition
            }
        )

        if observed_lengths != [
            desired_rendered
        ]:

            raise RuntimeError(
                f"{reuse_condition} preflight "
                f"rendered lengths are not fixed: "
                f"{observed_lengths}"
            )

        print(
            f"  all 32 formal namespaces = PASS"
        )

    print()
    print(
        "Frozen rendered shared-prefix lengths:"
    )

    for condition, length in (
        fixed_rendered_lengths.items()
    ):

        print(
            f"  {condition:<6} = "
            f"{length} tokens"
        )

    print()
    print(
        "SHARED-PREFIX PREFLIGHT = PASS"
    )

    return (
        preflight_map,
        fixed_rendered_lengths,
    )


# ============================================================
# 8. P0 workload construction
# ============================================================

async def build_p0_prompt(
    client,
    model_name,
    namespace,
    unique_word,
    request_id,
    role_name,
):

    base_text = (
        f"{unique_word} begins an isolated "
        f"{role_name} request. "
        f"Namespace={namespace}. "
        f"RequestID={request_id}. "
        f"This workload intentionally contains no designed "
        f"application-specific shared prefix with the other "
        f"requests in this condition. "
        f"The remaining content discusses independent "
        f"inference scheduling, GPU execution, memory "
        f"management, batching, kernels, and serving-system "
        f"behavior. "
    )

    text, token_count = (
        await exp03.fit_exact_prompt(
            client,
            model_name,
            base_text,
        )
    )

    if (
        token_count
        != TOTAL_PROMPT_TOKENS
    ):

        raise RuntimeError(
            f"P0 prompt token mismatch: "
            f"{token_count}"
        )

    return text


async def build_p0_workload(
    client,
    model_name,
    namespace,
    concurrency,
    block_id,
):

    marker_base = (
        block_id
        * 3
        + concurrency
    )

    primer_word = P0_WORDS[
        marker_base
        % len(P0_WORDS)
    ]

    primer_text = await build_p0_prompt(
        client,
        model_name,
        namespace + "_PRIMER",
        primer_word,
        0,
        "primer",
    )

    measured_prompts = []

    for request_id in range(
        1,
        concurrency + 1,
    ):

        word = P0_WORDS[
            (
                marker_base
                + request_id
                + 7
            )
            % len(P0_WORDS)
        ]

        text = await build_p0_prompt(
            client,
            model_name,
            (
                namespace
                + f"_REQ{request_id:02d}"
            ),
            word,
            request_id,
            "measured",
        )

        measured_prompts.append(
            text
        )

    return {
        "primer_text":
            primer_text,

        "measured_prompts":
            measured_prompts,

        "rendered_shared_tokens":
            0,

        "nominal_target_used":
            0,
    }


# ============================================================
# 9. Cached-prefix workload construction
# ============================================================

async def build_cached_workload(
    client,
    model_name,
    shared_prefix,
    rendered_shared_tokens,
    nominal_target_used,
    concurrency,
):

    # --------------------------------------------------------
    # Primer continuation immediately diverges from measured
    # request continuations.
    # --------------------------------------------------------

    primer_base = (
        shared_prefix
        + "Anchor primer-only continuation populates "
        + "the reusable prefix state before the synchronized "
        + "measured burst. "
    )

    (
        primer_text,
        primer_tokens,
    ) = await exp03.fit_exact_prompt(
        client,
        model_name,
        primer_base,
    )

    if (
        primer_tokens
        != TOTAL_PROMPT_TOKENS
    ):

        raise RuntimeError(
            f"Primer token mismatch: "
            f"{primer_tokens}"
        )

    measured_prompts = []

    for request_id in range(
        1,
        concurrency + 1,
    ):

        unique_word = (
            UNIQUE_SUFFIX_WORDS[
                request_id - 1
            ]
        )

        # IMPORTANT:
        # The suffix diverges immediately after the shared
        # prefix. No common measured-suffix preamble exists.
        measured_base = (
            shared_prefix
            + unique_word
            + f" request {request_id:02d} contains "
            + "an independent continuation describing "
            + "scheduler behavior, GPU execution, batching, "
            + "memory pressure, and inference serving. "
        )

        (
            text,
            token_count,
        ) = await exp03.fit_exact_prompt(
            client,
            model_name,
            measured_base,
        )

        if (
            token_count
            != TOTAL_PROMPT_TOKENS
        ):

            raise RuntimeError(
                f"Measured prompt "
                f"{request_id} token mismatch: "
                f"{token_count}"
            )

        measured_prompts.append(
            text
        )

    return {
        "primer_text":
            primer_text,

        "measured_prompts":
            measured_prompts,

        "rendered_shared_tokens":
            rendered_shared_tokens,

        "nominal_target_used":
            nominal_target_used,
    }



def _sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def _sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_frozen_namespace_map():

    global _NAMESPACE_MAP_CACHE

    if _NAMESPACE_MAP_CACHE is not None:

        return _NAMESPACE_MAP_CACHE

    actual_hash = _sha256_file(
        FROZEN_NAMESPACE_MAP_PATH
    )

    if (
        actual_hash
        != FROZEN_NAMESPACE_MAP_SHA256
    ):

        raise RuntimeError(
            "FROZEN NAMESPACE MAP HASH MISMATCH: "
            f"expected="
            f"{FROZEN_NAMESPACE_MAP_SHA256}, "
            f"actual={actual_hash}"
        )

    with FROZEN_NAMESPACE_MAP_PATH.open(
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 54:

        raise RuntimeError(
            "Expected 54 frozen cached-prefix rows, "
            f"got {len(rows)}"
        )

    mapping = {}

    for row in rows:

        key = (
            int(
                row[
                    "block_id"
                ]
            ),
            int(
                row[
                    "mbt"
                ]
            ),
            row[
                "reuse_condition"
            ],
            row[
                "phase"
            ],
            int(
                row[
                    "rep"
                ]
            ),
        )

        if key in mapping:

            raise RuntimeError(
                "Duplicate frozen namespace key: "
                f"{key}"
            )

        mapping[key] = {
            "formal_row_id":
                int(
                    row[
                        "formal_row_id"
                    ]
                ),

            "logical_namespace":
                row[
                    "logical_namespace"
                ],

            "frozen_namespace":
                row[
                    "frozen_namespace"
                ],

            "nominal_target_used":
                int(
                    row[
                        "nominal_target_used"
                    ]
                ),

            "rendered_shared_tokens":
                int(
                    row[
                        "rendered_shared_tokens"
                    ]
                ),

            "shared_prefix_sha256":
                row[
                    "shared_prefix_sha256"
                ],
        }

    if len(mapping) != 54:

        raise RuntimeError(
            "Frozen namespace mapping cardinality "
            "is not 54"
        )

    _NAMESPACE_MAP_CACHE = mapping

    return mapping


async def build_formal_workload(
    client,
    model_name,
    block_id,
    mbt,
    reuse_condition,
    phase,
    rep,
    fixed_rendered_lengths,
):

    logical_namespace = formal_namespace(
        block_id,
        mbt,
        reuse_condition,
        phase,
        rep,
    )

    # --------------------------------------------------------
    # P0 control:
    # no application-level shared prefix.
    # Its namespace is already replica-unique.
    # --------------------------------------------------------

    if reuse_condition == "P0":

        workload = await build_p0_workload(
            client,
            model_name,
            logical_namespace,
            CONCURRENCY,
            block_id,
        )

        workload[
            "namespace"
        ] = logical_namespace

        workload[
            "logical_namespace"
        ] = logical_namespace

        workload[
            "shared_prefix_sha256"
        ] = ""

        workload[
            "namespace_map_sha256"
        ] = FROZEN_NAMESPACE_MAP_SHA256

    else:

        namespace_map = (
            load_frozen_namespace_map()
        )

        key = (
            int(block_id),
            int(mbt),
            reuse_condition,
            phase,
            int(rep),
        )

        if key not in namespace_map:

            raise RuntimeError(
                "No frozen namespace-map entry "
                f"for key={key}"
            )

        info = namespace_map[
            key
        ]

        if (
            info[
                "logical_namespace"
            ]
            != logical_namespace
        ):

            raise RuntimeError(
                "Frozen logical namespace mismatch: "
                f"expected={logical_namespace}, "
                f"map="
                f"{info['logical_namespace']}"
            )

        frozen_namespace = (
            info[
                "frozen_namespace"
            ]
        )

        result = await (
            render_shared_candidate(
                client,
                model_name,
                frozen_namespace,
                reuse_condition,
                info[
                    "nominal_target_used"
                ],
            )
        )

        actual_rendered = int(
            result[
                "rendered_tokens"
            ]
        )

        expected_rendered = int(
            info[
                "rendered_shared_tokens"
            ]
        )

        if (
            actual_rendered
            != expected_rendered
        ):

            raise RuntimeError(
                "FROZEN PREFIX LENGTH "
                "RECONSTRUCTION FAILED: "
                f"key={key}, "
                f"expected={expected_rendered}, "
                f"actual={actual_rendered}"
            )

        actual_sha = _sha256_text(
            result[
                "shared_prefix"
            ]
        )

        expected_sha = (
            info[
                "shared_prefix_sha256"
            ]
        )

        if (
            actual_sha
            != expected_sha
        ):

            raise RuntimeError(
                "FROZEN PREFIX SHA256 "
                "RECONSTRUCTION FAILED: "
                f"key={key}, "
                f"expected={expected_sha}, "
                f"actual={actual_sha}"
            )

        desired_rendered = int(
            fixed_rendered_lengths[
                reuse_condition
            ]
        )

        if (
            actual_rendered
            != desired_rendered
        ):

            raise RuntimeError(
                "FROZEN PREFIX TARGET MISMATCH: "
                f"reuse={reuse_condition}, "
                f"fixed={desired_rendered}, "
                f"actual={actual_rendered}"
            )

        workload = await build_cached_workload(
            client,
            model_name,
            result[
                "shared_prefix"
            ],
            actual_rendered,
            info[
                "nominal_target_used"
            ],
            CONCURRENCY,
        )

        workload[
            "namespace"
        ] = frozen_namespace

        workload[
            "logical_namespace"
        ] = logical_namespace

        workload[
            "shared_prefix_sha256"
        ] = actual_sha

        workload[
            "namespace_map_sha256"
        ] = FROZEN_NAMESPACE_MAP_SHA256

        workload[
            "formal_row_id"
        ] = info[
            "formal_row_id"
        ]

    workload[
        "mbt"
    ] = mbt

    workload[
        "phase"
    ] = phase

    workload[
        "rep"
    ] = rep

    return workload


# ============================================================
# 10. One synchronized burst request
# ============================================================

async def run_burst_request(
    client,
    model_name,
    prompt_text,
    request_id,
    release_event,
    ready_event,
):

    ready_event.set()

    await release_event.wait()

    request_start = (
        time.perf_counter()
    )

    first_content = None
    last_content = None
    usage = None
    error = None

    try:

        async with client.chat.completions.stream(
            model=model_name,

            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],

            temperature=
                TEMPERATURE,

            top_p=
                TOP_P,

            max_tokens=
                COMPLETION_TOKENS,

            extra_body={
                "min_tokens":
                    COMPLETION_TOKENS,

                "ignore_eos":
                    True,
            },

            stream_options={
                "include_usage":
                    True,
            },

        ) as stream:

            async for event in stream:

                if (
                    event.type
                    == "content.delta"
                ):

                    if event.delta:

                        now = (
                            time.perf_counter()
                        )

                        if (
                            first_content
                            is None
                        ):

                            first_content = now

                        last_content = now

                elif (
                    event.type
                    == "chunk"
                ):

                    if (
                        event.chunk.usage
                        is not None
                    ):

                        usage = (
                            event.chunk.usage
                        )

    except Exception as exc:

        error = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    request_end = (
        time.perf_counter()
    )

    if error is not None:

        return {
            "request_id":
                request_id,

            "success":
                False,

            "error":
                error,

            "request_start":
                request_start,

            "request_end":
                request_end,

            "first_content":
                None,

            "last_content":
                None,

            "prompt_tokens":
                None,

            "completion_tokens":
                None,

            "ttft_ms":
                None,

            "tpot_ms":
                None,

            "e2e_ms":
                (
                    request_end
                    - request_start
                )
                * 1000.0,
        }

    if first_content is None:

        return {
            "request_id":
                request_id,

            "success":
                False,

            "error":
                "No non-empty content event.",

            "request_start":
                request_start,

            "request_end":
                request_end,

            "first_content":
                None,

            "last_content":
                None,

            "prompt_tokens":
                None,

            "completion_tokens":
                None,

            "ttft_ms":
                None,

            "tpot_ms":
                None,

            "e2e_ms":
                (
                    request_end
                    - request_start
                )
                * 1000.0,
        }

    if usage is None:

        return {
            "request_id":
                request_id,

            "success":
                False,

            "error":
                "No usage information.",

            "request_start":
                request_start,

            "request_end":
                request_end,

            "first_content":
                first_content,

            "last_content":
                last_content,

            "prompt_tokens":
                None,

            "completion_tokens":
                None,

            "ttft_ms":
                (
                    first_content
                    - request_start
                )
                * 1000.0,

            "tpot_ms":
                None,

            "e2e_ms":
                (
                    request_end
                    - request_start
                )
                * 1000.0,
        }

    ttft_ms = (
        first_content
        - request_start
    ) * 1000.0

    if (
        last_content is not None
        and
        COMPLETION_TOKENS > 1
    ):

        tpot_ms = (
            last_content
            - first_content
        ) / (
            COMPLETION_TOKENS
            - 1
        ) * 1000.0

    else:

        tpot_ms = None

    e2e_ms = (
        request_end
        - request_start
    ) * 1000.0

    token_valid = (
        usage.prompt_tokens
        == TOTAL_PROMPT_TOKENS
        and
        usage.completion_tokens
        == COMPLETION_TOKENS
    )

    return {
        "request_id":
            request_id,

        "success":
            token_valid,

        "error":
            (
                None
                if token_valid
                else (
                    "Token mismatch: "
                    f"prompt="
                    f"{usage.prompt_tokens}, "
                    f"completion="
                    f"{usage.completion_tokens}"
                )
            ),

        "request_start":
            request_start,

        "request_end":
            request_end,

        "first_content":
            first_content,

        "last_content":
            last_content,

        "prompt_tokens":
            usage.prompt_tokens,

        "completion_tokens":
            usage.completion_tokens,

        "ttft_ms":
            ttft_ms,

        "tpot_ms":
            tpot_ms,

        "e2e_ms":
            e2e_ms,
    }


# ============================================================
# 11. Run synchronized burst
# ============================================================

async def run_burst(
    client,
    model_name,
    prompts,
):

    concurrency = len(prompts)

    release_event = (
        asyncio.Event()
    )

    ready_events = [
        asyncio.Event()
        for _ in range(concurrency)
    ]

    tasks = []

    for request_id, prompt in enumerate(
        prompts,
        start=1,
    ):

        task = asyncio.create_task(
            run_burst_request(
                client,
                model_name,
                prompt,
                request_id,
                release_event,
                ready_events[
                    request_id - 1
                ],
            )
        )

        tasks.append(task)

    await asyncio.gather(
        *[
            event.wait()
            for event in ready_events
        ]
    )

    await asyncio.sleep(0)

    group_start = (
        time.perf_counter()
    )

    release_event.set()

    results = await asyncio.gather(
        *tasks
    )

    successful = [
        row
        for row in results
        if row["success"]
    ]

    if successful:

        group_end = max(
            row[
                "request_end"
            ]
            for row in successful
        )

        starts = [
            row[
                "request_start"
            ]
            for row in successful
        ]

        launch_skew_ms = (
            max(starts)
            - min(starts)
        ) * 1000.0

        makespan_s = (
            group_end
            - group_start
        )

    else:

        group_end = None
        launch_skew_ms = None
        makespan_s = None

    return {
        "group_start":
            group_start,

        "group_end":
            group_end,

        "launch_skew_ms":
            launch_skew_ms,

        "makespan_s":
            makespan_s,

        "results":
            results,
    }


# ============================================================
# 12. Run one formal burst condition
# ============================================================

async def run_formal_condition(
    client,
    model_name,
    block_id,
    order_index,
    mbt,
    reuse_condition,
    phase,
    rep,
    workload,
    fixed_rendered_lengths,
):

    concurrency = CONCURRENCY

    logical_namespace = formal_namespace(
        block_id,
        mbt,
        reuse_condition,
        phase,
        rep,
    )

    namespace = workload[
        "namespace"
    ]

    if (
        workload[
            "logical_namespace"
        ]
        != logical_namespace
    ):

        raise RuntimeError(
            "WORKLOAD LOGICAL NAMESPACE "
            "MISMATCH: "
            f"expected={logical_namespace}, "
            f"workload="
            f"{workload['logical_namespace']}"
        )

    # --------------------------------------------------------
    # Primer
    # --------------------------------------------------------

    before_primer = (
        await exp03.get_metrics_snapshot()
    )

    primer_result = await exp03.run_request(
        client,
        model_name,
        workload[
            "primer_text"
        ],
        PRIMER_COMPLETION_TOKENS,
    )

    await asyncio.sleep(
        METRIC_SETTLE_SEC
    )

    after_primer = (
        await exp03.get_metrics_snapshot()
    )

    primer_delta = (
        exp03.metric_delta(
            before_primer,
            after_primer,
        )
    )

    # --------------------------------------------------------
    # Measured burst
    # --------------------------------------------------------

    burst = await run_burst(
        client,
        model_name,
        workload[
            "measured_prompts"
        ],
    )

    await asyncio.sleep(
        METRIC_SETTLE_SEC
    )

    after_burst = (
        await exp03.get_metrics_snapshot()
    )

    group_delta = (
        exp03.metric_delta(
            after_primer,
            after_burst,
        )
    )

    request_results = (
        burst[
            "results"
        ]
    )

    successful = [
        row
        for row in request_results
        if row["success"]
    ]

    failed = [
        row
        for row in request_results
        if not row["success"]
    ]

    completed_requests = len(
        successful
    )

    # --------------------------------------------------------
    # Request metrics
    # --------------------------------------------------------

    if successful:

        ttft_values = [
            row[
                "ttft_ms"
            ]
            for row in successful
        ]

        tpot_values = [
            row[
                "tpot_ms"
            ]
            for row in successful
        ]

        e2e_values = [
            row[
                "e2e_ms"
            ]
            for row in successful
        ]

        mean_ttft_ms = mean_or_nan(
            ttft_values
        )

        p95_ttft_ms = percentile(
            ttft_values,
            0.95,
        )

        mean_tpot_ms = mean_or_nan(
            tpot_values
        )

        mean_e2e_ms = mean_or_nan(
            e2e_values
        )

    else:

        mean_ttft_ms = float("nan")
        p95_ttft_ms = float("nan")
        mean_tpot_ms = float("nan")
        mean_e2e_ms = float("nan")

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    makespan_s = (
        burst[
            "makespan_s"
        ]
    )

    if (
        makespan_s is not None
        and makespan_s > 0
    ):

        request_throughput = (
            completed_requests
            / makespan_s
        )

        total_output_tokens = sum(
            row[
                "completion_tokens"
            ]
            for row in successful
        )

        output_throughput = (
            total_output_tokens
            / makespan_s
        )

    else:

        request_throughput = float("nan")
        output_throughput = float("nan")
        total_output_tokens = 0

    # --------------------------------------------------------
    # Raw cache telemetry
    # --------------------------------------------------------

    group_queries = (
        group_delta[
            "prefix_queries"
        ]
    )

    group_hits = (
        group_delta[
            "prefix_hits"
        ]
    )

    group_compute = (
        group_delta[
            "local_compute"
        ]
    )

    group_local_hit = (
        group_delta[
            "local_cache_hit"
        ]
    )

    preemption_delta = (
        group_delta[
            "preemptions"
        ]
    )

    expected_queries = (
        concurrency
        * TOTAL_PROMPT_TOKENS
    )

    raw_group_hit_ratio = (
        group_hits
        / group_queries
        if group_queries
        else float("nan")
    )

    raw_hit_per_request = (
        group_hits
        / concurrency
    )

    compute_per_request = (
        group_compute
        / concurrency
    )

    # --------------------------------------------------------
    # Validity
    # --------------------------------------------------------

    all_requests_completed = (
        completed_requests
        == concurrency
    )

    prompt_tokens_valid = (
        all_requests_completed
        and all(
            row[
                "prompt_tokens"
            ]
            == TOTAL_PROMPT_TOKENS
            for row in successful
        )
    )

    completion_tokens_valid = (
        all_requests_completed
        and all(
            row[
                "completion_tokens"
            ]
            == COMPLETION_TOKENS
            for row in successful
        )
    )

    launch_skew_valid = (
        burst[
            "launch_skew_ms"
        ]
        is not None
        and
        burst[
            "launch_skew_ms"
        ]
        <= MAX_LAUNCH_SKEW_MS
    )

    query_accounting_valid = (
        group_queries
        == expected_queries
    )

    cache_arithmetic_valid = (
        abs(
            (
                group_hits
                + group_compute
            )
            - group_queries
        )
        < 1e-9
    )

    local_hit_valid = (
        group_local_hit
        == group_hits
    )

    # --------------------------------------------------------
    # Frozen rendered shared-prefix length
    # --------------------------------------------------------

    if reuse_condition == "P0":

        rendered_shared_valid = (
            workload[
                "rendered_shared_tokens"
            ]
            == 0
        )

    else:

        rendered_shared_valid = (
            workload[
                "rendered_shared_tokens"
            ]
            ==
            fixed_rendered_lengths[
                reuse_condition
            ]
        )

    # --------------------------------------------------------
    # Raw cache mechanism validity
    # --------------------------------------------------------

    lower, upper = (
        RAW_HIT_RATIO_RANGE[
            reuse_condition
        ]
    )

    reuse_mechanism_valid = (
        lower
        <= raw_group_hit_ratio
        <= upper
    )

    # --------------------------------------------------------
    # Primer state isolation
    #
    # Small engine/common-template reuse is tolerated.
    # Large inherited application-level reuse is not.
    # --------------------------------------------------------

    primer_isolation_valid = (
        primer_delta[
            "prefix_hits"
        ]
        <= 32
    )

    preemption_valid = (
        preemption_delta
        == 0
    )

    measurement_valid = all(
        [
            all_requests_completed,
            prompt_tokens_valid,
            completion_tokens_valid,
            launch_skew_valid,
            query_accounting_valid,
            cache_arithmetic_valid,
            local_hit_valid,
            rendered_shared_valid,
            reuse_mechanism_valid,
            primer_isolation_valid,
            preemption_valid,
        ]
    )

    burst_row = {
        "block_id":
            block_id,

        "order_index":
            order_index,

        "mbt":
            mbt,

        "reuse_condition":
            reuse_condition,

        "phase":
            phase,

        "rep":
            rep,

        "concurrency":
            concurrency,

        "logical_namespace":
            logical_namespace,

        "namespace":
            namespace,

        "shared_prefix_sha256":
            workload[
                "shared_prefix_sha256"
            ],

        "namespace_map_sha256":
            workload[
                "namespace_map_sha256"
            ],

        "formal_row_id":
            workload.get(
                "formal_row_id",
                "",
            ),

        "rendered_shared_tokens":
            workload[
                "rendered_shared_tokens"
            ],

        "nominal_target_used":
            workload[
                "nominal_target_used"
            ],

        # Primer
        "primer_ttft_ms":
            primer_result[
                "ttft_ms"
            ],

        "primer_prefix_hits":
            primer_delta[
                "prefix_hits"
            ],

        "primer_local_compute":
            primer_delta[
                "local_compute"
            ],

        # Requests
        "requested_requests":
            concurrency,

        "completed_requests":
            completed_requests,

        "failed_requests":
            len(failed),

        # Burst
        "launch_skew_ms":
            burst[
                "launch_skew_ms"
            ],

        "makespan_s":
            makespan_s,

        # Latency
        "mean_ttft_ms":
            mean_ttft_ms,

        "p95_ttft_ms":
            p95_ttft_ms,

        "mean_tpot_ms":
            mean_tpot_ms,

        "mean_e2e_ms":
            mean_e2e_ms,

        # Throughput
        "request_throughput_req_s":
            request_throughput,

        "output_throughput_tok_s":
            output_throughput,

        "total_output_tokens":
            total_output_tokens,

        # Cache
        "group_prefix_queries":
            group_queries,

        "group_prefix_hits":
            group_hits,

        "group_local_cache_hit":
            group_local_hit,

        "group_local_compute":
            group_compute,

        "raw_group_hit_ratio":
            raw_group_hit_ratio,

        "raw_hit_per_request":
            raw_hit_per_request,

        "compute_per_request":
            compute_per_request,

        # System
        "preemption_delta":
            preemption_delta,

        # Validity
        "all_requests_completed":
            all_requests_completed,

        "prompt_tokens_valid":
            prompt_tokens_valid,

        "completion_tokens_valid":
            completion_tokens_valid,

        "launch_skew_valid":
            launch_skew_valid,

        "query_accounting_valid":
            query_accounting_valid,

        "cache_arithmetic_valid":
            cache_arithmetic_valid,

        "local_hit_valid":
            local_hit_valid,

        "rendered_shared_valid":
            rendered_shared_valid,

        "reuse_mechanism_valid":
            reuse_mechanism_valid,

        "primer_isolation_valid":
            primer_isolation_valid,

        "preemption_valid":
            preemption_valid,

        "measurement_valid":
            measurement_valid,
    }

    request_rows = []

    for request_row in request_results:

        request_rows.append(
            {
                "block_id":
                    block_id,

                "order_index":
                    order_index,

                "mbt":
                    mbt,

                "reuse_condition":
                    reuse_condition,

                "phase":
                    phase,

                "rep":
                    rep,

                "concurrency":
                    concurrency,

                "logical_namespace":
                    logical_namespace,

                "namespace":
                    namespace,

                "shared_prefix_sha256":
                    workload[
                        "shared_prefix_sha256"
                    ],

                "namespace_map_sha256":
                    workload[
                        "namespace_map_sha256"
                    ],

                "formal_row_id":
                    workload.get(
                        "formal_row_id",
                        "",
                    ),

                "request_id":
                    request_row[
                        "request_id"
                    ],

                "success":
                    request_row[
                        "success"
                    ],

                "error":
                    request_row[
                        "error"
                    ],

                "prompt_tokens":
                    request_row[
                        "prompt_tokens"
                    ],

                "completion_tokens":
                    request_row[
                        "completion_tokens"
                    ],

                "ttft_ms":
                    request_row[
                        "ttft_ms"
                    ],

                "tpot_ms":
                    request_row[
                        "tpot_ms"
                    ],

                "e2e_ms":
                    request_row[
                        "e2e_ms"
                    ],
            }
        )

    return (
        burst_row,
        request_rows,
    )


# ============================================================
# 13. Preconditioning
# ============================================================

async def run_preconditioning(
    client,
    model_name,
):

    print()
    print(
        "=" * 118
    )

    print(
        "EXP04 FORMAL PRECONDITIONING"
    )

    print(
        "=" * 118
    )

    for round_id in range(
        1,
        PRECONDITION_ROUNDS + 1,
    ):

        print(
            f"Preconditioning round "
            f"{round_id}/"
            f"{PRECONDITION_ROUNDS}"
        )

        # ----------------------------------------------------
        # P0 C8
        # ----------------------------------------------------

        p0_namespace = (
            f"{PRECONDITION_NAMESPACE}_"
            f"R{round_id:02d}_P0_C8"
        )

        p0_workload = (
            await build_p0_workload(
                client,
                model_name,
                p0_namespace,
                8,
                round_id + 20,
            )
        )

        # Simple primer
        await exp03.run_request(
            client,
            model_name,
            p0_workload[
                "primer_text"
            ],
            PRIMER_COMPLETION_TOKENS,
        )

        burst = await run_burst(
            client,
            model_name,
            p0_workload[
                "measured_prompts"
            ],
        )

        p0_success = [
            r
            for r in burst[
                "results"
            ]
            if r["success"]
        ]

        if p0_success:

            p0_ttft = statistics.mean(
                r[
                    "ttft_ms"
                ]
                for r in p0_success
            )

            p0_out = (
                len(p0_success)
                * COMPLETION_TOKENS
                / burst[
                    "makespan_s"
                ]
            )

            print(
                f"  P0-C8    "
                f"TTFT={p0_ttft:.1f} ms, "
                f"out={p0_out:.1f} tok/s"
            )

        # ----------------------------------------------------
        # PHIGH C8
        #
        # This is preconditioning only. It does not use
        # formal namespaces or formal preflight prefixes.
        # ----------------------------------------------------

        pre_namespace = (
            f"{PRECONDITION_NAMESPACE}_"
            f"R{round_id:02d}_PHIGH_C8"
        )

        (
            shared_prefix,
            rendered_shared,
        ) = await exp03.build_shared_prefix(
            client,
            model_name,
            pre_namespace,
            "PHIGH",
            BASE_NOMINAL_TARGETS[
                "PHIGH"
            ],
        )

        phigh_workload = (
            await build_cached_workload(
                client,
                model_name,
                shared_prefix,
                rendered_shared,
                BASE_NOMINAL_TARGETS[
                    "PHIGH"
                ],
                8,
            )
        )

        await exp03.run_request(
            client,
            model_name,
            phigh_workload[
                "primer_text"
            ],
            PRIMER_COMPLETION_TOKENS,
        )

        burst = await run_burst(
            client,
            model_name,
            phigh_workload[
                "measured_prompts"
            ],
        )

        phigh_success = [
            r
            for r in burst[
                "results"
            ]
            if r["success"]
        ]

        if phigh_success:

            phigh_ttft = statistics.mean(
                r[
                    "ttft_ms"
                ]
                for r in phigh_success
            )

            phigh_out = (
                len(phigh_success)
                * COMPLETION_TOKENS
                / burst[
                    "makespan_s"
                ]
            )

            print(
                f"  PHIGH-C8 "
                f"TTFT={phigh_ttft:.1f} ms, "
                f"out={phigh_out:.1f} tok/s"
            )

    print(
        "Preconditioning complete."
    )


# ============================================================
# 14. Summary
# ============================================================

def build_condition_summary(
    burst_rows,
):

    grouped = defaultdict(list)

    for row in burst_rows:

        grouped[
            (
                row[
                    "reuse_condition"
                ],
                row[
                    "concurrency"
                ],
            )
        ].append(row)

    summary_rows = []

    for reuse_condition in (
        REUSE_CONDITIONS
    ):

        for concurrency in (
            CONCURRENCY_LEVELS
        ):

            rows = grouped[
                (
                    reuse_condition,
                    concurrency,
                )
            ]

            ttft = [
                row[
                    "mean_ttft_ms"
                ]
                for row in rows
            ]

            tpot = [
                row[
                    "mean_tpot_ms"
                ]
                for row in rows
            ]

            e2e = [
                row[
                    "mean_e2e_ms"
                ]
                for row in rows
            ]

            req_s = [
                row[
                    "request_throughput_req_s"
                ]
                for row in rows
            ]

            out_s = [
                row[
                    "output_throughput_tok_s"
                ]
                for row in rows
            ]

            hit_ratio = [
                row[
                    "raw_group_hit_ratio"
                ]
                for row in rows
            ]

            hit_per_req = [
                row[
                    "raw_hit_per_request"
                ]
                for row in rows
            ]

            compute_per_req = [
                row[
                    "compute_per_request"
                ]
                for row in rows
            ]

            summary_rows.append(
                {
                    "reuse_condition":
                        reuse_condition,

                    "concurrency":
                        concurrency,

                    "n_blocks":
                        len(rows),

                    "rendered_shared_tokens":
                        rows[0][
                            "rendered_shared_tokens"
                        ],

                    "raw_hit_ratio_mean":
                        statistics.mean(
                            hit_ratio
                        ),

                    "raw_hit_per_request_mean":
                        statistics.mean(
                            hit_per_req
                        ),

                    "compute_per_request_mean":
                        statistics.mean(
                            compute_per_req
                        ),

                    "ttft_mean_ms":
                        statistics.mean(
                            ttft
                        ),

                    "ttft_p50_ms":
                        percentile(
                            ttft,
                            0.50,
                        ),

                    "ttft_p95_ms":
                        percentile(
                            ttft,
                            0.95,
                        ),

                    "ttft_std_ms":
                        std_or_zero(
                            ttft
                        ),

                    "tpot_mean_ms":
                        statistics.mean(
                            tpot
                        ),

                    "e2e_mean_ms":
                        statistics.mean(
                            e2e
                        ),

                    "req_s_mean":
                        statistics.mean(
                            req_s
                        ),

                    "out_tok_s_mean":
                        statistics.mean(
                            out_s
                        ),
                }
            )

    return summary_rows


def build_interaction_summary(
    condition_summary,
):

    by_key = {
        (
            row[
                "reuse_condition"
            ],
            row[
                "concurrency"
            ],
        ):
        row
        for row in condition_summary
    }

    rows = []

    for concurrency in (
        CONCURRENCY_LEVELS
    ):

        p0 = by_key[
            (
                "P0",
                concurrency,
            )
        ]

        p50 = by_key[
            (
                "P50",
                concurrency,
            )
        ]

        phigh = by_key[
            (
                "PHIGH",
                concurrency,
            )
        ]

        def benefit_pct(
            reference,
            optimized,
        ):

            return (
                1.0
                - optimized
                / reference
            ) * 100.0

        def gain_pct(
            reference,
            optimized,
        ):

            return (
                optimized
                / reference
                - 1.0
            ) * 100.0

        rows.append(
            {
                "concurrency":
                    concurrency,

                "p0_ttft_ms":
                    p0[
                        "ttft_mean_ms"
                    ],

                "p50_ttft_ms":
                    p50[
                        "ttft_mean_ms"
                    ],

                "phigh_ttft_ms":
                    phigh[
                        "ttft_mean_ms"
                    ],

                "p50_ttft_benefit_vs_p0_pct":
                    benefit_pct(
                        p0[
                            "ttft_mean_ms"
                        ],
                        p50[
                            "ttft_mean_ms"
                        ],
                    ),

                "phigh_ttft_benefit_vs_p0_pct":
                    benefit_pct(
                        p0[
                            "ttft_mean_ms"
                        ],
                        phigh[
                            "ttft_mean_ms"
                        ],
                    ),

                "p50_out_gain_vs_p0_pct":
                    gain_pct(
                        p0[
                            "out_tok_s_mean"
                        ],
                        p50[
                            "out_tok_s_mean"
                        ],
                    ),

                "phigh_out_gain_vs_p0_pct":
                    gain_pct(
                        p0[
                            "out_tok_s_mean"
                        ],
                        phigh[
                            "out_tok_s_mean"
                        ],
                    ),

                "p50_tpot_change_vs_p0_pct":
                    gain_pct(
                        p0[
                            "tpot_mean_ms"
                        ],
                        p50[
                            "tpot_mean_ms"
                        ],
                    ),

                "phigh_tpot_change_vs_p0_pct":
                    gain_pct(
                        p0[
                            "tpot_mean_ms"
                        ],
                        phigh[
                            "tpot_mean_ms"
                        ],
                    ),
            }
        )

    return rows


# ============================================================
# 15. Main
# ============================================================

async def main():

    print(
        "=" * 118
    )

    print(
        "Exp04 Formal-v1: "
        "Prefix Reuse × Concurrency"
    )

    print(
        "=" * 118
    )

    print(
        f"Result directory : "
        f"{RESULT_DIR}"
    )

    print(
        f"Prompt tokens    : "
        f"{TOTAL_PROMPT_TOKENS}"
    )

    print(
        f"Completion       : "
        f"{COMPLETION_TOKENS}"
    )

    print(
        f"Reuse            : "
        f"{REUSE_CONDITIONS}"
    )

    print(
        f"Concurrency      : "
        f"{CONCURRENCY_LEVELS}"
    )

    print(
        f"Formal blocks    : "
        f"{NUM_BLOCKS}"
    )

    print(
        f"Formal bursts    : "
        f"{NUM_BLOCKS * len(FORMAL_CONDITIONS)}"
    )

    print(
        f"Random seed      : "
        f"{RANDOM_SEED}"
    )

    print(
        f"Launch skew max  : "
        f"{MAX_LAUNCH_SKEW_MS} ms"
    )

    async with AsyncOpenAI(
        base_url=API_BASE,
        api_key=API_KEY,
        http_client=
            DefaultAioHttpClient(),
    ) as client:

        model_name = (
            await exp03.exp02.get_model_name(
                client
            )
        )

        print(
            f"Model            : "
            f"{model_name}"
        )

        # ====================================================
        # PRECONDITION
        # ====================================================

        await run_preconditioning(
            client,
            model_name,
        )

        # ====================================================
        # PREFLIGHT
        #
        # Formal data collection MUST NOT start unless all
        # shared-prefix lengths can be held constant.
        # ====================================================

        (
            preflight_map,
            fixed_rendered_lengths,
        ) = await run_shared_prefix_preflight(
            client,
            model_name,
        )

        # Save preflight before formal collection.
        preflight_rows = []

        for (
            block_id,
            reuse_condition,
            concurrency,
        ), info in sorted(
            preflight_map.items()
        ):

            preflight_rows.append(
                {
                    "block_id":
                        block_id,

                    "reuse_condition":
                        reuse_condition,

                    "concurrency":
                        concurrency,

                    "rendered_shared_tokens":
                        info[
                            "rendered_shared_tokens"
                        ],

                    "nominal_target_used":
                        info[
                            "nominal_target_used"
                        ],
                }
            )

        write_csv(
            RESULT_DIR
            / "preflight_shared_prefix.csv",
            preflight_rows,
        )

        # ====================================================
        # Frozen randomized block order
        # ====================================================

        block_orders = {}

        for block_id in range(
            1,
            NUM_BLOCKS + 1,
        ):

            order = list(
                FORMAL_CONDITIONS
            )

            rng = random.Random(
                RANDOM_SEED
                + block_id
            )

            rng.shuffle(
                order
            )

            block_orders[
                block_id
            ] = order

        # Save order before data collection.
        order_rows = []

        for block_id, order in (
            block_orders.items()
        ):

            for index, (
                reuse_condition,
                concurrency,
            ) in enumerate(
                order,
                start=1,
            ):

                order_rows.append(
                    {
                        "block_id":
                            block_id,

                        "order_index":
                            index,

                        "reuse_condition":
                            reuse_condition,

                        "concurrency":
                            concurrency,
                    }
                )

        write_csv(
            RESULT_DIR
            / "formal_order.csv",
            order_rows,
        )

        print()
        print(
            "=" * 118
        )

        print(
            "FROZEN FORMAL BLOCK ORDER"
        )

        print(
            "=" * 118
        )

        for block_id in range(
            1,
            NUM_BLOCKS + 1,
        ):

            labels = [
                f"{reuse}-C{c}"
                for reuse, c
                in block_orders[
                    block_id
                ]
            ]

            print(
                f"B{block_id:02d}: "
                + " | ".join(
                    labels
                )
            )

        # ====================================================
        # Formal collection
        # ====================================================

        formal_start_metrics = (
            await exp03.get_metrics_snapshot()
        )

        burst_rows = []
        request_rows = []

        print()
        print(
            "=" * 118
        )

        print(
            "FORMAL COLLECTION START"
        )

        print(
            "=" * 118
        )

        for block_id in range(
            1,
            NUM_BLOCKS + 1,
        ):

            print()
            print(
                "#" * 118
            )

            print(
                f"BLOCK "
                f"{block_id}/"
                f"{NUM_BLOCKS}"
            )

            print(
                "#" * 118
            )

            order = (
                block_orders[
                    block_id
                ]
            )

            for order_index, (
                reuse_condition,
                concurrency,
            ) in enumerate(
                order,
                start=1,
            ):

                workload = (
                    await build_formal_workload(
                        client,
                        model_name,
                        block_id,
                        reuse_condition,
                        concurrency,
                        preflight_map,
                    )
                )

                (
                    burst_row,
                    req_rows,
                ) = await run_formal_condition(
                    client,
                    model_name,
                    block_id,
                    order_index,
                    reuse_condition,
                    concurrency,
                    workload,
                    fixed_rendered_lengths,
                )

                burst_rows.append(
                    burst_row
                )

                request_rows.extend(
                    req_rows
                )

                print(
                    f"B{block_id:02d} "
                    f"{order_index:02d}/12 "
                    f"{reuse_condition:<5} "
                    f"C{concurrency:<2} | "
                    f"Shared="
                    f"{burst_row['rendered_shared_tokens']:<4} "
                    f"Hit="
                    f"{burst_row['raw_group_hit_ratio'] * 100:6.2f}% "
                    f"TTFT="
                    f"{burst_row['mean_ttft_ms']:8.2f} "
                    f"TPOT="
                    f"{burst_row['mean_tpot_ms']:6.2f} "
                    f"Out="
                    f"{burst_row['output_throughput_tok_s']:7.2f} "
                    f"Skew="
                    f"{burst_row['launch_skew_ms']:6.2f} "
                    f"Preempt="
                    f"{burst_row['preemption_delta']:.0f} "
                    f"Valid="
                    f"{burst_row['measurement_valid']}"
                )

        formal_end_metrics = (
            await exp03.get_metrics_snapshot()
        )

        formal_metric_delta = (
            exp03.metric_delta(
                formal_start_metrics,
                formal_end_metrics,
            )
        )

    # ========================================================
    # Save raw formal data immediately
    # ========================================================

    write_csv(
        RESULT_DIR
        / "formal_bursts.csv",
        burst_rows,
    )

    write_csv(
        RESULT_DIR
        / "formal_requests.csv",
        request_rows,
    )

    # ========================================================
    # Block-level monotonic cache validation
    # ========================================================

    by_block_c = defaultdict(dict)

    for row in burst_rows:

        by_block_c[
            (
                row[
                    "block_id"
                ],
                row[
                    "concurrency"
                ],
            )
        ][
            row[
                "reuse_condition"
            ]
        ] = row

    monotonic_rows = []

    all_cache_monotonic = True

    for block_id in range(
        1,
        NUM_BLOCKS + 1,
    ):

        for concurrency in (
            CONCURRENCY_LEVELS
        ):

            data = by_block_c[
                (
                    block_id,
                    concurrency,
                )
            ]

            p0 = data[
                "P0"
            ][
                "raw_group_hit_ratio"
            ]

            p50 = data[
                "P50"
            ][
                "raw_group_hit_ratio"
            ]

            phigh = data[
                "PHIGH"
            ][
                "raw_group_hit_ratio"
            ]

            valid = (
                p0
                < p50
                < phigh
            )

            all_cache_monotonic = (
                all_cache_monotonic
                and valid
            )

            monotonic_rows.append(
                {
                    "block_id":
                        block_id,

                    "concurrency":
                        concurrency,

                    "p0_hit_ratio":
                        p0,

                    "p50_hit_ratio":
                        p50,

                    "phigh_hit_ratio":
                        phigh,

                    "monotonic":
                        valid,
                }
            )

    write_csv(
        RESULT_DIR
        / "cache_monotonicity.csv",
        monotonic_rows,
    )

    # ========================================================
    # Formal validity
    # ========================================================

    expected_bursts = (
        NUM_BLOCKS
        * len(
            FORMAL_CONDITIONS
        )
    )

    sample_count_valid = (
        len(burst_rows)
        == expected_bursts
    )

    all_measurements_valid = all(
        row[
            "measurement_valid"
        ]
        for row in burst_rows
    )

    all_launch_skew_valid = all(
        row[
            "launch_skew_valid"
        ]
        for row in burst_rows
    )

    all_requests_completed = all(
        row[
            "all_requests_completed"
        ]
        for row in burst_rows
    )

    all_cache_accounting_valid = all(
        row[
            "query_accounting_valid"
        ]
        and
        row[
            "cache_arithmetic_valid"
        ]
        and
        row[
            "local_hit_valid"
        ]
        for row in burst_rows
    )

    all_rendered_shared_valid = all(
        row[
            "rendered_shared_valid"
        ]
        for row in burst_rows
    )

    all_reuse_mechanism_valid = all(
        row[
            "reuse_mechanism_valid"
        ]
        for row in burst_rows
    )

    all_primer_isolation_valid = all(
        row[
            "primer_isolation_valid"
        ]
        for row in burst_rows
    )

    # --------------------------------------------------------
    # Preemption is intentionally NOT a hard validity rule.
    # --------------------------------------------------------

    formal_preemption_delta = (
        formal_metric_delta[
            "preemptions"
        ]
    )

    overall_valid = all(
        [
            sample_count_valid,
            all_measurements_valid,
            all_launch_skew_valid,
            all_requests_completed,
            all_cache_accounting_valid,
            all_rendered_shared_valid,
            all_reuse_mechanism_valid,
            all_primer_isolation_valid,
            all_cache_monotonic,
        ]
    )

    # ========================================================
    # Summary
    # ========================================================

    condition_summary = (
        build_condition_summary(
            burst_rows
        )
    )

    interaction_summary = (
        build_interaction_summary(
            condition_summary
        )
    )

    write_csv(
        RESULT_DIR
        / "condition_summary.csv",
        condition_summary,
    )

    write_csv(
        RESULT_DIR
        / "interaction_summary.csv",
        interaction_summary,
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print(
        "=" * 132
    )

    print(
        "EXP04 FORMAL CONDITION SUMMARY"
    )

    print(
        "=" * 132
    )

    print(
        f"{'Reuse':<7}"
        f"{'C':>4}"
        f"{'Shared':>9}"
        f"{'Hit%':>9}"
        f"{'Comp/req':>11}"
        f"{'TTFT':>11}"
        f"{'TTFTp95':>11}"
        f"{'TPOT':>10}"
        f"{'Req/s':>10}"
        f"{'Out tok/s':>12}"
    )

    print(
        "-" * 132
    )

    for row in (
        condition_summary
    ):

        print(
            f"{row['reuse_condition']:<7}"
            f"{row['concurrency']:>4}"
            f"{row['rendered_shared_tokens']:>9}"
            f"{row['raw_hit_ratio_mean'] * 100:>8.2f}%"
            f"{row['compute_per_request_mean']:>11.1f}"
            f"{row['ttft_mean_ms']:>11.3f}"
            f"{row['ttft_p95_ms']:>11.3f}"
            f"{row['tpot_mean_ms']:>10.3f}"
            f"{row['req_s_mean']:>10.3f}"
            f"{row['out_tok_s_mean']:>12.3f}"
        )

    print()
    print(
        "=" * 132
    )

    print(
        "PREFIX-REUSE × CONCURRENCY INTERACTION"
    )

    print(
        "=" * 132
    )

    print(
        f"{'C':>4}"
        f"{'P50 TTFT ben.':>17}"
        f"{'PHIGH TTFT ben.':>19}"
        f"{'P50 Out gain':>16}"
        f"{'PHIGH Out gain':>18}"
        f"{'P50 TPOT Δ':>14}"
        f"{'PHIGH TPOT Δ':>16}"
    )

    print(
        "-" * 104
    )

    for row in (
        interaction_summary
    ):

        print(
            f"{row['concurrency']:>4}"
            f"{row['p50_ttft_benefit_vs_p0_pct']:>16.2f}%"
            f"{row['phigh_ttft_benefit_vs_p0_pct']:>18.2f}%"
            f"{row['p50_out_gain_vs_p0_pct']:>15.2f}%"
            f"{row['phigh_out_gain_vs_p0_pct']:>17.2f}%"
            f"{row['p50_tpot_change_vs_p0_pct']:>13.2f}%"
            f"{row['phigh_tpot_change_vs_p0_pct']:>15.2f}%"
        )

    print()
    print(
        "=" * 132
    )

    print(
        "FORMAL VALIDITY"
    )

    print(
        "=" * 132
    )

    print(
        f"sample_count_valid            : "
        f"{sample_count_valid} "
        f"({len(burst_rows)}/"
        f"{expected_bursts})"
    )

    print(
        f"all_measurements_valid        : "
        f"{all_measurements_valid}"
    )

    print(
        f"all_launch_skew_valid         : "
        f"{all_launch_skew_valid}"
    )

    print(
        f"all_requests_completed        : "
        f"{all_requests_completed}"
    )

    print(
        f"all_cache_accounting_valid    : "
        f"{all_cache_accounting_valid}"
    )

    print(
        f"all_rendered_shared_valid     : "
        f"{all_rendered_shared_valid}"
    )

    print(
        f"all_reuse_mechanism_valid     : "
        f"{all_reuse_mechanism_valid}"
    )

    print(
        f"all_primer_isolation_valid    : "
        f"{all_primer_isolation_valid}"
    )

    print(
        f"all_cache_monotonic           : "
        f"{all_cache_monotonic}"
    )

    print(
        f"formal_preemption_delta       : "
        f"{formal_preemption_delta}"
    )

    print()

    print(
        "EXP04 FORMAL-v1 VALIDITY = "
        f"{'PASS' if overall_valid else 'FAIL'}"
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "experiment":
            (
                "Exp04 Formal-v1 "
                "Prefix Reuse x Concurrency"
            ),

        "timestamp":
            timestamp,

        "model":
            model_name,

        "protocol":
            {
                "prompt_tokens":
                    TOTAL_PROMPT_TOKENS,

                "completion_tokens":
                    COMPLETION_TOKENS,

                "primer_completion_tokens":
                    PRIMER_COMPLETION_TOKENS,

                "reuse_conditions":
                    REUSE_CONDITIONS,

                "concurrency_levels":
                    CONCURRENCY_LEVELS,

                "num_blocks":
                    NUM_BLOCKS,

                "random_seed":
                    RANDOM_SEED,

                "max_launch_skew_ms":
                    MAX_LAUNCH_SKEW_MS,

                "raw_hit_ratio_range":
                    RAW_HIT_RATIO_RANGE,

                "preemption_is_hard_validity":
                    False,
            },

        "fixed_rendered_shared_lengths":
            fixed_rendered_lengths,

        "formal_metric_delta":
            formal_metric_delta,

        "validity":
            {
                "sample_count_valid":
                    sample_count_valid,

                "all_measurements_valid":
                    all_measurements_valid,

                "all_launch_skew_valid":
                    all_launch_skew_valid,

                "all_requests_completed":
                    all_requests_completed,

                "all_cache_accounting_valid":
                    all_cache_accounting_valid,

                "all_rendered_shared_valid":
                    all_rendered_shared_valid,

                "all_reuse_mechanism_valid":
                    all_reuse_mechanism_valid,

                "all_primer_isolation_valid":
                    all_primer_isolation_valid,

                "all_cache_monotonic":
                    all_cache_monotonic,

                "formal_preemption_delta":
                    formal_preemption_delta,

                "overall_valid":
                    overall_valid,
            },
    }

    with open(
        RESULT_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Artifacts saved to: "
        f"{RESULT_DIR}"
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )