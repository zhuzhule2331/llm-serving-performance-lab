# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp03_formal.py
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
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import aiohttp
from openai import AsyncOpenAI, DefaultAioHttpClient

import exp02_formal as exp02


# ============================================================
# 1. Frozen Formal Configuration
# ============================================================

API_BASE = "http://127.0.0.1:8000/v1"
METRICS_URL = "http://127.0.0.1:8000/metrics"
API_KEY = "EMPTY"

TOTAL_PROMPT_TOKENS = 2048

PRIMER_COMPLETION_TOKENS = 1
MEASURED_COMPLETION_TOKENS = 128

TEMPERATURE = 0.0
TOP_P = 1.0

NUM_BLOCKS = 10
PRECONDITION_BLOCKS = 2

RANDOM_SEED = 20260928

FORMAL_NAMESPACE = "EXP03_FORMAL_V1"
PRECONDITION_NAMESPACE = "EXP03_PRECONDITION_V1"


# ============================================================
# 2. Conditions
# ============================================================

CONDITIONS = [
    {
        "name": "P0",
        "target_shared_tokens": 0,
        "min_hit_tokens": 0,
        "max_hit_tokens": 0,
    },
    {
        "name": "P25",
        "target_shared_tokens": 512,
        "min_hit_tokens": 448,
        "max_hit_tokens": 544,
    },
    {
        "name": "P50",
        "target_shared_tokens": 1024,
        "min_hit_tokens": 960,
        "max_hit_tokens": 1056,
    },
    {
        "name": "P75",
        "target_shared_tokens": 1536,
        "min_hit_tokens": 1472,
        "max_hit_tokens": 1568,
    },
    {
        "name": "PHIGH",
        "target_shared_tokens": 1984,
        "min_hit_tokens": 1920,
        "max_hit_tokens": 2032,
    },
]


# ============================================================
# 3. Text Units
# ============================================================

SHARED_UNIT = (
    "Reusable serving context contains stable instructions "
    "about inference behavior, output structure, reproducibility, "
    "measurement methodology, scheduler semantics, cache behavior, "
    "and technical response requirements. "
)

COARSE_PAD = (
    " additional request specific technical context"
)

FINE_PAD_CANDIDATES = [
    " x",
    " a",
    " z",
    " 1",
    ".",
]


# ============================================================
# 4. Metrics Helpers
# ============================================================

async def fetch_metrics():
    timeout = aiohttp.ClientTimeout(
        total=5
    )

    async with aiohttp.ClientSession(
        timeout=timeout
    ) as session:

        async with session.get(
            METRICS_URL
        ) as response:

            response.raise_for_status()

            return await response.text()


def extract_simple_metric(
    text,
    metric_name,
):
    pattern = re.compile(
        rf"^{re.escape(metric_name)}"
        rf"\{{[^}}]*\}}\s+"
        rf"([-+0-9.eE]+)$",
        re.MULTILINE,
    )

    match = pattern.search(text)

    if match is None:
        return None

    return float(
        match.group(1)
    )


def extract_source_metric(
    text,
    source,
):
    pattern = re.compile(
        r'^vllm:prompt_tokens_by_source_total'
        r'\{[^}]*'
        rf'source="{re.escape(source)}"'
        r'[^}]*\}\s+'
        r'([-+0-9.eE]+)$',
        re.MULTILINE,
    )

    match = pattern.search(text)

    if match is None:
        return None

    return float(
        match.group(1)
    )


async def get_metrics_snapshot():
    text = await fetch_metrics()

    return {
        "prefix_queries":
            extract_simple_metric(
                text,
                "vllm:prefix_cache_queries_total",
            ),

        "prefix_hits":
            extract_simple_metric(
                text,
                "vllm:prefix_cache_hits_total",
            ),

        "prompt_cached":
            extract_simple_metric(
                text,
                "vllm:prompt_tokens_cached_total",
            ),

        "local_compute":
            extract_source_metric(
                text,
                "local_compute",
            ),

        "local_cache_hit":
            extract_source_metric(
                text,
                "local_cache_hit",
            ),

        "preemptions":
            extract_simple_metric(
                text,
                "vllm:num_preemptions_total",
            ),
    }


def metric_delta(
    before,
    after,
):
    output = {}

    for key in before:

        a = before[key]
        b = after[key]

        if (
            a is None
            or b is None
        ):
            output[key] = None
        else:
            output[key] = (
                b - a
            )

    return output


# ============================================================
# 5. Render Helper
# ============================================================

async def rendered_tokens(
    client,
    model_name,
    text,
):
    return await exp02.measure_prompt_tokens(
        client,
        model_name,
        text,
    )


# ============================================================
# 6. Exact Prompt Fitting
# ============================================================

async def fit_exact_prompt(
    client,
    model_name,
    base_text,
):
    """
    Extend base_text until the actual rendered prompt
    is exactly 2048 tokens.
    """

    base_tokens = await rendered_tokens(
        client,
        model_name,
        base_text,
    )

    if (
        base_tokens
        > TOTAL_PROMPT_TOKENS
    ):
        raise RuntimeError(
            f"Base prompt too long: "
            f"{base_tokens}"
        )

    if (
        base_tokens
        == TOTAL_PROMPT_TOKENS
    ):
        return (
            base_text,
            base_tokens,
        )

    # --------------------------------------------------------
    # Coarse padding
    # --------------------------------------------------------

    best_text = base_text
    best_tokens = base_tokens

    for coarse_count in range(
        0,
        500,
    ):

        candidate = (
            base_text
            + COARSE_PAD * coarse_count
        )

        n = await rendered_tokens(
            client,
            model_name,
            candidate,
        )

        if (
            n
            == TOTAL_PROMPT_TOKENS
        ):
            return (
                candidate,
                n,
            )

        if (
            n
            < TOTAL_PROMPT_TOKENS
        ):
            if (
                n
                > best_tokens
            ):
                best_text = candidate
                best_tokens = n

        else:
            break

    remaining = (
        TOTAL_PROMPT_TOKENS
        - best_tokens
    )

    # --------------------------------------------------------
    # Fine padding
    # --------------------------------------------------------

    for fine_unit in (
        FINE_PAD_CANDIDATES
    ):

        for fine_count in range(
            0,
            remaining + 100,
        ):

            candidate = (
                best_text
                + fine_unit * fine_count
            )

            n = await rendered_tokens(
                client,
                model_name,
                candidate,
            )

            if (
                n
                == TOTAL_PROMPT_TOKENS
            ):
                return (
                    candidate,
                    n,
                )

            if (
                n
                > TOTAL_PROMPT_TOKENS
                + 4
            ):
                break

    # --------------------------------------------------------
    # Mixed micro-padding fallback
    # --------------------------------------------------------

    micro_units = [
        " x",
        " a",
        " 1",
        ".",
    ]

    for unit_a in micro_units:

        for count_a in range(
            0,
            50,
        ):

            intermediate = (
                best_text
                + unit_a * count_a
            )

            n_a = await rendered_tokens(
                client,
                model_name,
                intermediate,
            )

            if (
                n_a
                > TOTAL_PROMPT_TOKENS
            ):
                break

            for unit_b in micro_units:

                for count_b in range(
                    0,
                    50,
                ):

                    candidate = (
                        intermediate
                        + unit_b * count_b
                    )

                    n = await rendered_tokens(
                        client,
                        model_name,
                        candidate,
                    )

                    if (
                        n
                        == TOTAL_PROMPT_TOKENS
                    ):
                        return (
                            candidate,
                            n,
                        )

                    if (
                        n
                        > TOTAL_PROMPT_TOKENS
                    ):
                        break

    raise RuntimeError(
        "Could not construct exact "
        f"{TOTAL_PROMPT_TOKENS}-token prompt."
    )


# ============================================================
# 7. Shared Prefix Builder
# ============================================================

async def build_shared_prefix(
    client,
    model_name,
    namespace,
    condition_name,
    target_shared_tokens,
):
    if (
        target_shared_tokens
        == 0
    ):
        return (
            "",
            0,
        )

    base = (
        f"{namespace} "
        f"{condition_name} "
        f"isolated shared namespace.\n"
        f"Shared reusable context begins.\n"
    )

    base_tokens = await rendered_tokens(
        client,
        model_name,
        base,
    )

    one_unit_tokens = await rendered_tokens(
        client,
        model_name,
        base + SHARED_UNIT,
    )

    increment = (
        one_unit_tokens
        - base_tokens
    )

    if increment <= 0:
        raise RuntimeError(
            "Invalid shared-unit token increment."
        )

    estimated_repeats = max(
        0,
        int(
            (
                target_shared_tokens
                - base_tokens
            )
            / increment
        ),
    )

    best_text = base
    best_tokens = base_tokens

    for repeat_count in range(
        max(
            0,
            estimated_repeats - 10,
        ),
        estimated_repeats + 12,
    ):

        candidate = (
            base
            + SHARED_UNIT * repeat_count
        )

        n = await rendered_tokens(
            client,
            model_name,
            candidate,
        )

        if (
            n <= target_shared_tokens
            and n > best_tokens
        ):
            best_text = candidate
            best_tokens = n

    return (
        best_text,
        best_tokens,
    )


# ============================================================
# 8. Build Primer / Measured Pair
# ============================================================

async def build_condition_pair(
    client,
    model_name,
    namespace,
    condition,
):
    name = condition["name"]

    target_shared_tokens = (
        condition[
            "target_shared_tokens"
        ]
    )

    if (
        target_shared_tokens
        == 0
    ):

        primer_base = (
            f"{namespace} "
            f"{name} "
            f"PRIMER-NONSHARED-BEGIN.\n"
            f"This primer deliberately has a "
            f"different application-level prefix "
            f"from its measured request.\n"
        )

        measured_base = (
            f"{namespace} "
            f"{name} "
            f"MEASURED-NONSHARED-BEGIN.\n"
            f"This measured request deliberately "
            f"has a different application-level "
            f"prefix from its primer.\n"
        )

        primer_text, primer_tokens = (
            await fit_exact_prompt(
                client,
                model_name,
                primer_base,
            )
        )

        measured_text, measured_tokens = (
            await fit_exact_prompt(
                client,
                model_name,
                measured_base,
            )
        )

        return {
            "shared_render_tokens": 0,

            "primer_text":
                primer_text,

            "primer_tokens":
                primer_tokens,

            "measured_text":
                measured_text,

            "measured_tokens":
                measured_tokens,
        }

    # --------------------------------------------------------
    # Reuse conditions
    # --------------------------------------------------------

    (
        shared_prefix,
        shared_render_tokens,
    ) = await build_shared_prefix(
        client,
        model_name,
        namespace,
        name,
        target_shared_tokens,
    )

    primer_base = (
        shared_prefix
        + "\n"
        + "Primer-specific suffix begins here. "
        + "This suffix exists only to populate "
        + "the intended shared-prefix cache state. "
    )

    measured_base = (
        shared_prefix
        + "\n"
        + "Measured-request suffix begins here. "
        + "This suffix is unique to the measured "
        + "request and differs from the primer. "
    )

    primer_text, primer_tokens = (
        await fit_exact_prompt(
            client,
            model_name,
            primer_base,
        )
    )

    measured_text, measured_tokens = (
        await fit_exact_prompt(
            client,
            model_name,
            measured_base,
        )
    )

    return {
        "shared_render_tokens":
            shared_render_tokens,

        "primer_text":
            primer_text,

        "primer_tokens":
            primer_tokens,

        "measured_text":
            measured_text,

        "measured_tokens":
            measured_tokens,
    }


# ============================================================
# 9. Prompt Hash
# ============================================================

def text_sha256(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 10. Run Request
# ============================================================

async def run_request(
    client,
    model_name,
    prompt_text,
    completion_tokens,
):
    start = time.perf_counter()

    first_content = None
    last_content = None
    usage = None

    async with client.chat.completions.stream(
        model=model_name,

        messages=[
            {
                "role": "user",
                "content": prompt_text,
            }
        ],

        temperature=TEMPERATURE,
        top_p=TOP_P,

        max_tokens=
            completion_tokens,

        extra_body={
            "min_tokens":
                completion_tokens,

            "ignore_eos":
                True,
        },

        stream_options={
            "include_usage": True,
        },

    ) as stream:

        async for event in stream:

            if (
                event.type
                == "content.delta"
            ):

                content = event.delta

                if content:

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

    end = time.perf_counter()

    if first_content is None:
        raise RuntimeError(
            "No non-empty content event."
        )

    if usage is None:
        raise RuntimeError(
            "No usage information."
        )

    if (
        usage.prompt_tokens
        != TOTAL_PROMPT_TOKENS
    ):
        raise RuntimeError(
            f"prompt_tokens="
            f"{usage.prompt_tokens}, "
            f"expected="
            f"{TOTAL_PROMPT_TOKENS}"
        )

    if (
        usage.completion_tokens
        != completion_tokens
    ):
        raise RuntimeError(
            f"completion_tokens="
            f"{usage.completion_tokens}, "
            f"expected="
            f"{completion_tokens}"
        )

    ttft_ms = (
        first_content
        - start
    ) * 1000.0

    e2e_ms = (
        end
        - start
    ) * 1000.0

    if (
        completion_tokens > 1
        and last_content is not None
    ):

        tpot_ms = (
            last_content
            - first_content
        ) / (
            completion_tokens - 1
        ) * 1000.0

    else:
        tpot_ms = None

    return {
        "ttft_ms":
            ttft_ms,

        "tpot_ms":
            tpot_ms,

        "e2e_ms":
            e2e_ms,

        "prompt_tokens":
            usage.prompt_tokens,

        "completion_tokens":
            usage.completion_tokens,
    }


# ============================================================
# 11. Percentile
# ============================================================

def percentile(
    values,
    p,
):
    xs = sorted(values)

    if not xs:
        return float("nan")

    if len(xs) == 1:
        return xs[0]

    position = (
        len(xs) - 1
    ) * p

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return xs[lower]

    fraction = (
        position - lower
    )

    return (
        xs[lower]
        + (
            xs[upper]
            - xs[lower]
        )
        * fraction
    )


# ============================================================
# 12. CSV Helper
# ============================================================

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
            fieldnames=
                list(
                    rows[0].keys()
                ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 13. Preconditioning
# ============================================================

async def run_preconditioning(
    client,
    model_name,
):
    print()
    print("=" * 110)
    print(
        "PRECONDITIONING"
    )
    print("=" * 110)

    for block_id in range(
        1,
        PRECONDITION_BLOCKS + 1,
    ):

        print(
            f"Preconditioning block "
            f"{block_id}/"
            f"{PRECONDITION_BLOCKS}"
        )

        for condition in CONDITIONS:

            name = condition["name"]

            namespace = (
                f"{PRECONDITION_NAMESPACE}_"
                f"B{block_id:02d}_"
                f"{name}"
            )

            pair = (
                await build_condition_pair(
                    client,
                    model_name,
                    namespace,
                    condition,
                )
            )

            await run_request(
                client,
                model_name,
                pair["primer_text"],
                PRIMER_COMPLETION_TOKENS,
            )

            result = await run_request(
                client,
                model_name,
                pair["measured_text"],
                MEASURED_COMPLETION_TOKENS,
            )

            print(
                f"  {name:<6} "
                f"TTFT="
                f"{result['ttft_ms']:.1f} ms"
            )

    print(
        "Preconditioning complete."
    )


# ============================================================
# 14. Formal Main
# ============================================================

async def main():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    result_dir = Path(
        f"exp03_formal_{timestamp}"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 110)
    print(
        "Exp03 Formal Benchmark"
    )
    print("=" * 110)

    print(
        f"Result directory : "
        f"{result_dir}"
    )

    print(
        f"Blocks           : "
        f"{NUM_BLOCKS}"
    )

    print(
        f"Conditions       : "
        f"{[x['name'] for x in CONDITIONS]}"
    )

    print(
        f"Random seed      : "
        f"{RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Fixed randomized block order
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    block_orders = []

    condition_names = [
        x["name"]
        for x in CONDITIONS
    ]

    for block_id in range(
        1,
        NUM_BLOCKS + 1,
    ):

        order = (
            condition_names.copy()
        )

        rng.shuffle(order)

        block_orders.append(
            {
                "block_id":
                    block_id,

                "order":
                    order,
            }
        )

    print()
    print("Frozen block order:")

    for item in block_orders:

        print(
            f"B{item['block_id']:02d}: "
            + " -> ".join(
                item["order"]
            )
        )

    condition_map = {
        x["name"]: x
        for x in CONDITIONS
    }

    async with AsyncOpenAI(
        base_url=API_BASE,
        api_key=API_KEY,
        http_client=
            DefaultAioHttpClient(),
    ) as client:

        model_name = (
            await exp02.get_model_name(
                client
            )
        )

        print()
        print(
            f"Model = {model_name}"
        )

        # ----------------------------------------------------
        # Preconditioning
        # ----------------------------------------------------

        await run_preconditioning(
            client,
            model_name,
        )

        # ----------------------------------------------------
        # Metrics before formal
        # ----------------------------------------------------

        formal_start_metrics = (
            await get_metrics_snapshot()
        )

        formal_rows = []

        sample_index = 0

        print()
        print("=" * 110)
        print(
            "FORMAL COLLECTION"
        )
        print("=" * 110)

        for block_item in block_orders:

            block_id = (
                block_item["block_id"]
            )

            print()
            print(
                f"--- Formal Block "
                f"{block_id:02d} ---"
            )

            block_hit_sequence = []

            for order_index, name in enumerate(
                block_item["order"],
                start=1,
            ):

                sample_index += 1

                condition = (
                    condition_map[name]
                )

                namespace = (
                    f"{FORMAL_NAMESPACE}_"
                    f"B{block_id:02d}_"
                    f"{name}"
                )

                pair = (
                    await build_condition_pair(
                        client,
                        model_name,
                        namespace,
                        condition,
                    )
                )

                # --------------------------------------------
                # Snapshot before primer
                # --------------------------------------------

                before_primer = (
                    await get_metrics_snapshot()
                )

                # --------------------------------------------
                # Primer
                # --------------------------------------------

                primer_result = (
                    await run_request(
                        client,
                        model_name,
                        pair["primer_text"],
                        PRIMER_COMPLETION_TOKENS,
                    )
                )

                after_primer = (
                    await get_metrics_snapshot()
                )

                primer_delta = (
                    metric_delta(
                        before_primer,
                        after_primer,
                    )
                )

                # --------------------------------------------
                # Measured request
                # --------------------------------------------

                measured_result = (
                    await run_request(
                        client,
                        model_name,
                        pair["measured_text"],
                        MEASURED_COMPLETION_TOKENS,
                    )
                )

                after_measured = (
                    await get_metrics_snapshot()
                )

                measured_delta = (
                    metric_delta(
                        after_primer,
                        after_measured,
                    )
                )

                hit_tokens = (
                    measured_delta[
                        "prefix_hits"
                    ]
                )

                compute_tokens = (
                    measured_delta[
                        "local_compute"
                    ]
                )

                local_hit_tokens = (
                    measured_delta[
                        "local_cache_hit"
                    ]
                )

                query_tokens = (
                    measured_delta[
                        "prefix_queries"
                    ]
                )

                if hit_tokens is None:
                    hit_ratio = None
                else:
                    hit_ratio = (
                        hit_tokens
                        / TOTAL_PROMPT_TOKENS
                    )

                # --------------------------------------------
                # Validity
                # --------------------------------------------

                primer_cold_valid = (
                    primer_delta[
                        "prefix_hits"
                    ]
                    == 0
                )

                primer_compute_valid = (
                    primer_delta[
                        "local_compute"
                    ]
                    == TOTAL_PROMPT_TOKENS
                )

                measured_query_valid = (
                    query_tokens
                    == TOTAL_PROMPT_TOKENS
                )

                cache_arithmetic_valid = (
                    hit_tokens is not None
                    and compute_tokens is not None
                    and query_tokens is not None
                    and abs(
                        (
                            hit_tokens
                            + compute_tokens
                        )
                        - query_tokens
                    )
                    < 1e-6
                )

                local_hit_valid = (
                    local_hit_tokens
                    == hit_tokens
                )

                hit_range_valid = (
                    hit_tokens is not None
                    and
                    condition[
                        "min_hit_tokens"
                    ]
                    <= hit_tokens
                    <=
                    condition[
                        "max_hit_tokens"
                    ]
                )

                sample_valid = all(
                    [
                        primer_cold_valid,
                        primer_compute_valid,
                        measured_query_valid,
                        cache_arithmetic_valid,
                        local_hit_valid,
                        hit_range_valid,
                    ]
                )

                block_hit_sequence.append(
                    (
                        name,
                        hit_tokens,
                    )
                )

                row = {
                    "sample_index":
                        sample_index,

                    "block_id":
                        block_id,

                    "order_index":
                        order_index,

                    "condition":
                        name,

                    "target_shared_tokens":
                        condition[
                            "target_shared_tokens"
                        ],

                    "rendered_shared_tokens":
                        pair[
                            "shared_render_tokens"
                        ],

                    "prompt_tokens":
                        measured_result[
                            "prompt_tokens"
                        ],

                    "completion_tokens":
                        measured_result[
                            "completion_tokens"
                        ],

                    "primer_prompt_tokens":
                        primer_result[
                            "prompt_tokens"
                        ],

                    "primer_completion_tokens":
                        primer_result[
                            "completion_tokens"
                        ],

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

                    "prefix_queries":
                        query_tokens,

                    "prefix_hits":
                        hit_tokens,

                    "local_cache_hit":
                        local_hit_tokens,

                    "local_compute":
                        compute_tokens,

                    "observed_hit_ratio":
                        hit_ratio,

                    "ttft_ms":
                        measured_result[
                            "ttft_ms"
                        ],

                    "tpot_ms":
                        measured_result[
                            "tpot_ms"
                        ],

                    "e2e_ms":
                        measured_result[
                            "e2e_ms"
                        ],

                    "primer_cold_valid":
                        primer_cold_valid,

                    "primer_compute_valid":
                        primer_compute_valid,

                    "measured_query_valid":
                        measured_query_valid,

                    "cache_arithmetic_valid":
                        cache_arithmetic_valid,

                    "local_hit_valid":
                        local_hit_valid,

                    "hit_range_valid":
                        hit_range_valid,

                    "sample_valid":
                        sample_valid,

                    "primer_sha256":
                        text_sha256(
                            pair["primer_text"]
                        ),

                    "measured_sha256":
                        text_sha256(
                            pair["measured_text"]
                        ),
                }

                formal_rows.append(row)

                print(
                    f"[{sample_index:02d}/50] "
                    f"B{block_id:02d} "
                    f"{name:<6} "
                    f"hit="
                    f"{hit_tokens:>6.0f} "
                    f"({hit_ratio * 100:>6.2f}%) "
                    f"compute="
                    f"{compute_tokens:>6.0f} "
                    f"TTFT="
                    f"{measured_result['ttft_ms']:>8.3f} "
                    f"TPOT="
                    f"{measured_result['tpot_ms']:>7.3f} "
                    f"valid="
                    f"{sample_valid}"
                )

            # --------------------------------------------
            # Block-level monotonicity
            #
            # Reconstruct by condition order, not
            # randomized execution order.
            # --------------------------------------------

            hit_map = {
                name: hit
                for name, hit
                in block_hit_sequence
            }

            ordered_hits = [
                hit_map["P0"],
                hit_map["P25"],
                hit_map["P50"],
                hit_map["P75"],
                hit_map["PHIGH"],
            ]

            monotonic = all(
                ordered_hits[i]
                < ordered_hits[i + 1]
                for i in range(
                    len(ordered_hits) - 1
                )
            )

            print(
                f"Block {block_id:02d} "
                f"hit monotonicity = "
                f"{monotonic}"
            )

        # ----------------------------------------------------
        # Metrics after formal
        # ----------------------------------------------------

        formal_end_metrics = (
            await get_metrics_snapshot()
        )

        formal_metric_delta = (
            metric_delta(
                formal_start_metrics,
                formal_end_metrics,
            )
        )

    # ========================================================
    # 15. Save raw formal CSV
    # ========================================================

    formal_csv = (
        result_dir
        / "formal_samples.csv"
    )

    write_csv(
        formal_csv,
        formal_rows,
    )

    # ========================================================
    # 16. Condition Summary
    # ========================================================

    summary_rows = []

    for condition in CONDITIONS:

        name = condition["name"]

        rows = [
            row
            for row in formal_rows
            if row["condition"] == name
        ]

        hit_values = [
            row["prefix_hits"]
            for row in rows
        ]

        ratio_values = [
            row["observed_hit_ratio"]
            for row in rows
        ]

        ttft_values = [
            row["ttft_ms"]
            for row in rows
        ]

        tpot_values = [
            row["tpot_ms"]
            for row in rows
        ]

        e2e_values = [
            row["e2e_ms"]
            for row in rows
        ]

        ttft_mean = (
            statistics.mean(
                ttft_values
            )
        )

        ttft_std = (
            statistics.stdev(
                ttft_values
            )
            if len(ttft_values) > 1
            else 0.0
        )

        ttft_cv = (
            ttft_std
            / ttft_mean
            if ttft_mean != 0
            else float("nan")
        )

        summary_rows.append(
            {
                "condition":
                    name,

                "n":
                    len(rows),

                "target_shared_tokens":
                    condition[
                        "target_shared_tokens"
                    ],

                "mean_hit_tokens":
                    statistics.mean(
                        hit_values
                    ),

                "mean_hit_ratio":
                    statistics.mean(
                        ratio_values
                    ),

                "mean_compute_tokens":
                    statistics.mean(
                        [
                            row[
                                "local_compute"
                            ]
                            for row in rows
                        ]
                    ),

                "ttft_mean_ms":
                    ttft_mean,

                "ttft_p50_ms":
                    percentile(
                        ttft_values,
                        0.50,
                    ),

                "ttft_p95_ms":
                    percentile(
                        ttft_values,
                        0.95,
                    ),

                "ttft_std_ms":
                    ttft_std,

                "ttft_cv":
                    ttft_cv,

                "tpot_mean_ms":
                    statistics.mean(
                        tpot_values
                    ),

                "tpot_p95_ms":
                    percentile(
                        tpot_values,
                        0.95,
                    ),

                "e2e_mean_ms":
                    statistics.mean(
                        e2e_values
                    ),

                "e2e_p95_ms":
                    percentile(
                        e2e_values,
                        0.95,
                    ),
            }
        )

    summary_csv = (
        result_dir
        / "condition_summary.csv"
    )

    write_csv(
        summary_csv,
        summary_rows,
    )

    # ========================================================
    # 17. Block-level hit monotonicity
    # ========================================================

    block_validity = []

    for block_id in range(
        1,
        NUM_BLOCKS + 1,
    ):

        rows = [
            row
            for row in formal_rows
            if row["block_id"] == block_id
        ]

        hit_map = {
            row["condition"]:
                row["prefix_hits"]
            for row in rows
        }

        ordered_hits = [
            hit_map["P0"],
            hit_map["P25"],
            hit_map["P50"],
            hit_map["P75"],
            hit_map["PHIGH"],
        ]

        monotonic = all(
            ordered_hits[i]
            < ordered_hits[i + 1]
            for i in range(
                len(ordered_hits) - 1
            )
        )

        block_validity.append(
            {
                "block_id":
                    block_id,

                "P0":
                    hit_map["P0"],

                "P25":
                    hit_map["P25"],

                "P50":
                    hit_map["P50"],

                "P75":
                    hit_map["P75"],

                "PHIGH":
                    hit_map["PHIGH"],

                "strictly_monotonic":
                    monotonic,
            }
        )

    write_csv(
        result_dir
        / "block_validity.csv",
        block_validity,
    )

    # ========================================================
    # 18. Formal Validity
    # ========================================================

    request_count_valid = (
        len(formal_rows)
        == NUM_BLOCKS
        * len(CONDITIONS)
    )

    samples_valid = all(
        row["sample_valid"]
        for row in formal_rows
    )

    block_monotonic_valid = all(
        row["strictly_monotonic"]
        for row in block_validity
    )

    preemption_delta = (
        formal_metric_delta[
            "preemptions"
        ]
    )

    no_preemption_valid = (
        preemption_delta == 0
    )

    overall_valid = all(
        [
            request_count_valid,
            samples_valid,
            block_monotonic_valid,
            no_preemption_valid,
        ]
    )

    validity = {
        "request_count_valid":
            request_count_valid,

        "samples_valid":
            samples_valid,

        "block_monotonic_valid":
            block_monotonic_valid,

        "no_formal_preemption":
            no_preemption_valid,

        "formal_preemption_delta":
            preemption_delta,

        "overall_valid":
            overall_valid,
    }

    # ========================================================
    # 19. Metadata
    # ========================================================

    metadata = {
        "experiment":
            "Exp03 Prefix Reuse Scaling",

        "timestamp":
            timestamp,

        "model":
            model_name,

        "total_prompt_tokens":
            TOTAL_PROMPT_TOKENS,

        "primer_completion_tokens":
            PRIMER_COMPLETION_TOKENS,

        "measured_completion_tokens":
            MEASURED_COMPLETION_TOKENS,

        "num_blocks":
            NUM_BLOCKS,

        "precondition_blocks":
            PRECONDITION_BLOCKS,

        "random_seed":
            RANDOM_SEED,

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "conditions":
            CONDITIONS,

        "block_orders":
            block_orders,

        "formal_metric_delta":
            formal_metric_delta,

        "validity":
            validity,
    }

    with open(
        result_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # 20. Console Summary
    # ========================================================

    print()
    print("=" * 110)
    print(
        "FORMAL CONDITION SUMMARY"
    )
    print("=" * 110)

    print(
        f"{'Cond':<7}"
        f"{'N':>4}"
        f"{'HitTok':>10}"
        f"{'Hit%':>9}"
        f"{'Compute':>10}"
        f"{'TTFT mean':>13}"
        f"{'TTFT p95':>12}"
        f"{'TPOT':>10}"
        f"{'E2E':>11}"
    )

    print("-" * 110)

    for row in summary_rows:

        print(
            f"{row['condition']:<7}"
            f"{row['n']:>4}"
            f"{row['mean_hit_tokens']:>10.1f}"
            f"{row['mean_hit_ratio'] * 100:>8.2f}%"
            f"{row['mean_compute_tokens']:>10.1f}"
            f"{row['ttft_mean_ms']:>13.3f}"
            f"{row['ttft_p95_ms']:>12.3f}"
            f"{row['tpot_mean_ms']:>10.3f}"
            f"{row['e2e_mean_ms']:>11.3f}"
        )

    print()
    print("=" * 110)
    print(
        "FORMAL VALIDITY"
    )
    print("=" * 110)

    for key, value in validity.items():
        print(
            f"{key:<30}: {value}"
        )

    print()

    if overall_valid:
        print(
            "EXP03 FORMAL VALIDITY = PASS"
        )
    else:
        print(
            "EXP03 FORMAL VALIDITY = FAIL"
        )

    print()
    print(
        f"Artifacts saved to: "
        f"{result_dir}"
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )