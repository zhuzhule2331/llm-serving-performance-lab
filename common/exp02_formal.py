# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp02_formal.py
# Copied  : verbatim except for the changes listed in ../../PUBLIC_CODE_CHANGELOG.md
# Frozen  : this file is the FINAL public-facing implementation used in the
#           formal campaign; it is a copy, the original is unmodified.
# =============================================================================
import asyncio
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

import aiohttp
from openai import AsyncOpenAI, DefaultAioHttpClient


# ============================================================
# 1. Formal configuration
# ============================================================

API_BASE = "http://127.0.0.1:8000/v1"
METRICS_URL = "http://127.0.0.1:8000/metrics"
API_KEY = "EMPTY"

TARGET_PROMPT_TOKENS = 2048
TARGET_COMPLETION_TOKENS = 128

MAX_MODEL_LEN = 8192

# Exp02 primary scaling conditions
CONCURRENCY_LEVELS = [1, 2, 4, 8]

# Formal experiment:
# 10 blocks × 4 conditions = 40 measured rounds
NUM_BLOCKS = 10

# Reproducible interleaving
RANDOM_SEED = 20260928

# Exp02 workload-specific preconditioning
PRECONDITION_CONCURRENCY = 8
PRECONDITION_ROUNDS = 10

# Enough prompt variants to rotate across measured rounds
PROMPT_POOL_SIZE = 24

# No intentional cooling gap
ROUND_GAP_SECONDS = 0.0

# Formal launch synchronization validity threshold
MAX_LAUNCH_SKEW_MS = 25.0

# Sampling is explicitly frozen.
TEMPERATURE = 0.0
TOP_P = 1.0

# Do not silently rely on EOS.
MIN_TOKENS = TARGET_COMPLETION_TOKENS
IGNORE_EOS = True


# ============================================================
# 2. Data structures
# ============================================================

@dataclass
class RequestResult:
    phase: str
    block_id: int
    round_id: int

    concurrency: int

    request_id: int
    prompt_id: int

    request_start: float
    first_content_time: float
    last_content_time: float
    request_end: float

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    ttft_ms: float
    tpot_ms: float
    e2e_ms: float


@dataclass
class RoundResult:
    phase: str

    block_id: int
    round_id: int

    concurrency: int

    makespan_s: float
    launch_skew_ms: float

    request_throughput: float
    output_throughput: float

    mean_ttft_ms: float
    median_ttft_ms: float

    mean_tpot_ms: float
    median_tpot_ms: float

    mean_e2e_ms: float
    median_e2e_ms: float

    total_completion_tokens: int

    launch_skew_valid: bool


# ============================================================
# 3. Utility
# ============================================================

def now_string():
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def safe_mean(values):
    values = list(values)

    if not values:
        return float("nan")

    return mean(values)


def safe_stdev(values):
    values = list(values)

    if len(values) < 2:
        return float("nan")

    return stdev(values)


def coefficient_of_variation(values):
    values = list(values)

    if len(values) < 2:
        return float("nan")

    m = mean(values)

    if m == 0:
        return float("nan")

    return stdev(values) / m


# ============================================================
# 4. Server/model helpers
# ============================================================

async def get_model_name(client):
    models = await client.models.list()

    if not models.data:
        raise RuntimeError(
            "vLLM /v1/models 没有返回模型。"
        )

    return models.data[0].id


async def measure_prompt_tokens(
    client,
    model_name,
    user_text,
):
    response = await client.post(
        "/chat/completions/render",
        cast_to=object,
        body={
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": user_text,
                }
            ],
            "max_tokens":
                TARGET_COMPLETION_TOKENS,
        },
    )

    if not isinstance(response, dict):
        raise RuntimeError(
            "render endpoint 返回类型异常: "
            f"{type(response)}"
        )

    token_ids = response.get(
        "token_ids"
    )

    if token_ids is None:
        raise RuntimeError(
            "render endpoint 没有返回 token_ids。"
        )

    return len(token_ids)


# ============================================================
# 5. Exact 2048-token prompt construction
# ============================================================

PROMPT_PREFIX_TEMPLATE = (
    "Request identifier: {prompt_id:06d}.\n"
    "Read the following technical passage carefully. "
    "When answering, continue generating concise technical "
    "observations until the generation limit is reached.\n\n"
)

BASE_UNIT = (
    "Artificial intelligence systems require careful "
    "measurement, reproducibility, workload control, "
    "latency analysis, throughput analysis, and "
    "experimental validation. "
)

# Fine-grained padding unit.
# We still verify every final prompt using vLLM render.
FINE_UNIT = " measurement"


async def build_exact_prompt(
    client,
    model_name,
    prompt_id,
):
    """
    Construct one unique prompt whose post-chat-template
    prompt length is exactly TARGET_PROMPT_TOKENS.

    Strategy:
      1. unique request identifier
      2. coarse BASE_UNIT repetition
      3. fine padding adjustment
      4. authoritative verification through vLLM render
    """

    prefix = PROMPT_PREFIX_TEMPLATE.format(
        prompt_id=prompt_id
    )

    prefix_tokens = await measure_prompt_tokens(
        client,
        model_name,
        prefix,
    )

    one_unit_tokens = await measure_prompt_tokens(
        client,
        model_name,
        prefix + BASE_UNIT,
    )

    coarse_increment = (
        one_unit_tokens
        - prefix_tokens
    )

    if coarse_increment <= 0:
        raise RuntimeError(
            "无法估算 BASE_UNIT token 增量。"
        )

    # Leave some room for fine adjustment.
    estimated_coarse = max(
        1,
        int(
            (
                TARGET_PROMPT_TOKENS
                - prefix_tokens
                - 30
            )
            / coarse_increment
        ),
    )

    # Search a small neighborhood.
    candidate_coarse_values = range(
        max(1, estimated_coarse - 4),
        estimated_coarse + 6,
    )

    best_under_text = None
    best_under_tokens = -1

    for coarse_count in candidate_coarse_values:

        text = (
            prefix
            + BASE_UNIT * coarse_count
        )

        tokens = await measure_prompt_tokens(
            client,
            model_name,
            text,
        )

        if (
            tokens
            <= TARGET_PROMPT_TOKENS
            and tokens > best_under_tokens
        ):
            best_under_text = text
            best_under_tokens = tokens

    if best_under_text is None:
        raise RuntimeError(
            f"prompt_id={prompt_id}: "
            "无法找到低于目标 token 数的 coarse prompt。"
        )

    if best_under_tokens == TARGET_PROMPT_TOKENS:
        return (
            best_under_text,
            best_under_tokens,
        )

    # Estimate fine increment.
    one_fine_tokens = await measure_prompt_tokens(
        client,
        model_name,
        best_under_text + FINE_UNIT,
    )

    fine_increment = (
        one_fine_tokens
        - best_under_tokens
    )

    if fine_increment <= 0:
        raise RuntimeError(
            f"prompt_id={prompt_id}: "
            "FINE_UNIT 没有产生正 token 增量。"
        )

    estimated_fine = max(
        0,
        round(
            (
                TARGET_PROMPT_TOKENS
                - best_under_tokens
            )
            / fine_increment
        ),
    )

    # Exhaustively search nearby counts.
    fine_low = max(
        0,
        estimated_fine - 20,
    )

    fine_high = (
        estimated_fine + 20
    )

    for fine_count in range(
        fine_low,
        fine_high + 1,
    ):

        text = (
            best_under_text
            + FINE_UNIT * fine_count
        )

        tokens = await measure_prompt_tokens(
            client,
            model_name,
            text,
        )

        if tokens == TARGET_PROMPT_TOKENS:

            if (
                tokens
                + TARGET_COMPLETION_TOKENS
                > MAX_MODEL_LEN
            ):
                raise RuntimeError(
                    "Prompt + completion 超过 max_model_len。"
                )

            return text, tokens

    raise RuntimeError(
        f"prompt_id={prompt_id}: "
        "无法构造精确 2048-token prompt。"
    )


async def build_prompt_pool(
    client,
    model_name,
):
    print()
    print(
        f"Building {PROMPT_POOL_SIZE} unique "
        f"exact-{TARGET_PROMPT_TOKENS}-token prompts..."
    )

    pool = []

    for prompt_id in range(
        PROMPT_POOL_SIZE
    ):

        prompt_text, token_count = (
            await build_exact_prompt(
                client,
                model_name,
                prompt_id,
            )
        )

        if token_count != TARGET_PROMPT_TOKENS:
            raise RuntimeError(
                f"prompt_id={prompt_id}: "
                f"token_count={token_count}"
            )

        pool.append(
            {
                "prompt_id": prompt_id,
                "text": prompt_text,
            }
        )

        print(
            f"  prompt {prompt_id:02d} "
            f"= {token_count} tokens"
        )

    print(
        "Prompt pool validation = PASS"
    )

    return pool


# ============================================================
# 6. vLLM telemetry
# ============================================================

async def fetch_metrics_text():
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


def extract_metric(
    metrics_text,
    metric_name,
):
    """
    Extract the first scalar value for a metric.
    """

    pattern = re.compile(
        rf"^{re.escape(metric_name)}"
        rf"\{{[^}}]*\}}\s+"
        rf"([-+0-9.eE]+)$",
        re.MULTILINE,
    )

    match = pattern.search(
        metrics_text
    )

    if match is None:
        return None

    return float(
        match.group(1)
    )


async def get_vllm_snapshot():
    text = await fetch_metrics_text()

    return {
        "preemptions_total":
            extract_metric(
                text,
                "vllm:num_preemptions_total",
            ),

        "kv_cache_usage":
            extract_metric(
                text,
                "vllm:kv_cache_usage_perc",
            ),

        "running":
            extract_metric(
                text,
                "vllm:num_requests_running",
            ),

        "waiting":
            extract_metric(
                text,
                "vllm:num_requests_waiting",
            ),
    }


# ============================================================
# 7. One request
# ============================================================

async def worker(
    *,
    phase,
    block_id,
    round_id,

    concurrency,

    request_id,

    prompt_entry,

    model_name,
    client,

    start_event,
    ready_event,
    ready_state,
    ready_lock,
):

    async with ready_lock:

        ready_state["count"] += 1

        if (
            ready_state["count"]
            == ready_state["target"]
        ):
            ready_event.set()

    await start_event.wait()

    request_start = time.perf_counter()

    first_content_time = None
    last_content_time = None

    usage = None

    async with client.chat.completions.stream(
        model=model_name,

        messages=[
            {
                "role": "user",
                "content":
                    prompt_entry["text"],
            }
        ],

        temperature=TEMPERATURE,
        top_p=TOP_P,

        max_tokens=
            TARGET_COMPLETION_TOKENS,

        extra_body={
            "min_tokens":
                MIN_TOKENS,

            "ignore_eos":
                IGNORE_EOS,
        },

        stream_options={
            "include_usage": True,
        },

    ) as stream:

        async for event in stream:

            if event.type == "content.delta":

                content = event.delta

                if content:

                    now = (
                        time.perf_counter()
                    )

                    if first_content_time is None:
                        first_content_time = now

                    last_content_time = now

            elif event.type == "chunk":

                chunk = event.chunk

                if chunk.usage is not None:
                    usage = chunk.usage

    request_end = time.perf_counter()

    if first_content_time is None:
        raise RuntimeError(
            f"phase={phase}, "
            f"round={round_id}, "
            f"request={request_id}: "
            "没有收到非空 content。"
        )

    if last_content_time is None:
        raise RuntimeError(
            "缺少 last_content_time。"
        )

    if usage is None:
        raise RuntimeError(
            "没有收到 usage。"
        )

    # --------------------------------------------------------
    # Formal fail-fast validity checks
    # --------------------------------------------------------

    if (
        usage.prompt_tokens
        != TARGET_PROMPT_TOKENS
    ):
        raise RuntimeError(
            f"prompt_tokens="
            f"{usage.prompt_tokens}, "
            f"expected="
            f"{TARGET_PROMPT_TOKENS}"
        )

    if (
        usage.completion_tokens
        != TARGET_COMPLETION_TOKENS
    ):
        raise RuntimeError(
            f"completion_tokens="
            f"{usage.completion_tokens}, "
            f"expected="
            f"{TARGET_COMPLETION_TOKENS}"
        )

    if (
        usage.total_tokens
        != usage.prompt_tokens
        + usage.completion_tokens
    ):
        raise RuntimeError(
            "usage arithmetic check FAILED"
        )

    ttft_ms = (
        first_content_time
        - request_start
    ) * 1000.0

    e2e_ms = (
        request_end
        - request_start
    ) * 1000.0

    if usage.completion_tokens <= 1:
        raise RuntimeError(
            "completion_tokens <= 1，"
            "无法计算 TPOT。"
        )

    tpot_ms = (
        last_content_time
        - first_content_time
    ) / (
        usage.completion_tokens - 1
    ) * 1000.0

    return RequestResult(
        phase=phase,

        block_id=block_id,
        round_id=round_id,

        concurrency=concurrency,

        request_id=request_id,
        prompt_id=
            prompt_entry["prompt_id"],

        request_start=request_start,
        first_content_time=
            first_content_time,
        last_content_time=
            last_content_time,
        request_end=request_end,

        prompt_tokens=
            usage.prompt_tokens,

        completion_tokens=
            usage.completion_tokens,

        total_tokens=
            usage.total_tokens,

        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
    )


# ============================================================
# 8. One synchronized burst round
# ============================================================

async def run_round(
    *,
    phase,
    block_id,
    round_id,

    concurrency,

    prompt_entries,

    model_name,
    client,
):

    if len(prompt_entries) != concurrency:
        raise RuntimeError(
            "prompt_entries 数量必须等于 concurrency。"
        )

    start_event = asyncio.Event()
    ready_event = asyncio.Event()
    ready_lock = asyncio.Lock()

    ready_state = {
        "count": 0,
        "target": concurrency,
    }

    tasks = []

    for request_id in range(
        concurrency
    ):

        task = asyncio.create_task(
            worker(
                phase=phase,

                block_id=block_id,
                round_id=round_id,

                concurrency=concurrency,

                request_id=request_id,

                prompt_entry=
                    prompt_entries[
                        request_id
                    ],

                model_name=model_name,
                client=client,

                start_event=start_event,
                ready_event=ready_event,
                ready_state=ready_state,
                ready_lock=ready_lock,
            )
        )

        tasks.append(task)

    await ready_event.wait()

    group_start = time.perf_counter()

    start_event.set()

    results = await asyncio.gather(
        *tasks
    )

    group_end = time.perf_counter()

    makespan_s = (
        group_end
        - group_start
    )

    starts = [
        result.request_start
        for result in results
    ]

    launch_skew_ms = (
        max(starts)
        - min(starts)
    ) * 1000.0

    total_completion_tokens = sum(
        result.completion_tokens
        for result in results
    )

    request_throughput = (
        concurrency
        / makespan_s
    )

    output_throughput = (
        total_completion_tokens
        / makespan_s
    )

    ttfts = [
        result.ttft_ms
        for result in results
    ]

    tpots = [
        result.tpot_ms
        for result in results
    ]

    e2es = [
        result.e2e_ms
        for result in results
    ]

    round_result = RoundResult(
        phase=phase,

        block_id=block_id,
        round_id=round_id,

        concurrency=concurrency,

        makespan_s=makespan_s,

        launch_skew_ms=
            launch_skew_ms,

        request_throughput=
            request_throughput,

        output_throughput=
            output_throughput,

        mean_ttft_ms=
            mean(ttfts),

        median_ttft_ms=
            median(ttfts),

        mean_tpot_ms=
            mean(tpots),

        median_tpot_ms=
            median(tpots),

        mean_e2e_ms=
            mean(e2es),

        median_e2e_ms=
            median(e2es),

        total_completion_tokens=
            total_completion_tokens,

        launch_skew_valid=(
            launch_skew_ms
            <= MAX_LAUNCH_SKEW_MS
        ),
    )

    return (
        results,
        round_result,
    )


# ============================================================
# 9. CSV helpers
# ============================================================

def write_csv(
    path,
    rows,
):
    rows = list(rows)

    if not rows:
        return

    dictionaries = [
        asdict(row)
        for row in rows
    ]

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
                    dictionaries[0].keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            dictionaries
        )


# ============================================================
# 10. Prompt assignment
# ============================================================

def choose_prompts(
    prompt_pool,
    *,
    concurrency,
    cursor,
):
    """
    Rotate through the pool so:
      - requests inside a round are unique
      - prompt IDs vary across rounds
    """

    selected = []

    pool_size = len(
        prompt_pool
    )

    for i in range(
        concurrency
    ):

        index = (
            cursor + i
        ) % pool_size

        selected.append(
            prompt_pool[index]
        )

    new_cursor = (
        cursor + concurrency
    ) % pool_size

    return selected, new_cursor


# ============================================================
# 11. Build deterministic interleaved block order
# ============================================================

def make_block_orders():
    rng = random.Random(
        RANDOM_SEED
    )

    block_orders = []

    for block_id in range(
        1,
        NUM_BLOCKS + 1,
    ):

        order = (
            CONCURRENCY_LEVELS.copy()
        )

        rng.shuffle(
            order
        )

        block_orders.append(
            {
                "block_id": block_id,
                "order": order,
            }
        )

    return block_orders


# ============================================================
# 12. Formal summary
# ============================================================

def print_condition_summary(
    formal_rounds,
):
    print()
    print("=" * 100)
    print("EXP02 FORMAL CONDITION SUMMARY")
    print("=" * 100)

    print(
        f"{'C':>4} "
        f"{'Rounds':>8} "
        f"{'TTFT mean':>12} "
        f"{'TPOT mean':>12} "
        f"{'OutTok/s':>12} "
        f"{'OutTok CV':>11} "
        f"{'Req/s':>10} "
        f"{'Skew max':>10}"
    )

    print("-" * 100)

    for concurrency in (
        CONCURRENCY_LEVELS
    ):

        rows = [
            row
            for row in formal_rounds
            if row.concurrency
            == concurrency
        ]

        ttft_values = [
            r.mean_ttft_ms
            for r in rows
        ]

        tpot_values = [
            r.mean_tpot_ms
            for r in rows
        ]

        output_values = [
            r.output_throughput
            for r in rows
        ]

        req_values = [
            r.request_throughput
            for r in rows
        ]

        skew_values = [
            r.launch_skew_ms
            for r in rows
        ]

        print(
            f"{concurrency:>4} "
            f"{len(rows):>8} "
            f"{mean(ttft_values):>12.3f} "
            f"{mean(tpot_values):>12.3f} "
            f"{mean(output_values):>12.3f} "
            f"{coefficient_of_variation(output_values):>11.4f} "
            f"{mean(req_values):>10.3f} "
            f"{max(skew_values):>10.3f}"
        )


# ============================================================
# 13. Main
# ============================================================

async def main():

    run_tag = now_string()

    output_dir = Path(
        f"exp02_formal_{run_tag}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    request_csv_path = (
        output_dir
        / "formal_requests.csv"
    )

    round_csv_path = (
        output_dir
        / "formal_rounds.csv"
    )

    precondition_csv_path = (
        output_dir
        / "preconditioning_rounds.csv"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    print("=" * 100)
    print("Exp02 Formal Benchmark")
    print("=" * 100)

    print(
        f"Output directory        : "
        f"{output_dir}"
    )

    print(
        f"Concurrency levels      : "
        f"{CONCURRENCY_LEVELS}"
    )

    print(
        f"Formal blocks           : "
        f"{NUM_BLOCKS}"
    )

    print(
        f"Preconditioning         : "
        f"{PRECONDITION_ROUNDS} × "
        f"C{PRECONDITION_CONCURRENCY}"
    )

    print(
        f"Prompt tokens           : "
        f"{TARGET_PROMPT_TOKENS}"
    )

    print(
        f"Completion tokens       : "
        f"{TARGET_COMPLETION_TOKENS}"
    )

    print(
        f"Random seed             : "
        f"{RANDOM_SEED}"
    )

    print(
        f"Launch skew threshold   : "
        f"{MAX_LAUNCH_SKEW_MS:.1f} ms"
    )

    block_orders = (
        make_block_orders()
    )

    print()
    print(
        "Formal interleaved order:"
    )

    for block in block_orders:

        print(
            f"  Block "
            f"{block['block_id']:02d}: "
            + " -> ".join(
                f"C{x}"
                for x
                in block["order"]
            )
        )

    async with AsyncOpenAI(
        base_url=API_BASE,
        api_key=API_KEY,
        http_client=
            DefaultAioHttpClient(),
    ) as client:

        model_name = (
            await get_model_name(
                client
            )
        )

        print()
        print(
            f"Model                   : "
            f"{model_name}"
        )

        prompt_pool = (
            await build_prompt_pool(
                client,
                model_name,
            )
        )

        # ----------------------------------------------------
        # Initial telemetry
        # ----------------------------------------------------

        metrics_initial = (
            await get_vllm_snapshot()
        )

        print()
        print(
            "Initial vLLM telemetry:"
        )

        print(
            json.dumps(
                metrics_initial,
                indent=2,
            )
        )

        # ----------------------------------------------------
        # Preconditioning
        # ----------------------------------------------------

        print()
        print("=" * 100)
        print(
            "PRECONDITIONING "
            "(NOT FORMAL DATA)"
        )
        print("=" * 100)

        preconditioning_rounds = []

        prompt_cursor = 0

        for pre_round in range(
            1,
            PRECONDITION_ROUNDS + 1,
        ):

            prompts, prompt_cursor = (
                choose_prompts(
                    prompt_pool,

                    concurrency=
                        PRECONDITION_CONCURRENCY,

                    cursor=
                        prompt_cursor,
                )
            )

            _, round_result = (
                await run_round(
                    phase="preconditioning",

                    block_id=0,

                    round_id=
                        pre_round,

                    concurrency=
                        PRECONDITION_CONCURRENCY,

                    prompt_entries=
                        prompts,

                    model_name=
                        model_name,

                    client=client,
                )
            )

            preconditioning_rounds.append(
                round_result
            )

            print(
                f"Pre {pre_round:02d}/"
                f"{PRECONDITION_ROUNDS} | "
                f"TTFT="
                f"{round_result.mean_ttft_ms:.1f} ms | "
                f"TPOT="
                f"{round_result.mean_tpot_ms:.3f} | "
                f"OutTok/s="
                f"{round_result.output_throughput:.2f} | "
                f"Skew="
                f"{round_result.launch_skew_ms:.2f} ms"
            )

        write_csv(
            precondition_csv_path,
            preconditioning_rounds,
        )

        metrics_after_precondition = (
            await get_vllm_snapshot()
        )

        # ----------------------------------------------------
        # Formal experiment
        # ----------------------------------------------------

        print()
        print("=" * 100)
        print(
            "FORMAL MEASUREMENT START"
        )
        print("=" * 100)

        formal_requests = []
        formal_rounds = []

        formal_round_id = 0

        # Reset prompt rotation so formal assignment
        # is independent of preconditioning.
        prompt_cursor = 0

        for block in block_orders:

            block_id = block[
                "block_id"
            ]

            order = block[
                "order"
            ]

            print()
            print(
                f"--- Block "
                f"{block_id:02d}/"
                f"{NUM_BLOCKS} ---"
            )

            for concurrency in order:

                formal_round_id += 1

                prompts, prompt_cursor = (
                    choose_prompts(
                        prompt_pool,

                        concurrency=
                            concurrency,

                        cursor=
                            prompt_cursor,
                    )
                )

                request_results, round_result = (
                    await run_round(
                        phase="formal",

                        block_id=
                            block_id,

                        round_id=
                            formal_round_id,

                        concurrency=
                            concurrency,

                        prompt_entries=
                            prompts,

                        model_name=
                            model_name,

                        client=client,
                    )
                )

                formal_requests.extend(
                    request_results
                )

                formal_rounds.append(
                    round_result
                )

                # Save after every round.
                # If anything crashes later, existing
                # raw evidence is preserved.
                write_csv(
                    request_csv_path,
                    formal_requests,
                )

                write_csv(
                    round_csv_path,
                    formal_rounds,
                )

                validity_text = (
                    "PASS"
                    if round_result.launch_skew_valid
                    else "FAIL"
                )

                print(
                    f"Round "
                    f"{formal_round_id:02d} | "
                    f"Block="
                    f"{block_id:02d} | "
                    f"C={concurrency:02d} | "
                    f"TTFT="
                    f"{round_result.mean_ttft_ms:8.1f} ms | "
                    f"TPOT="
                    f"{round_result.mean_tpot_ms:7.3f} | "
                    f"OutTok/s="
                    f"{round_result.output_throughput:7.2f} | "
                    f"Skew="
                    f"{round_result.launch_skew_ms:6.2f} | "
                    f"SkewValid="
                    f"{validity_text}"
                )

                if (
                    not round_result
                    .launch_skew_valid
                ):
                    raise RuntimeError(
                        "Formal round launch skew "
                        "超过预注册阈值 "
                        f"{MAX_LAUNCH_SKEW_MS} ms。"
                    )

                if ROUND_GAP_SECONDS > 0:
                    await asyncio.sleep(
                        ROUND_GAP_SECONDS
                    )

        # ----------------------------------------------------
        # Final telemetry
        # ----------------------------------------------------

        metrics_final = (
            await get_vllm_snapshot()
        )

        # ----------------------------------------------------
        # Global formal validity
        # ----------------------------------------------------

        pre_before_formal = (
            metrics_after_precondition
            .get(
                "preemptions_total"
            )
        )

        pre_after_formal = (
            metrics_final.get(
                "preemptions_total"
            )
        )

        if (
            pre_before_formal is not None
            and pre_after_formal is not None
        ):
            formal_preemption_delta = (
                pre_after_formal
                - pre_before_formal
            )
        else:
            formal_preemption_delta = None

        all_skew_valid = all(
            r.launch_skew_valid
            for r in formal_rounds
        )

        expected_rounds = (
            NUM_BLOCKS
            * len(
                CONCURRENCY_LEVELS
            )
        )

        round_count_valid = (
            len(formal_rounds)
            == expected_rounds
        )

        request_count_expected = (
            NUM_BLOCKS
            * sum(
                CONCURRENCY_LEVELS
            )
        )

        request_count_valid = (
            len(formal_requests)
            == request_count_expected
        )

        no_formal_preemption = (
            formal_preemption_delta
            in (
                None,
                0.0,
            )
        )

        formal_valid = (
            all_skew_valid
            and round_count_valid
            and request_count_valid
            and no_formal_preemption
        )

        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        metadata = {
            "run_tag":
                run_tag,

            "model":
                model_name,

            "api_base":
                API_BASE,

            "target_prompt_tokens":
                TARGET_PROMPT_TOKENS,

            "target_completion_tokens":
                TARGET_COMPLETION_TOKENS,

            "max_model_len":
                MAX_MODEL_LEN,

            "concurrency_levels":
                CONCURRENCY_LEVELS,

            "num_blocks":
                NUM_BLOCKS,

            "random_seed":
                RANDOM_SEED,

            "preconditioning": {
                "concurrency":
                    PRECONDITION_CONCURRENCY,

                "rounds":
                    PRECONDITION_ROUNDS,

                "round_gap_seconds":
                    ROUND_GAP_SECONDS,
            },

            "sampling": {
                "temperature":
                    TEMPERATURE,

                "top_p":
                    TOP_P,

                "min_tokens":
                    MIN_TOKENS,

                "ignore_eos":
                    IGNORE_EOS,
            },

            "max_launch_skew_ms":
                MAX_LAUNCH_SKEW_MS,

            "block_orders":
                block_orders,

            "metrics_initial":
                metrics_initial,

            "metrics_after_preconditioning":
                metrics_after_precondition,

            "metrics_final":
                metrics_final,

            "formal_preemption_delta":
                formal_preemption_delta,

            "formal_round_count":
                len(formal_rounds),

            "expected_round_count":
                expected_rounds,

            "formal_request_count":
                len(formal_requests),

            "expected_request_count":
                request_count_expected,

            "validity": {
                "all_launch_skew_valid":
                    all_skew_valid,

                "round_count_valid":
                    round_count_valid,

                "request_count_valid":
                    request_count_valid,

                "no_formal_preemption":
                    no_formal_preemption,

                "formal_valid":
                    formal_valid,
            },
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print_condition_summary(
            formal_rounds
        )

        print()
        print("=" * 100)
        print("FORMAL VALIDITY")
        print("=" * 100)

        print(
            f"Round count       : "
            f"{len(formal_rounds)} / "
            f"{expected_rounds} "
            f"{'PASS' if round_count_valid else 'FAIL'}"
        )

        print(
            f"Request count     : "
            f"{len(formal_requests)} / "
            f"{request_count_expected} "
            f"{'PASS' if request_count_valid else 'FAIL'}"
        )

        print(
            f"Launch skew       : "
            f"{'PASS' if all_skew_valid else 'FAIL'}"
        )

        print(
            f"Formal preemptions: "
            f"{formal_preemption_delta}"
        )

        print(
            f"No preemption     : "
            f"{'PASS' if no_formal_preemption else 'FAIL'}"
        )

        print()

        print(
            "EXP02 FORMAL VALIDITY = "
            f"{'PASS' if formal_valid else 'FAIL'}"
        )

        print()
        print(
            f"Raw requests      : "
            f"{request_csv_path}"
        )

        print(
            f"Round summaries   : "
            f"{round_csv_path}"
        )

        print(
            f"Preconditioning   : "
            f"{precondition_csv_path}"
        )

        print(
            f"Metadata          : "
            f"{metadata_path}"
        )


if __name__ == "__main__":
    asyncio.run(
        main()
    )