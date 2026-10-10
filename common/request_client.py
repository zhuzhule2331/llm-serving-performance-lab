# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp08_mbt_concurrency/exp08_burst_FROZEN.py
# Copied  : verbatim except for the changes listed in ../../PUBLIC_CODE_CHANGELOG.md
# Frozen  : this file is the FINAL public-facing implementation used in the
#           formal campaign; it is a copy, the original is unmodified.
# =============================================================================
import argparse
import asyncio
import time
from dataclasses import dataclass
from statistics import mean, median

from openai import AsyncOpenAI, DefaultAioHttpClient


API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"

CONCURRENCY = 8

TARGET_PROMPT_TOKENS = 2048
TARGET_COMPLETION_TOKENS = 128
MAX_MODEL_LEN = 8192

TEMPERATURE = 0.0
TOP_P = 1.0

MIN_TOKENS = TARGET_COMPLETION_TOKENS
IGNORE_EOS = True

MAX_LAUNCH_SKEW_MS = 25.0


# ============================================================
# Exact prompt construction
# Same basic workload semantics as Exp02
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

FINE_UNIT = " measurement"


@dataclass
class RequestResult:
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
            "max_tokens": TARGET_COMPLETION_TOKENS,
        },
    )

    if not isinstance(response, dict):
        raise RuntimeError(
            f"render endpoint 返回类型异常: {type(response)}"
        )

    token_ids = response.get("token_ids")

    if token_ids is None:
        raise RuntimeError(
            "render endpoint 没有返回 token_ids。"
        )

    return len(token_ids)


async def build_exact_prompt(
    client,
    model_name,
    prompt_id,
):
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
        one_unit_tokens - prefix_tokens
    )

    if coarse_increment <= 0:
        raise RuntimeError(
            f"prompt_id={prompt_id}: "
            "无法估算 BASE_UNIT token 增量。"
        )

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
            tokens <= TARGET_PROMPT_TOKENS
            and tokens > best_under_tokens
        ):
            best_under_text = text
            best_under_tokens = tokens

    if best_under_text is None:
        raise RuntimeError(
            f"prompt_id={prompt_id}: "
            "无法找到 coarse prompt。"
        )

    if best_under_tokens == TARGET_PROMPT_TOKENS:
        return best_under_text, best_under_tokens

    one_fine_tokens = await measure_prompt_tokens(
        client,
        model_name,
        best_under_text + FINE_UNIT,
    )

    fine_increment = (
        one_fine_tokens - best_under_tokens
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

    fine_low = max(
        0,
        estimated_fine - 20,
    )

    fine_high = estimated_fine + 20

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
        f"无法构造精确 {TARGET_PROMPT_TOKENS}-token prompt。"
    )


async def build_prompt_pool(
    client,
    model_name,
):
    print()
    print(
        f"Building {CONCURRENCY} unique "
        f"exact-{TARGET_PROMPT_TOKENS}-token prompts..."
    )

    prompts = []

    for prompt_id in range(CONCURRENCY):
        text, token_count = await build_exact_prompt(
            client,
            model_name,
            prompt_id,
        )

        if token_count != TARGET_PROMPT_TOKENS:
            raise RuntimeError(
                f"prompt_id={prompt_id}: "
                f"token_count={token_count}"
            )

        prompts.append(
            {
                "prompt_id": prompt_id,
                "text": text,
            }
        )

        print(
            f"  prompt {prompt_id}: "
            f"{token_count} tokens"
        )

    return prompts


# ============================================================
# One synchronized worker
# ============================================================

async def worker(
    *,
    request_id,
    prompt_entry,
    model_name,
    client,
    start_event,
    ready_event,
    ready_state,
    ready_lock,
):
    # Tell controller this task is ready to launch.
    async with ready_lock:
        ready_state["count"] += 1

        if (
            ready_state["count"]
            == ready_state["target"]
        ):
            ready_event.set()

    # All workers wait here.
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
                "content": prompt_entry["text"],
            }
        ],

        temperature=TEMPERATURE,
        top_p=TOP_P,

        max_tokens=TARGET_COMPLETION_TOKENS,

        extra_body={
            "min_tokens": MIN_TOKENS,
            "ignore_eos": IGNORE_EOS,
        },

        stream_options={
            "include_usage": True,
        },

    ) as stream:

        async for event in stream:

            if event.type == "content.delta":
                content = event.delta

                if content:
                    now = time.perf_counter()

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
            f"request={request_id}: "
            "没有收到非空 content。"
        )

    if last_content_time is None:
        raise RuntimeError(
            f"request={request_id}: "
            "缺少 last_content_time。"
        )

    if usage is None:
        raise RuntimeError(
            f"request={request_id}: "
            "没有收到 usage。"
        )

    # --------------------------------------------------------
    # Fail-fast validity
    # --------------------------------------------------------

    if usage.prompt_tokens != TARGET_PROMPT_TOKENS:
        raise RuntimeError(
            f"request={request_id}: "
            f"prompt_tokens={usage.prompt_tokens}, "
            f"expected={TARGET_PROMPT_TOKENS}"
        )

    if usage.completion_tokens != TARGET_COMPLETION_TOKENS:
        raise RuntimeError(
            f"request={request_id}: "
            f"completion_tokens={usage.completion_tokens}, "
            f"expected={TARGET_COMPLETION_TOKENS}"
        )

    if (
        usage.total_tokens
        != usage.prompt_tokens
        + usage.completion_tokens
    ):
        raise RuntimeError(
            f"request={request_id}: "
            "usage arithmetic FAILED"
        )

    ttft_ms = (
        first_content_time
        - request_start
    ) * 1000.0

    e2e_ms = (
        request_end
        - request_start
    ) * 1000.0

    tpot_ms = (
        last_content_time
        - first_content_time
    ) / (
        usage.completion_tokens - 1
    ) * 1000.0

    return RequestResult(
        request_id=request_id,
        prompt_id=prompt_entry["prompt_id"],

        request_start=request_start,
        first_content_time=first_content_time,
        last_content_time=last_content_time,
        request_end=request_end,

        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,

        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
    )


# ============================================================
# Synchronized burst
# ============================================================

async def run_burst(
    *,
    prompt_entries,
    model_name,
    client,
):
    if len(prompt_entries) != CONCURRENCY:
        raise RuntimeError(
            "prompt_entries 数量错误。"
        )

    start_event = asyncio.Event()
    ready_event = asyncio.Event()
    ready_lock = asyncio.Lock()

    ready_state = {
        "count": 0,
        "target": CONCURRENCY,
    }

    tasks = []

    for request_id in range(CONCURRENCY):
        task = asyncio.create_task(
            worker(
                request_id=request_id,

                prompt_entry=
                    prompt_entries[request_id],

                model_name=model_name,
                client=client,

                start_event=start_event,
                ready_event=ready_event,
                ready_state=ready_state,
                ready_lock=ready_lock,
            )
        )

        tasks.append(task)

    # Do not release until all workers are ready.
    await ready_event.wait()

    group_start = time.perf_counter()

    start_event.set()

    results = await asyncio.gather(
        *tasks
    )

    group_end = time.perf_counter()

    makespan_s = (
        group_end - group_start
    )

    starts = [
        r.request_start
        for r in results
    ]

    launch_skew_ms = (
        max(starts)
        - min(starts)
    ) * 1000.0

    total_completion_tokens = sum(
        r.completion_tokens
        for r in results
    )

    request_throughput = (
        CONCURRENCY / makespan_s
    )

    output_throughput = (
        total_completion_tokens
        / makespan_s
    )

    return (
        results,
        makespan_s,
        launch_skew_ms,
        request_throughput,
        output_throughput,
    )



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Exp07 synchronized exact-token serving burst"
        )
    )

    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=2048,
        help="Exact rendered prompt token target.",
    )

    parser.add_argument(
        "--completion-tokens",
        type=int,
        default=128,
        help="Exact completion token target.",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of synchronized requests.",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Server max_model_len used for validity checks.",
    )

    parser.add_argument(
        "--max-launch-skew-ms",
        type=float,
        default=25.0,
        help="Maximum valid synchronized launch skew.",
    )

    args = parser.parse_args()

    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be > 0")

    if args.completion_tokens <= 1:
        parser.error(
            "--completion-tokens must be > 1 "
            "because TPOT uses completion_tokens - 1"
        )

    if args.concurrency <= 0:
        parser.error("--concurrency must be > 0")

    if (
        args.prompt_tokens
        + args.completion_tokens
        > args.max_model_len
    ):
        parser.error(
            "prompt_tokens + completion_tokens "
            "exceeds max_model_len"
        )

    return args


async def main():
    global CONCURRENCY
    global TARGET_PROMPT_TOKENS
    global TARGET_COMPLETION_TOKENS
    global MAX_MODEL_LEN
    global MIN_TOKENS
    global MAX_LAUNCH_SKEW_MS

    args = parse_args()

    CONCURRENCY = args.concurrency
    TARGET_PROMPT_TOKENS = args.prompt_tokens
    TARGET_COMPLETION_TOKENS = args.completion_tokens
    MAX_MODEL_LEN = args.max_model_len
    MIN_TOKENS = TARGET_COMPLETION_TOKENS
    MAX_LAUNCH_SKEW_MS = args.max_launch_skew_ms

    print("=" * 76)
    print(
        f"Exp07: C{CONCURRENCY} P0 "
        f"exact-{TARGET_PROMPT_TOKENS} -> "
        f"exact-{TARGET_COMPLETION_TOKENS} "
        "synchronized burst"
    )
    print("=" * 76)

    async with AsyncOpenAI(
        base_url=API_BASE,
        api_key=API_KEY,
        http_client=DefaultAioHttpClient(),
    ) as client:

        model_name = await get_model_name(
            client
        )

        print()
        print("Model:")
        print(model_name)

        prompt_entries = await build_prompt_pool(
            client,
            model_name,
        )

        print()
        print(f"Launching synchronized C{CONCURRENCY} burst...")

        (
            results,
            makespan_s,
            launch_skew_ms,
            request_throughput,
            output_throughput,
        ) = await run_burst(
            prompt_entries=prompt_entries,
            model_name=model_name,
            client=client,
        )

        print()
        print("=" * 76)
        print("PER-REQUEST RESULTS")
        print("=" * 76)

        for r in sorted(
            results,
            key=lambda x: x.request_id,
        ):
            print(
                f"req={r.request_id} "
                f"prompt={r.prompt_tokens} "
                f"completion={r.completion_tokens} "
                f"TTFT={r.ttft_ms:.3f} ms "
                f"TPOT={r.tpot_ms:.3f} ms/token "
                f"E2E={r.e2e_ms:.3f} ms"
            )

        ttfts = [
            r.ttft_ms
            for r in results
        ]

        tpots = [
            r.tpot_ms
            for r in results
        ]

        e2es = [
            r.e2e_ms
            for r in results
        ]

        print()
        print("=" * 76)
        print("BURST SUMMARY")
        print("=" * 76)

        print(
            f"Concurrency        : {CONCURRENCY}"
        )

        print(
            f"Launch skew        : "
            f"{launch_skew_ms:.3f} ms"
        )

        print(
            f"Launch skew valid  : "
            f"{launch_skew_ms <= MAX_LAUNCH_SKEW_MS}"
        )

        print(
            f"Makespan           : "
            f"{makespan_s:.6f} s"
        )

        print(
            f"Mean TTFT          : "
            f"{mean(ttfts):.3f} ms"
        )

        print(
            f"Median TTFT        : "
            f"{median(ttfts):.3f} ms"
        )

        print(
            f"Mean TPOT          : "
            f"{mean(tpots):.3f} ms/token"
        )

        print(
            f"Median TPOT        : "
            f"{median(tpots):.3f} ms/token"
        )

        print(
            f"Mean E2E           : "
            f"{mean(e2es):.3f} ms"
        )

        print(
            f"Request throughput : "
            f"{request_throughput:.3f} req/s"
        )

        print(
            f"Output throughput  : "
            f"{output_throughput:.3f} tok/s"
        )

        # Final validity
        valid = (
            len(results) == CONCURRENCY
            and launch_skew_ms
                <= MAX_LAUNCH_SKEW_MS
            and all(
                r.prompt_tokens
                == TARGET_PROMPT_TOKENS
                for r in results
            )
            and all(
                r.completion_tokens
                == TARGET_COMPLETION_TOKENS
                for r in results
            )
        )

        print()
        print(
            "VALIDITY: "
            + (
                "PASS"
                if valid
                else "FAIL"
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
