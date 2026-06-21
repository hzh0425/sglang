"""Build a long single-request history, then probe short turns.

This benchmark is meant for radix-tree A/B tests.  It first grows one
conversation in fixed-size chunks so the prefix cache contains a long path.
Then it sends short probe turns whose prompt is the full history.  The probe
turns keep model prefill small while forcing match_prefix to walk a 128k+
context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from sglang.benchmark.utils import get_tokenizer
from sglang.test.kits.cache_hit_kit import async_request_sglang_generate, gen_payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-context", type=int, required=True)
    parser.add_argument("--build-chunk", type=int, default=8192)
    parser.add_argument("--probe-turns", type=int, default=8)
    parser.add_argument("--probe-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--disable-tree-flush", action="store_true")
    parser.add_argument("--tree-flush-sleep", type=float, default=1.1)
    parser.add_argument("--tree-flush-prompt-len", type=int, default=4)
    return parser.parse_args()


def make_token_source(model_path: str, seed: int):
    tokenizer = get_tokenizer(model_path)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_ids = [
        token_id
        for token_id in tokenizer.get_vocab().values()
        if isinstance(token_id, int) and token_id >= 0 and token_id not in special
    ]
    rng = random.Random(seed)

    def sample(n: int) -> list[int]:
        return rng.choices(vocab_ids, k=n)

    return sample


async def send_once(url: str, history: list[int], output_len: int):
    payload = gen_payload(history, output_len)
    response = await async_request_sglang_generate(payload, url)
    if not response.success:
        raise RuntimeError(response.error)
    return response


def summarize(rows: list[dict], phase: str) -> dict:
    selected = [row for row in rows if row["phase"] == phase]
    if not selected:
        return {}
    duration = selected[-1]["finished_at"] - selected[0]["started_at"]
    prompt_tokens = sum(row["prompt_len"] for row in selected)
    output_tokens = sum(row["generated_len"] for row in selected)
    cached_tokens = sum(row["cached_tokens"] for row in selected)
    ttft = [row["ttft"] for row in selected]
    latency = [row["latency"] for row in selected]
    return {
        "requests": len(selected),
        "duration": duration,
        "avg_prompt_len": prompt_tokens / len(selected),
        "avg_cached_tokens": cached_tokens / len(selected),
        "cache_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "avg_ttft": sum(ttft) / len(ttft),
        "avg_latency": sum(latency) / len(latency),
        "request_throughput": len(selected) / duration if duration > 0 else 0.0,
        "input_token_throughput": prompt_tokens / duration if duration > 0 else 0.0,
        "output_token_throughput": output_tokens / duration if duration > 0 else 0.0,
    }


async def main_async(args):
    url = f"http://{args.host}:{args.port}/generate"
    sample_tokens = make_token_source(args.model_path, args.seed)
    history: list[int] = []
    rows: list[dict] = []
    markers: list[dict] = []

    async def flush_tree_perf_marker(name: str) -> None:
        if args.disable_tree_flush:
            return
        if args.tree_flush_sleep > 0:
            await asyncio.sleep(args.tree_flush_sleep)
        marker_prompt = sample_tokens(args.tree_flush_prompt_len)
        started_wall = time.time()
        started_at = time.perf_counter()
        response = await send_once(url, marker_prompt, args.output_len)
        finished_at = time.perf_counter()
        finished_wall = time.time()
        markers.append(
            {
                "tag": args.tag,
                "name": name,
                "started_wall": started_wall,
                "finished_wall": finished_wall,
                "started_at": started_at,
                "finished_at": finished_at,
                "prompt_len": response.prompt_len,
                "cached_tokens": response.cached_tokens,
                "generated_len": response.generated_len,
                "ttft": response.ttft,
                "latency": response.latency,
            }
        )

    await flush_tree_perf_marker("before_build")
    build_round = 0
    while len(history) + args.probe_len + args.output_len < args.target_context:
        remaining = args.target_context - args.probe_len - args.output_len - len(history)
        chunk_len = min(args.build_chunk, remaining)
        if chunk_len <= 0:
            break
        history.extend(sample_tokens(chunk_len))
        started_wall = time.time()
        started_at = time.perf_counter()
        response = await send_once(url, history, args.output_len)
        finished_at = time.perf_counter()
        finished_wall = time.time()
        history.extend(response.output_ids)
        rows.append(
            {
                "tag": args.tag,
                "phase": "build",
                "round": build_round,
                "started_wall": started_wall,
                "finished_wall": finished_wall,
                "started_at": started_at,
                "finished_at": finished_at,
                "prompt_len": response.prompt_len,
                "cached_tokens": response.cached_tokens,
                "generated_len": response.generated_len,
                "ttft": response.ttft,
                "latency": response.latency,
                "context_len_after": len(history),
            }
        )
        build_round += 1

    await flush_tree_perf_marker("after_build")
    for probe_round in range(args.probe_turns):
        history.extend(sample_tokens(args.probe_len))
        started_wall = time.time()
        started_at = time.perf_counter()
        response = await send_once(url, history, args.output_len)
        finished_at = time.perf_counter()
        finished_wall = time.time()
        history.extend(response.output_ids)
        rows.append(
            {
                "tag": args.tag,
                "phase": "probe",
                "round": probe_round,
                "started_wall": started_wall,
                "finished_wall": finished_wall,
                "started_at": started_at,
                "finished_at": finished_at,
                "prompt_len": response.prompt_len,
                "cached_tokens": response.cached_tokens,
                "generated_len": response.generated_len,
                "ttft": response.ttft,
                "latency": response.latency,
                "context_len_after": len(history),
            }
        )

    await flush_tree_perf_marker("after_probe")
    out = {
        "tag": args.tag,
        "target_context": args.target_context,
        "build_chunk": args.build_chunk,
        "probe_turns": args.probe_turns,
        "probe_len": args.probe_len,
        "output_len": args.output_len,
        "rows": rows,
        "markers": markers,
        "summary": {
            "build": summarize(rows, "build"),
            "probe": summarize(rows, "probe"),
        },
    }

    path = Path(args.log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out["summary"], indent=2))
    print(f"wrote {path}")


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
