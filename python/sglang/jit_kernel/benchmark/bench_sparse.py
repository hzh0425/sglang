import argparse
import time

import torch

from sglang.jit_kernel.hisparse import (
    load_cache_to_device_buffer_mla,
    load_cache_to_device_buffer_mla_fused,
)


def build_inputs(
    batch_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    max_seq_len: int,
    feature_dim: int,
    dtype: torch.dtype,
):
    device = "cuda"
    max_seq_len = max(max_seq_len, hot_buffer_size + 97)
    padded_buffer_size = hot_buffer_size + 1

    host_cache_locs = torch.empty(
        (batch_size, max_seq_len), dtype=torch.int64, device=device
    )
    device_buffer_locs = torch.empty(
        (batch_size, padded_buffer_size), dtype=torch.int32, device=device
    )
    device_buffer_tokens = torch.empty(
        (batch_size, padded_buffer_size), dtype=torch.int32, device=device
    )
    lru_slots = (
        torch.arange(hot_buffer_size, dtype=torch.int16, device=device)
        .view(batch_size, hot_buffer_size)
        .clone()
    )
    residency_map = torch.full(
        (batch_size, max_seq_len), -1, dtype=torch.int16, device=device
    )
    top_k_tokens = torch.empty(
        (batch_size, num_top_k), dtype=torch.int32, device=device
    )
    top_k_device_locs = torch.full(
        (batch_size, num_top_k), -1, dtype=torch.int32, device=device
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int64, device=device)
    num_real_reqs = torch.tensor([batch_size], dtype=torch.int32, device=device)

    total_host_locs = batch_size * max_seq_len
    total_device_locs = batch_size * padded_buffer_size
    host_cache = torch.empty((total_host_locs, feature_dim), dtype=dtype, device=device)
    device_buffer = torch.empty(
        (total_device_locs, feature_dim), dtype=dtype, device=device
    )

    host_values = torch.arange(total_host_locs, dtype=torch.float32, device=device).view(
        -1, 1
    )
    host_cache.copy_(host_values.repeat(1, feature_dim).to(dtype))

    for b in range(batch_size):
        host_base = b * max_seq_len
        host_cache_locs[b] = torch.arange(
            host_base, host_base + max_seq_len, dtype=torch.int64, device=device
        )

        device_base = b * padded_buffer_size
        device_buffer_locs[b] = torch.arange(
            device_base,
            device_base + padded_buffer_size,
            dtype=torch.int32,
            device=device,
        )

        device_buffer_tokens[b, :hot_buffer_size] = torch.arange(
            hot_buffer_size, dtype=torch.int32, device=device
        )
        device_buffer_tokens[b, hot_buffer_size] = max_seq_len - 1

        device_buffer[device_base : device_base + hot_buffer_size].copy_(
            host_cache[host_base : host_base + hot_buffer_size]
        )
        device_buffer[device_base + hot_buffer_size].copy_(
            host_cache[host_base + max_seq_len - 1]
        )

        miss_count = min(64, max(1, num_top_k // 8))
        keep_tokens = torch.arange(
            num_top_k - miss_count, dtype=torch.int32, device=device
        )
        new_tokens = torch.arange(
            hot_buffer_size + 32,
            hot_buffer_size + 32 + miss_count,
            dtype=torch.int32,
            device=device,
        )
        curr_tokens = torch.cat([keep_tokens, new_tokens], dim=0)
        curr_tokens[-1] = max_seq_len - 1
        top_k_tokens[b] = curr_tokens[torch.randperm(num_top_k, device=device)]

    transfer_tasks_src = torch.full(
        (batch_size * (num_top_k + 1),), -1, dtype=torch.int64, device=device
    )
    transfer_tasks_dst = torch.full(
        (batch_size * (num_top_k + 1),), -1, dtype=torch.int64, device=device
    )

    return {
        "top_k_tokens": top_k_tokens,
        "device_buffer_tokens": device_buffer_tokens,
        "host_cache_locs": host_cache_locs,
        "device_buffer_locs": device_buffer_locs,
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "top_k_device_locs": top_k_device_locs,
        "residency_map": residency_map,
        "req_pool_indices": req_pool_indices,
        "seq_lens": seq_lens,
        "lru_slots": lru_slots,
        "transfer_tasks_src": transfer_tasks_src,
        "transfer_tasks_dst": transfer_tasks_dst,
        "num_real_reqs": num_real_reqs,
        "item_size_bytes": feature_dim * torch.tensor([], dtype=dtype).element_size(),
    }


def clone_inputs(inputs):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def run_impl(
    impl: str,
    inputs,
    num_top_k: int,
    hot_buffer_size: int,
    block_size: int,
):
    if impl == "split":
        load_cache_to_device_buffer_mla(
            top_k_tokens=inputs["top_k_tokens"],
            device_buffer_tokens=inputs["device_buffer_tokens"],
            host_cache_locs=inputs["host_cache_locs"],
            device_buffer_locs=inputs["device_buffer_locs"],
            host_cache=inputs["host_cache"],
            device_buffer=inputs["device_buffer"],
            top_k_device_locs=inputs["top_k_device_locs"],
            residency_map=inputs["residency_map"],
            req_pool_indices=inputs["req_pool_indices"],
            seq_lens=inputs["seq_lens"],
            lru_slots=inputs["lru_slots"],
            transfer_tasks_src=inputs["transfer_tasks_src"],
            transfer_tasks_dst=inputs["transfer_tasks_dst"],
            item_size_bytes=inputs["item_size_bytes"],
            num_top_k=num_top_k,
            hot_buffer_size=hot_buffer_size,
            page_size=1,
            block_size=block_size,
            num_real_reqs=inputs["num_real_reqs"],
        )
        return

    if impl == "fused":
        load_cache_to_device_buffer_mla_fused(
            top_k_tokens=inputs["top_k_tokens"],
            device_buffer_tokens=inputs["device_buffer_tokens"],
            host_cache_locs=inputs["host_cache_locs"],
            device_buffer_locs=inputs["device_buffer_locs"],
            host_cache=inputs["host_cache"],
            device_buffer=inputs["device_buffer"],
            top_k_device_locs=inputs["top_k_device_locs"],
            residency_map=inputs["residency_map"],
            req_pool_indices=inputs["req_pool_indices"],
            seq_lens=inputs["seq_lens"],
            lru_slots=inputs["lru_slots"],
            item_size_bytes=inputs["item_size_bytes"],
            num_top_k=num_top_k,
            hot_buffer_size=hot_buffer_size,
            page_size=1,
            block_size=block_size,
            num_real_reqs=inputs["num_real_reqs"],
        )
        return

    raise ValueError(f"Unknown impl: {impl}")


def check_correctness(ref_inputs, test_inputs):
    assert torch.equal(
        ref_inputs["top_k_device_locs"], test_inputs["top_k_device_locs"]
    ), "top_k_device_locs mismatch"
    assert torch.equal(
        ref_inputs["device_buffer_tokens"], test_inputs["device_buffer_tokens"]
    ), "device_buffer_tokens mismatch"
    assert torch.equal(ref_inputs["lru_slots"], test_inputs["lru_slots"]), "lru mismatch"

    unique_locs = torch.unique(ref_inputs["top_k_device_locs"])
    unique_locs = unique_locs[unique_locs >= 0].to(torch.int64)
    assert torch.equal(
        ref_inputs["device_buffer"][unique_locs], test_inputs["device_buffer"][unique_locs]
    ), "device buffer payload mismatch"

    _check_payload_matches_host(ref_inputs)
    _check_payload_matches_host(test_inputs)


def _check_payload_matches_host(inputs):
    batch_size = inputs["top_k_tokens"].shape[0]
    num_top_k = inputs["top_k_tokens"].shape[1]

    for batch_id in range(batch_size):
        for token_idx in range(num_top_k):
            token_pos = int(inputs["top_k_tokens"][batch_id, token_idx].item())
            device_loc = int(inputs["top_k_device_locs"][batch_id, token_idx].item())
            host_loc = int(inputs["host_cache_locs"][batch_id, token_pos].item())
            assert device_loc >= 0, "invalid device loc"
            assert torch.equal(
                inputs["device_buffer"][device_loc],
                inputs["host_cache"][host_loc],
            ), (
                f"payload mismatch: batch={batch_id} token_pos={token_pos} "
                f"device_loc={device_loc} host_loc={host_loc}"
            )


def benchmark_impl(
    impl: str,
    base_inputs,
    num_top_k: int,
    hot_buffer_size: int,
    block_size: int,
    warmup: int,
    rounds: int,
) -> float:
    timings = []
    for iter_idx in range(warmup + rounds):
        inputs = clone_inputs(base_inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_impl(impl, inputs, num_top_k, hot_buffer_size, block_size)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if iter_idx >= warmup:
            timings.append(elapsed_ms)
    return sum(timings) / len(timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--hot-buffer-size", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--feature-dim", type=int, default=576)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    base_inputs = build_inputs(
        batch_size=args.batch_size,
        num_top_k=args.top_k,
        hot_buffer_size=args.hot_buffer_size,
        max_seq_len=args.max_seq_len,
        feature_dim=args.feature_dim,
        dtype=torch.float16,
    )

    fused_inputs = clone_inputs(base_inputs)
    split_inputs = clone_inputs(base_inputs)
    run_impl(
        "fused",
        fused_inputs,
        args.top_k,
        args.hot_buffer_size,
        args.block_size,
    )
    run_impl(
        "split",
        split_inputs,
        args.top_k,
        args.hot_buffer_size,
        args.block_size,
    )
    torch.cuda.synchronize()
    check_correctness(fused_inputs, split_inputs)

    fused_ms = benchmark_impl(
        "fused",
        base_inputs,
        args.top_k,
        args.hot_buffer_size,
        args.block_size,
        args.warmup,
        args.rounds,
    )
    split_ms = benchmark_impl(
        "split",
        base_inputs,
        args.top_k,
        args.hot_buffer_size,
        args.block_size,
        args.warmup,
        args.rounds,
    )

    print(
        f"fused_avg_ms={fused_ms:.3f} split_avg_ms={split_ms:.3f} "
        f"speedup={fused_ms / split_ms:.3f}x"
    )


if __name__ == "__main__":
    main()
