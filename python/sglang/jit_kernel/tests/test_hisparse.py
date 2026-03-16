import os
import time

import pytest
import torch

from sglang.jit_kernel.hisparse import prepare_decode_metadata


def _create_test_inputs(
    num_reqs: int = 32,
    num_layers: int = 24,
    max_num_reqs: int = 64,
    max_context_len: int = 4096,
    device_buffer_size: int = 128,
    device: str = "cuda",
):
    req_pool_indices = torch.randperm(max_num_reqs, device=device, dtype=torch.int64)[
        :num_reqs
    ].contiguous()
    seq_lens = torch.randint(
        1, max_context_len, (num_reqs,), device=device, dtype=torch.int64
    )
    if num_reqs >= 4:
        seq_lens[:4] = torch.tensor(
            [1, device_buffer_size, device_buffer_size + 1, device_buffer_size + 7],
            device=device,
            dtype=torch.int64,
        )

    out_cache_loc = torch.randperm(
        max_context_len * 2, device=device, dtype=torch.int64
    )[:num_reqs].contiguous()

    req_to_host_pool = torch.full(
        (max_num_reqs, max_context_len), -1, dtype=torch.int64, device=device
    )
    req_to_device_buffer = torch.randint(
        0,
        max_context_len * 2,
        (max_num_reqs, device_buffer_size + 1),
        dtype=torch.int64,
        device=device,
    )
    req_device_buffer_token_locs = torch.full(
        (num_layers, max_num_reqs, device_buffer_size + 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    full_to_hisparse_device_index_mapping = torch.full(
        (max_context_len * 2 + 1,), -1, dtype=torch.int64, device=device
    )

    prev_pos = seq_lens - 2
    long_mask = (prev_pos >= 0) & (seq_lens > device_buffer_size)
    candidate_reqs = req_pool_indices[long_mask]
    candidate_pos = prev_pos[long_mask]
    if len(candidate_reqs) > 0:
        req_to_host_pool[candidate_reqs[::2], candidate_pos[::2]] = 7

    return {
        "seq_lens": seq_lens,
        "out_cache_loc": out_cache_loc,
        "req_pool_indices": req_pool_indices,
        "req_to_host_pool": req_to_host_pool,
        "req_to_device_buffer": req_to_device_buffer,
        "req_device_buffer_token_locs": req_device_buffer_token_locs,
        "full_to_hisparse_device_index_mapping": full_to_hisparse_device_index_mapping,
        "device_buffer_size": device_buffer_size,
    }


def _reference_prepare_decode_metadata(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_host_pool: torch.Tensor,
    req_to_device_buffer: torch.Tensor,
    req_device_buffer_token_locs: torch.Tensor,
    full_to_hisparse_device_index_mapping: torch.Tensor,
    device_buffer_size: int,
):
    reserved_buffer_loc = req_to_device_buffer[req_pool_indices, device_buffer_size].clone()

    short_reqs = seq_lens <= device_buffer_size
    if torch.any(short_reqs):
        reserved_buffer_loc[short_reqs] = req_to_device_buffer[
            req_pool_indices[short_reqs], seq_lens[short_reqs] - 1
        ]

    req_device_buffer_token_locs[
        :, req_pool_indices, device_buffer_size
    ] = reserved_buffer_loc.to(torch.int32)
    full_to_hisparse_device_index_mapping[out_cache_loc] = reserved_buffer_loc

    prev_pos = seq_lens - 2
    long_mask = (prev_pos >= 0) & (seq_lens > device_buffer_size)
    if not torch.any(long_mask):
        empty = req_pool_indices[:0]
        return empty, empty, req_to_device_buffer[:0, device_buffer_size]

    candidate_reqs = req_pool_indices[long_mask]
    candidate_pos = prev_pos[long_mask]
    needs_backup = req_to_host_pool[candidate_reqs, candidate_pos] < 0

    backup_req_indices = candidate_reqs[needs_backup]
    backup_positions = candidate_pos[needs_backup]
    device_locs = req_to_device_buffer[backup_req_indices, device_buffer_size]
    return backup_req_indices, backup_positions, device_locs


def _run_fused(data):
    backup_meta = torch.empty(
        (3, data["req_pool_indices"].numel()),
        dtype=data["req_pool_indices"].dtype,
        device=data["seq_lens"].device,
    )
    backup_count = torch.zeros(1, dtype=torch.int32, device=data["seq_lens"].device)

    prepare_decode_metadata(
        data["seq_lens"],
        data["out_cache_loc"],
        data["req_pool_indices"],
        data["req_to_host_pool"],
        data["req_to_device_buffer"],
        data["req_device_buffer_token_locs"],
        data["full_to_hisparse_device_index_mapping"],
        backup_meta[0],
        backup_meta[1],
        backup_meta[2],
        backup_count,
        data["device_buffer_size"],
    )
    count = int(backup_count.item())
    return (
        backup_meta[0, :count],
        backup_meta[1, :count],
        backup_meta[2, :count],
    )


def _to_sorted_tuples(*tensors):
    if len(tensors[0]) == 0:
        return []
    rows = torch.stack(tensors, dim=1).cpu().tolist()
    return sorted(tuple(int(v) for v in row) for row in rows)


def test_prepare_decode_metadata_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data_ref = _create_test_inputs()
    data_fused = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in data_ref.items()
    }

    ref = _reference_prepare_decode_metadata(
        data_ref["seq_lens"],
        data_ref["out_cache_loc"],
        data_ref["req_pool_indices"],
        data_ref["req_to_host_pool"],
        data_ref["req_to_device_buffer"],
        data_ref["req_device_buffer_token_locs"],
        data_ref["full_to_hisparse_device_index_mapping"],
        data_ref["device_buffer_size"],
    )
    fused = _run_fused(data_fused)
    torch.cuda.synchronize()

    assert torch.equal(
        data_ref["req_device_buffer_token_locs"], data_fused["req_device_buffer_token_locs"]
    )
    assert torch.equal(
        data_ref["full_to_hisparse_device_index_mapping"],
        data_fused["full_to_hisparse_device_index_mapping"],
    )
    assert _to_sorted_tuples(*ref) == _to_sorted_tuples(*fused)


def test_prepare_decode_metadata_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if os.environ.get("SGLANG_ENABLE_HISPARSE_PERF_TEST", "0") != "1":
        pytest.skip("Perf test disabled by default")

    data_ref = _create_test_inputs(
        num_reqs=128,
        num_layers=64,
        max_num_reqs=256,
        max_context_len=8192,
        device_buffer_size=2048,
    )
    data_fused = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in data_ref.items()
    }

    for _ in range(20):
        _reference_prepare_decode_metadata(
            data_ref["seq_lens"],
            data_ref["out_cache_loc"],
            data_ref["req_pool_indices"],
            data_ref["req_to_host_pool"],
            data_ref["req_to_device_buffer"],
            data_ref["req_device_buffer_token_locs"],
            data_ref["full_to_hisparse_device_index_mapping"],
            data_ref["device_buffer_size"],
        )
        _run_fused(data_fused)
    torch.cuda.synchronize()

    num_iters = 200

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        _reference_prepare_decode_metadata(
            data_ref["seq_lens"],
            data_ref["out_cache_loc"],
            data_ref["req_pool_indices"],
            data_ref["req_to_host_pool"],
            data_ref["req_to_device_buffer"],
            data_ref["req_device_buffer_token_locs"],
            data_ref["full_to_hisparse_device_index_mapping"],
            data_ref["device_buffer_size"],
        )
    torch.cuda.synchronize()
    ref_time = time.perf_counter() - start

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        _run_fused(data_fused)
    torch.cuda.synchronize()
    fused_time = time.perf_counter() - start

    print(
        f"hisparse prepare_decode_metadata ref={ref_time*1000:.3f}ms "
        f"fused={fused_time*1000:.3f}ms speedup={ref_time / fused_time:.2f}x"
    )
    assert fused_time < ref_time
