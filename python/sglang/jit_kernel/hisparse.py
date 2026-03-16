from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
) -> Module:
    template_args = make_cpp_args(block_size, num_top_k, hot_buffer_size, is_mla)
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            ),
            (
                "load_cache_to_device_buffer_fused",
                f"load_cache_to_device_buffer_fused<{template_args}>",
            ),
        ],
    )


def _get_num_real_reqs(
    top_k_tokens: torch.Tensor,
    num_real_reqs: torch.Tensor | None,
) -> torch.Tensor:
    if num_real_reqs is not None:
        return num_real_reqs
    return torch.tensor(
        [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    residency_map: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    transfer_tasks_src: torch.Tensor,
    transfer_tasks_dst: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True
    )
    empty = torch.empty(0)

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        residency_map,
        req_pool_indices,
        seq_lens,
        lru_slots,
        transfer_tasks_src,
        transfer_tasks_dst,
        _get_num_real_reqs(top_k_tokens, num_real_reqs),
        page_size,
        item_size_bytes,
    )


def load_cache_to_device_buffer_mla_fused(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    residency_map: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True
    )
    empty = torch.empty(0)

    module.load_cache_to_device_buffer_fused(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        residency_map,
        req_pool_indices,
        seq_lens,
        lru_slots,
        _get_num_real_reqs(top_k_tokens, num_real_reqs),
        page_size,
        item_size_bytes,
    )


def prepare_decode_metadata(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_host_pool: torch.Tensor,
    req_to_device_buffer: torch.Tensor,
    req_device_buffer_token_locs: torch.Tensor,
    full_to_hisparse_device_index_mapping: torch.Tensor,
    backup_req_indices: torch.Tensor,
    backup_positions: torch.Tensor,
    backup_device_locs: torch.Tensor,
    backup_count: torch.Tensor,
    device_buffer_size: int,
    block_size: int = 256,
) -> None:
    del block_size

    backup_count.zero_()
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
        return

    candidate_reqs = req_pool_indices[long_mask]
    candidate_pos = prev_pos[long_mask]
    needs_backup = req_to_host_pool[candidate_reqs, candidate_pos] < 0
    if not torch.any(needs_backup):
        return

    selected_reqs = candidate_reqs[needs_backup]
    selected_pos = candidate_pos[needs_backup]
    selected_locs = req_to_device_buffer[selected_reqs, device_buffer_size]
    count = selected_reqs.numel()

    backup_req_indices[:count] = selected_reqs
    backup_positions[:count] = selected_pos
    backup_device_locs[:count] = selected_locs
    backup_count[0] = count
