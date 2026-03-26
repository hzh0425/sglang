from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _build_backup_indices_kernel(
    req_pool_indices_ptr,
    skip_first_backup_ptr,
    backup_indices_ptr,
    backup_count_ptr,
    num_reqs,
):
    pid = tl.program_id(0)
    in_range = pid < num_reqs

    req_idx = tl.load(req_pool_indices_ptr + pid, mask=in_range, other=0)
    skip_flag = tl.load(skip_first_backup_ptr + req_idx, mask=in_range, other=1)
    tl.store(skip_first_backup_ptr + req_idx, 0, mask=in_range)

    should_backup = in_range & (skip_flag == 0)
    out_pos = tl.atomic_add(backup_count_ptr, 1, mask=should_backup)
    tl.store(backup_indices_ptr + out_pos, pid, mask=should_backup)


def build_backup_indices(
    req_pool_indices: torch.Tensor,
    skip_first_backup: torch.Tensor,
    backup_indices: torch.Tensor,
    backup_count: torch.Tensor,
) -> None:
    """Build per-batch backup positions and clear one-shot skip flags.

    Args:
        req_pool_indices: int64 CUDA tensor [batch_size].
        skip_first_backup: uint8 CUDA tensor [max_num_reqs], modified in-place.
        backup_indices: int32 CUDA tensor [>= batch_size], output positions.
        backup_count: int32 CUDA tensor [1], output count (must be zeroed by caller).
    """
    if not req_pool_indices.is_cuda:
        raise RuntimeError("req_pool_indices must be a CUDA tensor")
    if req_pool_indices.dtype != torch.int64:
        raise RuntimeError(
            f"req_pool_indices must be int64, got {req_pool_indices.dtype}"
        )
    if not skip_first_backup.is_cuda or skip_first_backup.dtype != torch.uint8:
        raise RuntimeError("skip_first_backup must be a CUDA uint8 tensor")
    if not backup_indices.is_cuda or backup_indices.dtype != torch.int32:
        raise RuntimeError("backup_indices must be a CUDA int32 tensor")
    if not backup_count.is_cuda or backup_count.dtype != torch.int32:
        raise RuntimeError("backup_count must be a CUDA int32 tensor")

    num_reqs = req_pool_indices.numel()
    if num_reqs == 0:
        return

    _build_backup_indices_kernel[(num_reqs,)](
        req_pool_indices.contiguous(),
        skip_first_backup,
        backup_indices,
        backup_count,
        num_reqs,
        num_warps=1,
        num_stages=1,
    )
