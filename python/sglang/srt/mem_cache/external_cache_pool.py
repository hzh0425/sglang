from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)


class LogicalDevicePool:
    """Index anchor with no physical cache payload."""

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.kv_buffer = None

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return []


class PagedDevicePool:
    """Read/write view of page-row GPU buffers used by direct connectors."""

    def __init__(
        self,
        buffers: Sequence[torch.Tensor],
        *,
        slot_page_size: int,
    ):
        if slot_page_size <= 0:
            raise ValueError(f"slot_page_size must be positive, got {slot_page_size}")
        self.kv_buffer = list(buffers)
        self.page_size = slot_page_size
        self.slot_page_size = slot_page_size
        for buffer in self.kv_buffer:
            if buffer.ndim < 2 or not buffer.is_contiguous():
                raise ValueError(
                    "Direct external cache buffers must be contiguous page rows, "
                    f"got shape={tuple(buffer.shape)}"
                )

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def get_page_buffer_meta(
        self, indices: torch.Tensor
    ) -> tuple[list[int], list[int]]:
        flat = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if flat.numel() % self.slot_page_size != 0:
            raise ValueError(
                "Direct external cache indices must be page-aligned, "
                f"got {flat.numel()} indices for page size {self.slot_page_size}"
            )

        pages = flat.reshape(-1, self.slot_page_size)
        first_slots = pages[:, 0]
        expected = first_slots[:, None] + torch.arange(self.slot_page_size)
        if not torch.equal(pages, expected):
            raise ValueError(
                "Direct external cache requires contiguous slots within each page"
            )
        rows = (first_slots // self.slot_page_size).tolist()

        pointers: list[int] = []
        sizes: list[int] = []
        for row in rows:
            for buffer in self.kv_buffer:
                if not 0 <= row < buffer.shape[0]:
                    raise ValueError(
                        f"Page row {row} exceeds buffer shape {tuple(buffer.shape)}"
                    )
                page = buffer[row]
                pointers.append(page.data_ptr())
                sizes.append(page.numel() * page.element_size())
        return pointers, sizes


@dataclass(frozen=True)
class ExternalPoolStack:
    anchor: LogicalDevicePool
    pools: dict[PoolName, PagedDevicePool]
    sidecars: tuple[SidecarPoolSpec, ...]


def build_deepseek_v4_external_pool_stack(kvcache, page_size: int) -> ExternalPoolStack:
    if not getattr(kvcache, "_unified_kv", False):
        raise ValueError(
            "Direct Mooncake currently requires DeepSeek V4 unified-kv layout"
        )

    pools: dict[PoolName, PagedDevicePool] = {}

    c4_buffers, _ = kvcache.unified_region_buffers(4)
    if c4_buffers:
        pools[PoolName.DEEPSEEK_V4_C4] = PagedDevicePool(
            c4_buffers, slot_page_size=page_size
        )
        pools[PoolName.DEEPSEEK_V4_C4_INDEXER] = PagedDevicePool(
            kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
            slot_page_size=page_size,
        )

    c128_buffers, _ = kvcache.unified_region_buffers(128)
    if c128_buffers:
        pools[PoolName.DEEPSEEK_V4_C128] = PagedDevicePool(
            c128_buffers, slot_page_size=page_size
        )

    sidecars = tuple(
        SidecarPoolSpec(
            pool_name=name,
            indices_from_pool=PoolName.KV,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        for name in (
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
        )
        if name in pools
    )
    if not sidecars:
        raise ValueError("DeepSeek V4 direct Mooncake found no compressed KV pools")

    return ExternalPoolStack(
        anchor=LogicalDevicePool(page_size),
        pools=pools,
        sidecars=sidecars,
    )
