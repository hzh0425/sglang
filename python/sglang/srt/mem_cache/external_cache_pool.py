from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType


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
    component_pools: dict[ComponentType, PoolName] = field(default_factory=dict)


def _state_page_views(state_pools: Sequence) -> list[torch.Tensor]:
    views = []
    for pool in state_pools:
        state = pool.kv_score_buffer.kv_score
        ring_size = pool.ring_size
        usable_slots = state.shape[0] // ring_size * ring_size
        views.append(
            state.view(torch.uint8)[:usable_slots].reshape(
                -1, ring_size * state[0].nbytes
            )
        )
    return views


def build_deepseek_v4_external_pool_stack(kvcache, page_size: int) -> ExternalPoolStack:
    pools: dict[PoolName, PagedDevicePool] = {}
    component_pools: dict[ComponentType, PoolName] = {}
    is_unified_kv = getattr(kvcache, "_unified_kv", False)

    if is_unified_kv:
        c4_buffers, _ = kvcache.unified_region_buffers(4)
        c128_buffers, _ = kvcache.unified_region_buffers(128)
    else:
        if kvcache.swa_page_size != page_size:
            raise ValueError(
                "Direct Mooncake requires DeepSeek V4 SWA and Full to use the "
                f"same page size, got swa={kvcache.swa_page_size}, full={page_size}"
            )
        pools[PoolName.SWA] = PagedDevicePool(
            kvcache.swa_kv_pool.kv_buffer,
            slot_page_size=page_size,
        )
        component_pools[ComponentType.SWA] = PoolName.SWA
        c4_buffers = kvcache.c4_kv_pool.kv_buffer
        c128_buffers = kvcache.c128_kv_pool.kv_buffer

    if c4_buffers:
        pools[PoolName.DEEPSEEK_V4_C4] = PagedDevicePool(
            c4_buffers, slot_page_size=page_size
        )
        pools[PoolName.DEEPSEEK_V4_C4_INDEXER] = PagedDevicePool(
            kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
            slot_page_size=page_size,
        )

    if c128_buffers:
        pools[PoolName.DEEPSEEK_V4_C128] = PagedDevicePool(
            c128_buffers, slot_page_size=page_size
        )

    if not is_unified_kv:
        c4_layers = [
            kvcache.start_layer + local_layer
            for local_layer, item in enumerate(
                kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
            )
            if item.compress_ratio == 4
        ]
        c4_state_pools = [kvcache.compress_state_pools[i] for i in c4_layers]
        c4_indexer_state_pools = [
            kvcache.indexer_compress_state_pools[i] for i in c4_layers
        ]
        if c4_state_pools:
            pools[PoolName.DEEPSEEK_V4_C4_STATE] = PagedDevicePool(
                _state_page_views(c4_state_pools),
                slot_page_size=page_size,
            )
            pools[PoolName.DEEPSEEK_V4_C4_INDEXER_STATE] = PagedDevicePool(
                _state_page_views(c4_indexer_state_pools),
                slot_page_size=page_size,
            )

    sidecars = tuple(
        SidecarPoolSpec(
            pool_name=name,
            indices_from_pool=source,
            hit_policy=policy,
        )
        for name, source, policy in (
            (PoolName.DEEPSEEK_V4_C4, PoolName.KV, PoolHitPolicy.ALL_PAGES),
            (
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.KV,
                PoolHitPolicy.ALL_PAGES,
            ),
            (PoolName.DEEPSEEK_V4_C128, PoolName.KV, PoolHitPolicy.ALL_PAGES),
            (
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.SWA,
                PoolHitPolicy.TRAILING_PAGES,
            ),
            (
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                PoolName.SWA,
                PoolHitPolicy.TRAILING_PAGES,
            ),
        )
        if name in pools
    )
    if not sidecars:
        raise ValueError("DeepSeek V4 direct Mooncake found no compressed KV pools")

    return ExternalPoolStack(
        anchor=LogicalDevicePool(page_size),
        pools=pools,
        sidecars=sidecars,
        component_pools=component_pools,
    )
