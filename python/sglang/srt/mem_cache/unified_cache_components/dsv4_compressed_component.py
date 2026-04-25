from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import (
        DecLockRefParams,
        EvictParams,
        IncLockRefResult,
        InsertParams,
        InsertResult,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class DeepSeekV4CompressedComponent(TreeComponent):
    """DSV4_COMPRESSED component for V4 HiCache L2.

    This component carries NO per-node state.  It only participates via
    HiCache hooks, deriving C4/C128/C4_INDEXER page indices from the
    FULL component's logical indices at transfer time.
    """

    component_type = ComponentType.DSV4_COMPRESSED

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        super().__init__(cache, params)
        self.full_page_size = params.page_size  # typically 256

    # ---- Match ----
    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        # Compressed state follows FULL: always valid.
        return lambda node: True

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        # L2 host hit is compressed-ready, not decode-ready.
        # Subtract one full page (replay_window) so the scheduler replays
        # the tail tokens to rebuild SWA + compress_state.
        if result.host_hit_length > 0:
            replay_window = self.full_page_size
            adjusted = max(0, result.host_hit_length - replay_window)
            return result._replace(host_hit_length=adjusted)
        return result

    # ---- Node lifecycle: all no-ops ----
    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        pass  # no per-node data

    def evict_component(
        self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE
    ) -> tuple[int, int]:
        return 0, 0  # no data to free

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        pass  # does not drive independent eviction

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        return result  # no lock participation

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        pass

    # ---- HiCache hooks ----
    def build_hicache_transfers(
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw
    ) -> Optional[list[PoolTransfer]]:
        fps = self.full_page_size

        if phase == CacheTransferPhase.BACKUP_HOST:
            full_device = node.component_data[BASE_COMPONENT_TYPE].value
            if full_device is None:
                return None
            page_dev = torch.unique(full_device // fps)
            return [
                PoolTransfer(name=PoolName.C4, device_indices=page_dev),
                PoolTransfer(
                    name=PoolName.C4_INDEXER, device_indices=page_dev.clone()
                ),
                PoolTransfer(name=PoolName.C128, device_indices=page_dev.clone()),
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            # Walk evicted chain collecting FULL host_values
            full_host_parts: list[torch.Tensor] = []
            cur = node
            while cur.evicted:
                hv = cur.component_data[BASE_COMPONENT_TYPE].host_value
                if hv is not None:
                    full_host_parts.append(hv)
                cur = cur.parent
            full_host_parts.reverse()
            if not full_host_parts:
                return None
            full_host = torch.cat(full_host_parts)
            page_host = torch.unique(full_host // fps)
            return [
                PoolTransfer(name=PoolName.C4, host_indices=page_host),
                PoolTransfer(
                    name=PoolName.C4_INDEXER, host_indices=page_host.clone()
                ),
                PoolTransfer(name=PoolName.C128, host_indices=page_host.clone()),
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
    ) -> None:
        # BACKUP_HOST: no storage needed (derivable from FULL.host_value)
        # LOAD_BACK: no device-side mapping update needed
        #   (SWA mapping is rebuilt via replay, not via load-back)
        pass
