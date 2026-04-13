from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    HiCachePhase,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def __init__(self, cache, params):
        super().__init__(cache, params)
        allocator = cache.token_to_kv_pool_allocator
        # When SWA is present, only free full-attention KV here;
        # SWA KV will be freed by cascade via SWAComponent.evict_component.
        if ComponentType.SWA in cache.tree_components:
            self._free_full = allocator.full_attn_allocator.free
        else:
            self._free_full = allocator.free
        self._hicache_enabled = False  # set True when cache_controller is set

    def node_has_component_data(self, node: UnifiedTreeNode) -> bool:
        # Override so _for_each_component_lru includes Full in LRU operations
        return node.component_data[self.component_type].value is not None

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        # HiCache: evicted + backuped nodes are valid match boundaries
        return lambda node: (
            node.component_data[self.component_type].value is not None
            or node.backuped
        )

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        # Compute Full KV host hit length: walk from last_host_node up to
        # last_device_node, summing host_value lengths of evicted nodes.
        ct = self.component_type
        kv_host_hit = 0
        node = result.last_host_node
        while node is not result.last_device_node:
            full_host = node.component_data[ct].host_value
            if full_host is not None:
                kv_host_hit += len(full_host)
            node = node.parent
        if kv_host_hit > 0:
            return result._replace(
                host_hit_length=max(result.host_hit_length, kv_host_hit)
            )
        return result

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        # HiCache: split host_value if present
        child_cd = child.component_data[ct]
        if child_cd.host_value is not None:
            split_len = len(new_parent.key)
            new_parent.component_data[ct].host_value = child_cd.host_value[:split_len].clone()
            child_cd.host_value = child_cd.host_value[split_len:].clone()

    def evict_component(self, node: UnifiedTreeNode, is_leaf: bool) -> int:
        cd = node.component_data[self.component_type]
        freed = 0
        if cd.value is not None:
            self._free_full(cd.value)
            freed = len(cd.value)
            self.cache.component_evictable_size_[self.component_type] -= freed
            if not is_leaf:
                cd.value = None  # tombstone
        if is_leaf and cd.host_value is not None:
            if self.cache.cache_controller is not None:
                self.cache.cache_controller.evict_host(cd.host_value)
            cd.host_value = None
        self.cache.evictable_device_leaves.discard(node)
        return freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.num_tokens
        if self._hicache_enabled:
            # HiCache: heap-based eviction from device leaves
            heap = [
                (n.last_access_time, n) for n in self.cache.evictable_device_leaves
            ]
            heapq.heapify(heap)
            while tracker[self.component_type] < request and heap:
                _, x = heapq.heappop(heap)
                if x not in self.cache.evictable_device_leaves:
                    continue
                evicted = self.cache._evict_device_leaf(x)
                tracker[self.component_type] += evicted
                if (
                    x.parent is not None
                    and x.parent in self.cache.evictable_device_leaves
                ):
                    heapq.heappush(
                        heap, (x.parent.last_access_time, x.parent)
                    )
        else:
            # Original: LRU leaf eviction
            lru = self.cache.lru_lists[self.component_type]
            while tracker[self.component_type] < request:
                x = lru.get_leaf_lru_no_lock()
                if x is None:
                    break
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=True, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict host leaves to free KV host pool space."""
        heap = [
            (n.last_access_time, n)
            for n in self.cache.evictable_host_leaves
        ]
        heapq.heapify(heap)
        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x not in self.cache.evictable_host_leaves:
                continue
            num_evicted += self.cache._evict_host_leaf(x, tracker)
            if (
                x.parent is not None
                and x.parent in self.cache.evictable_host_leaves
            ):
                heapq.heappush(
                    heap, (x.parent.last_access_time, x.parent)
                )

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        ct = self.component_type
        root = self.cache.root_node
        delta = 0
        cur = node
        while cur != root:
            cd = cur.component_data[ct]
            # HiCache: skip evicted nodes (no device value to protect)
            if cd.value is None:
                cur = cur.parent
                continue
            if cd.lock_ref == 0:
                key_len = len(cd.value)
                self.cache.component_evictable_size_[ct] -= key_len
                self.cache.component_protected_size_[ct] += key_len
                delta += key_len
            cd.lock_ref += 1
            self.cache.evictable_device_leaves.discard(cur)
            cur = cur.parent
        result = IncLockRefResult(delta=delta, swa_uuid_for_lock=result.swa_uuid_for_lock)
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        ct = self.component_type
        root = self.cache.root_node
        cur = node
        while cur != root:
            cd = cur.component_data[ct]
            # HiCache: skip evicted nodes
            if cd.value is None:
                cur = cur.parent
                continue
            assert cd.lock_ref > 0
            if cd.lock_ref == 1:
                key_len = len(cd.value)
                self.cache.component_evictable_size_[ct] += key_len
                self.cache.component_protected_size_[ct] -= key_len
            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                self.cache._update_device_leaf_status(cur)
            cur = cur.parent

    # ---- HiCache Hooks ----

    @property
    def hicache_pool_name(self) -> Optional[PoolName]:
        return PoolName.KV

    def build_hicache_transfers(
        self, node: UnifiedTreeNode, phase: HiCachePhase, **kw
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == HiCachePhase.BACKUP:
            # Full KV backup is handled by the main flow
            # (write_backup → cache_controller.write on host_value directly).
            # No extra PoolTransfer needed.
            return None

        if phase == HiCachePhase.RESTORE:
            # Collect host_value from the evicted chain (root→leaf order)
            backed_up: list[torch.Tensor] = []
            cur = node
            while cur.evicted:
                cd = cur.component_data[ct]
                if cd.host_value is not None:
                    backed_up.append(cd.host_value)
                cur = cur.parent
            backed_up.reverse()
            if backed_up:
                return [
                    PoolTransfer(
                        name=PoolName.KV,
                        host_indices=torch.cat(backed_up),
                        device_indices=None,  # auto-allocated by controller
                    )
                ]
            return None

        return None

    def commit_hicache_transfer(
        self, node: UnifiedTreeNode, phase: HiCachePhase, **kw
    ) -> None:
        if phase == HiCachePhase.RESTORE:
            ct = self.component_type
            transfers = kw.get("transfers", [])
            my_transfers = [t for t in transfers if t.name == PoolName.KV]
            nodes_to_load = kw.get("nodes_to_load", [])

            if my_transfers and my_transfers[0].device_indices is not None and nodes_to_load:
                device_indices = my_transfers[0].device_indices
                offset = 0
                for n in nodes_to_load:
                    cd = n.component_data[ct]
                    if cd.host_value is not None:
                        n_len = len(cd.host_value)
                        cd.value = device_indices[offset : offset + n_len].clone()
                        offset += n_len
                        self.cache.lru_lists[ct].insert_mru(n)
                        self.cache.component_evictable_size_[ct] += n_len
                        self.cache._update_evictable_leaf_sets(n)

            # Update leaf status for the target node
            self.cache._update_device_leaf_status(node)
