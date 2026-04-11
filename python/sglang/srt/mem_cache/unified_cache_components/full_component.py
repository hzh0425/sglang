from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
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
        # HiCache: device leaf set for heap-based eviction
        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
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

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component_data[self.component_type].lock_ref = child.component_data[
            self.component_type
        ].lock_ref
        # HiCache: split host_value if present
        if child.host_value is not None:
            split_len = len(new_parent.key)
            new_parent.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

    def evict_component(self, node: UnifiedTreeNode, is_leaf: bool) -> int:
        cd = node.component_data[self.component_type]
        freed = 0
        if cd.value is not None:
            self._free_full(cd.value)
            freed = len(cd.value)
            self.cache.component_evictable_size_[self.component_type] -= freed
            if not is_leaf:
                cd.value = None  # tombstone
        if is_leaf and node.host_value is not None:
            # Full host data is stored at tree-level node.host_value
            if self.cache.cache_controller is not None:
                self.cache.cache_controller.evict_host(node.host_value)
            node.host_value = None
        self.evictable_device_leaves.discard(node)
        return freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def _update_device_leaf_status(self, node: UnifiedTreeNode) -> None:
        """Update whether a node qualifies as an evictable device leaf."""
        cd = node.component_data[self.component_type]
        if node is self.cache.root_node or cd.value is None or cd.lock_ref > 0:
            self.evictable_device_leaves.discard(node)
            return
        for child in node.children.values():
            if child.component_data[self.component_type].value is not None:
                self.evictable_device_leaves.discard(node)
                return
        self.evictable_device_leaves.add(node)

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.num_tokens
        if self._hicache_enabled:
            # HiCache: heap-based eviction from device leaves
            heap = [
                (n.last_access_time, n) for n in self.evictable_device_leaves
            ]
            heapq.heapify(heap)
            while tracker[self.component_type] < request and heap:
                _, x = heapq.heappop(heap)
                if x not in self.evictable_device_leaves:
                    continue
                evicted = self.cache._evict_device_leaf(x)
                tracker[self.component_type] += evicted
                if (
                    x.parent is not None
                    and x.parent in self.evictable_device_leaves
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
            self.evictable_device_leaves.discard(cur)
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
            if cd.lock_ref == 0 and self._hicache_enabled:
                self._update_device_leaf_status(cur)
            cur = cur.parent

    # ---- HiCache Hooks ----

    @property
    def hicache_pool_name(self) -> Optional[PoolName]:
        return PoolName.KV

    def build_hicache_transfers(
        self, node: UnifiedTreeNode, phase: HiCachePhase, **kw
    ) -> Optional[list[PoolTransfer]]:
        # Full KV transfers are handled by the main flow (write_backup/load_back)
        # via cache_controller.write/load directly on node.host_value.
        # No extra PoolTransfer needed for Full.
        return None

    def commit_hicache_transfer(
        self, node: UnifiedTreeNode, phase: HiCachePhase, **kw
    ) -> None:
        if phase == HiCachePhase.RESTORE:
            # After restore, update leaf status for restored nodes
            self._update_device_leaf_status(node)
