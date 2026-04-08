from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        return lambda node: True

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref

    def evict_component(self, node: UnifiedTreeNode, is_leaf: bool) -> int:
        ct = self.component_type
        self.cache.token_to_kv_pool_allocator.free(node.component_data[ct].value)
        freed = len(node.component_data[ct].value)
        self.cache.component_evictable_size_[ct] -= freed
        return freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        ct = self.component_type
        request = params.num_tokens
        lru = self.cache.lru_lists[ct]
        while tracker[ct] < request:
            x = lru.get_leaf_lru_no_lock()
            if x is None:
                break
            self.cache._evict_component_and_detach_lru(
                x, self, is_leaf=True, tracker=tracker
            )
            self.cache._cascade_evict(x, self, tracker)

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        ct = self.component_type
        cache = self.cache
        root = cache.root_node
        evictable = cache.component_evictable_size_
        protected = cache.component_protected_size_
        cur = node
        while cur != root:
            cd = cur.component_data[ct]
            if cd.lock_ref == 0:
                key_len = len(cur.key)
                evictable[ct] -= key_len
                protected[ct] += key_len
            cd.lock_ref += 1
            cur = cur.parent
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        ct = self.component_type
        cache = self.cache
        root = cache.root_node
        evictable = cache.component_evictable_size_
        protected = cache.component_protected_size_
        cur = node
        while cur != root:
            cd = cur.component_data[ct]
            assert cd.lock_ref > 0
            if cd.lock_ref == 1:
                key_len = len(cur.key)
                evictable[ct] += key_len
                protected[ct] -= key_len
            cd.lock_ref -= 1
            cur = cur.parent
