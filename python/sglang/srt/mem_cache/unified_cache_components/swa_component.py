from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    ComponentType,
    TreeComponent,
    next_component_uuid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class SWAComponent(TreeComponent):
    """Sliding window attention component.

    Each SWA node stores translated SWA pool indices as its component
    value, independent of the full attention indices on the same tree node.
    When SWA data is evicted from an internal node the node is tombstoned
    — its SWA component value becomes None while the full attention
    value stays intact.
    """

    component_type = ComponentType.SWA
    _simple_match_validator = False  # SWA validator is stateful (tracks window length)

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        assert isinstance(
            cache.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), f"SWAComponent requires SWATokenToKVPoolAllocator, got {type(cache.token_to_kv_pool_allocator)}"
        super().__init__(cache, params)
        self.sliding_window_size = params.sliding_window_size

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full_indices
        )

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        sliding_window_size = self.sliding_window_size
        ct = self.component_type
        state = {"len": float("inf")}

        def validator(node: UnifiedTreeNode) -> bool:
            if node.component_data[ct].value is None:
                state["len"] = 0
                return False
            state["len"] += len(node.key)
            return state["len"] >= sliding_window_size

        return validator

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> int:
        if params.prev_prefix_len >= total_prefix_len + prefix_len:
            return prefix_len

        ct = self.component_type
        cd = node.component_data[ct]
        is_tombstone = cd.value is None
        if not is_tombstone:
            return prefix_len

        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            cd.lock_ref == 0
        ), f"tombstone {ct} lock_ref should be 0, node {node.id}"
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{ct}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        base_cd = node._base_cd
        if swa_evicted_seqlen <= total_prefix_len:
            # Branch 1: entire value_slice is within SWA window — recover
            self.cache.token_to_kv_pool_allocator.free(base_cd.value)
            base_cd.value = value_slice.clone()
            swa_value = self._translate_full_to_swa(base_cd.value)
            cd.value = swa_value
            self.cache.lru_lists[ct].insert_mru(node)
            self.cache.component_evictable_size_[ct] += len(swa_value)
            return 0
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # Branch 2: value_slice[start_idx:] is within SWA window — partial recover
            start_idx = swa_evicted_seqlen - total_prefix_len
            self.cache.token_to_kv_pool_allocator.free(base_cd.value[start_idx:])
            self.cache._split_node(node.key, node, start_idx)
            base_cd = node._base_cd  # re-fetch after split
            base_cd.value = value_slice[start_idx:].clone()
            swa_value = self._translate_full_to_swa(base_cd.value)
            cd.value = swa_value
            self.cache.lru_lists[ct].insert_mru(node)
            self.cache.component_evictable_size_[ct] += len(swa_value)
            return start_idx
        else:
            # Branch 3: entire value_slice is outside SWA window — not consumed
            return prefix_len

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        return params.swa_evicted_seqlen >= total_prefix_len + key_len

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        if not is_new_leaf:
            return

        ct = self.component_type
        node_start = result.prefix_len
        split_pos = params.swa_evicted_seqlen - node_start

        if split_pos <= 0:
            swa_value = self._translate_full_to_swa(node._base_cd.value)
            node.component_data[ct].value = swa_value
            self.cache.lru_lists[ct].insert_mru(node)
            self.cache.component_evictable_size_[ct] += len(swa_value)
        elif split_pos < len(node.key):
            # Node straddles the SWA eviction boundary
            # Split into parent (tombstone, no SWA) and child (with SWA)
            # After _split_node, `node` becomes the child
            self.cache._split_node(node.key, node, split_pos)
            swa_value = self._translate_full_to_swa(node._base_cd.value)
            node.component_data[ct].value = swa_value
            self.cache.lru_lists[ct].insert_mru(node)
            self.cache.component_evictable_size_[ct] += len(swa_value)

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        parent_cd = new_parent.component_data[ct]
        child_cd = child.component_data[ct]
        parent_cd.lock_ref = child_cd.lock_ref

        child_swa_value = child_cd.value
        if child_swa_value is not None:
            split_len = len(new_parent.key)
            parent_cd.value = child_swa_value[:split_len].clone()
            child_cd.value = child_swa_value[split_len:].clone()
        else:
            parent_cd.value = None

        # parent inherits the swa_uuid from child for swa lock ref
        parent_cd.metadata["uuid"] = child_cd.metadata.get("uuid")
        child_cd.metadata.pop("uuid", None)

    def evict_component(self, node: UnifiedTreeNode, is_leaf: bool) -> int:
        ct = self.component_type
        cd = node.component_data[ct]
        swa_value = cd.value
        if swa_value is None:
            return 0
        # free_swa(full_value) uses the mapping guard to avoid double-free
        self.cache.token_to_kv_pool_allocator.free_swa(node._base_cd.value)
        freed = len(swa_value)
        self.cache.component_evictable_size_[ct] -= freed
        if not is_leaf:
            cd.value = None
        return freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 1

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        ct = self.component_type
        request = params.swa_num_tokens
        lru = self.cache.lru_lists[ct]
        x = lru.get_lru_no_lock()
        while (
            tracker[ct] < request and x is not None and lru.in_list(x)
        ):
            assert x.component_data[ct].value is not None
            if len(x.children) > 0:
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=False, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next
            else:
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=True, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = lru.get_lru_no_lock()

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        ct = self.component_type
        sliding_window_size = self.sliding_window_size
        swa_lock_size = 0
        swa_uuid_for_lock = None
        root = self.cache.root_node
        evictable = self.cache.component_evictable_size_
        protected = self.cache.component_protected_size_

        cur = node
        while cur != root and swa_lock_size < sliding_window_size:
            cd = cur.component_data[ct]
            assert (
                cd.value is not None
            ), f"acquire_component_lock({ct}) on tombstoned node {cur.id}"
            if cd.lock_ref == 0:
                key_len = len(cur.key)
                evictable[ct] -= key_len
                protected[ct] += key_len
            cd.lock_ref += 1
            swa_lock_size += len(cur.key)
            if swa_lock_size >= sliding_window_size:
                if cd.metadata.get("uuid") is None:
                    cd.metadata["uuid"] = next_component_uuid()
                swa_uuid_for_lock = cd.metadata["uuid"]
            cur = cur.parent

        result.swa_uuid_for_lock = swa_uuid_for_lock
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        ct = self.component_type
        swa_uuid_for_lock = params.swa_uuid_for_lock if params else None
        dec_swa = True
        root = self.cache.root_node
        evictable = self.cache.component_evictable_size_
        protected = self.cache.component_protected_size_

        cur = node
        while cur != root and dec_swa:
            cd = cur.component_data[ct]
            assert (
                cd.value is not None
            ), f"release_component_lock({ct}) on tombstoned node {cur.id}"
            assert (
                cd.lock_ref > 0
            ), f"release_component_lock({ct}) on node with lock_ref=0, node {cur.id}"
            if cd.lock_ref == 1:
                key_len = len(cur.key)
                evictable[ct] += key_len
                protected[ct] -= key_len
            cd.lock_ref -= 1
            if swa_uuid_for_lock and cd.metadata.get("uuid") == swa_uuid_for_lock:
                dec_swa = False
            cur = cur.parent

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        if is_finished:
            insert_params.swa_evicted_seqlen = req.swa_evicted_seqlen
        return None
