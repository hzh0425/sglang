from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.hybrid_cache.tree_component import (
    ComponentInsertResult,
    LockHandle,
    TreeComponent,
    gen_component_uuid,
)
from sglang.srt.mem_cache.radix_cache import RadixKey, maybe_bigram_convert

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_radix_cache import HybridTreeNode


class SWAComponent(TreeComponent):
    @property
    def name(self) -> str:
        return "swa"

    def transform_key_for_match(self, key: RadixKey) -> Optional[RadixKey]:
        key, _ = maybe_bigram_convert(self.cache.is_eagle, key)
        if self.cache.disable or len(key) == 0:
            return None
        if self.cache.page_size != 1:
            page_aligned_len = len(key) // self.cache.page_size * self.cache.page_size
            key = key[:page_aligned_len]
        return key

    def transform_key_value_for_insert(
        self, key: RadixKey, value: Optional[torch.Tensor]
    ) -> tuple[RadixKey, torch.Tensor]:
        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        return maybe_bigram_convert(self.cache.is_eagle, key, value)

    def init_match_walk_state(self) -> dict[str, Any]:
        return {"match_len_since_release": float("inf")}

    def is_valid_match_endpoint(
        self, node: "HybridTreeNode", state: dict[str, Any]
    ) -> bool:
        if node.component_value(self.name) is None:
            state["match_len_since_release"] = 0
            return False
        state["match_len_since_release"] += len(node.full_value)
        return state["match_len_since_release"] >= self.cache.sliding_window_size

    def compute_match_result_extras(
        self,
        params: MatchPrefixParams,
        last_node: "HybridTreeNode",
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> dict[str, Any]:
        return {}

    def update_component_on_insert_overlap(
        self,
        node: "HybridTreeNode",
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None:
        if params.prev_prefix_len >= total_prefix_len + prefix_len:
            return
        if node.component_value(self.name) is not None:
            self.cache.token_to_kv_pool_allocator.free(value_slice)
            return

        assert params.swa_evicted_seqlen % self.cache.page_size == 0
        assert node.component(self.name).lock_ref == 0

        if params.swa_evicted_seqlen <= total_prefix_len:
            self.cache.token_to_kv_pool_allocator.free(node.full_value[:prefix_len])
            node.full_value = value_slice.clone()
            node.set_component_value(self.name, node.full_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(node.component_value(self.name))
        elif params.swa_evicted_seqlen < total_prefix_len + prefix_len:
            start_update_idx = params.swa_evicted_seqlen - total_prefix_len
            self.cache.token_to_kv_pool_allocator.free(
                node.full_value[start_update_idx:prefix_len]
            )
            self.cache._split_node(node.key, node, start_update_idx)
            node.full_value = value_slice[start_update_idx:prefix_len].clone()
            self.cache.token_to_kv_pool_allocator.free(value_slice[:start_update_idx])
            node.set_component_value(self.name, node.full_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(node.component_value(self.name))
        else:
            self.cache.token_to_kv_pool_allocator.free(value_slice)

    def get_tombstone_prefix_len_for_insert(
        self, total_prefix_len: int, new_key_len: int, params: InsertParams
    ) -> int:
        if (
            params.swa_evicted_seqlen > total_prefix_len
            and params.swa_evicted_seqlen < total_prefix_len + new_key_len
        ):
            return params.swa_evicted_seqlen - total_prefix_len
        return 0

    def commit_insert_component_data(
        self, node: "HybridTreeNode", is_new_leaf: bool, params: InsertParams
    ) -> ComponentInsertResult:
        if is_new_leaf:
            node.set_component_value(self.name, node.full_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(node.component_value(self.name))
            return ComponentInsertResult()
        return ComponentInsertResult(reused_existing=True)

    def redistribute_on_node_split(self, new_parent: "HybridTreeNode", child: "HybridTreeNode"):
        child_value = child.component_value(self.name)
        new_parent.set_component_value(
            self.name,
            new_parent.full_value.clone() if child_value is not None else None,
        )
        new_parent.component(self.name).lock_ref = child.component(self.name).lock_ref
        if "component_uuid" in child.component(self.name).metadata:
            new_parent.component(self.name).metadata["component_uuid"] = child.component(
                self.name
            ).metadata["component_uuid"]
            child.component(self.name).metadata.pop("component_uuid", None)

    def node_has_component_data(self, node: "HybridTreeNode") -> bool:
        return node.component_value(self.name) is not None

    def evict_component_from_internal_node(self, node: "HybridTreeNode") -> int:
        self.cache.token_to_kv_pool_allocator.free_swa(node.full_value)
        freed = len(node.full_value)
        self.cache.component_evictable_size_[self.name] -= freed
        node.set_component_value(self.name, None)
        return freed

    def release_component_on_leaf_eviction(self, node: "HybridTreeNode") -> int:
        return len(node.component_value(self.name))

    def count_full_tokens_on_component_leaf_eviction(self) -> bool:
        return True

    def acquire_component_lock(self, node: "HybridTreeNode") -> Any:
        secondary_lock_size = 0
        stop_uuid = None
        cur = node
        while cur != self.cache.root_node and secondary_lock_size < self.cache.sliding_window_size:
            value = cur.component_value(self.name)
            assert value is not None
            if cur.component(self.name).lock_ref == 0:
                self.cache.component_evictable_size_[self.name] -= len(value)
                self.cache.component_protected_size_[self.name] += len(value)
            cur.component(self.name).lock_ref += 1
            secondary_lock_size += len(value)
            if secondary_lock_size >= self.cache.sliding_window_size:
                if "component_uuid" not in cur.component(self.name).metadata:
                    cur.component(self.name).metadata["component_uuid"] = gen_component_uuid()
                stop_uuid = cur.component(self.name).metadata["component_uuid"]
            cur = cur.parent
        return stop_uuid

    def release_component_lock(self, node: "HybridTreeNode", handle: Any) -> None:
        dec_secondary = True
        stop_uuid = handle
        while node != self.cache.root_node and dec_secondary:
            value = node.component_value(self.name)
            assert value is not None
            assert node.component(self.name).lock_ref > 0
            if node.component(self.name).lock_ref == 1:
                self.cache.component_evictable_size_[self.name] += len(value)
                self.cache.component_protected_size_[self.name] -= len(value)
            node.component(self.name).lock_ref -= 1
            if stop_uuid and node.component(self.name).metadata.get("component_uuid") == stop_uuid:
                dec_secondary = False
            node = node.parent

    def export_public_lock_handle(self, handle: LockHandle) -> Any:
        return handle.legacy_swa_uuid

    def import_public_lock_handle(self, handle: Any) -> LockHandle:
        if isinstance(handle, LockHandle):
            return handle
        return LockHandle(component_handles={self.name: handle})
