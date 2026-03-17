from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.hybrid_cache.tree_component import (
    ComponentInsertResult,
    TreeComponent,
    get_last_access_time,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_radix_cache import HybridTreeNode


class MambaComponent(TreeComponent):
    @property
    def name(self) -> str:
        return "mamba"

    def transform_key_for_match(self, key: RadixKey) -> Optional[RadixKey]:
        if self.cache.disable or len(key) == 0:
            return None
        return key

    def transform_key_value_for_insert(
        self, key: RadixKey, value: Optional[torch.Tensor]
    ) -> tuple[RadixKey, torch.Tensor]:
        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        return key, value

    def init_match_walk_state(self) -> dict[str, Any]:
        return {}

    def is_valid_match_endpoint(
        self, node: "HybridTreeNode", state: dict[str, Any]
    ) -> bool:
        return node.component_value(self.name) is not None

    def compute_match_result_extras(
        self,
        params: MatchPrefixParams,
        last_node: "HybridTreeNode",
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> dict[str, Any]:
        cow_mamba = params.cow_mamba
        req = params.req

        if len(value_chunks) > best_value_len:
            chunk_size = get_global_server_args().mamba_cache_chunk_size
            aligned_seqlen = (sum(len(v) for v in value_chunks) // chunk_size) * chunk_size
            branching_seqlen = aligned_seqlen if aligned_seqlen > 0 else None
        else:
            branching_seqlen = None

        mamba_value = last_node.component_value(self.name)
        if cow_mamba and mamba_value is not None:
            assert req is not None
            if req.mamba_pool_idx is None:
                dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                if dst_index is None:
                    self.cache.inc_lock_ref(last_node)
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                    self.cache.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                self.cache.req_to_token_pool.mamba_pool.copy_from(mamba_value, dst_index)
                req.mamba_pool_idx = dst_index[0]
            else:
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.cache.req_to_token_pool.mamba_pool.copy_from(mamba_value, dst_index)

        return {"mamba_branching_seqlen": branching_seqlen}

    def update_component_on_insert_overlap(
        self,
        node: "HybridTreeNode",
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None:
        return

    def get_tombstone_prefix_len_for_insert(
        self, total_prefix_len: int, new_key_len: int, params: InsertParams
    ) -> int:
        return 0

    def commit_insert_component_data(
        self, node: "HybridTreeNode", is_new_leaf: bool, params: InsertParams
    ) -> ComponentInsertResult:
        assert params.mamba_value is not None
        if is_new_leaf:
            node.set_component_value(self.name, params.mamba_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(params.mamba_value)
            return ComponentInsertResult()
        if node.component_value(self.name) is None:
            node.set_component_value(self.name, params.mamba_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(params.mamba_value)
            node.last_access_time = get_last_access_time()
            return ComponentInsertResult(reused_existing=False)
        self.cache.lru_lists[self.name].reset_node_mru(node)
        node.last_access_time = get_last_access_time()
        return ComponentInsertResult(reused_existing=True)

    def redistribute_on_node_split(self, new_parent: "HybridTreeNode", child: "HybridTreeNode"):
        new_parent.set_component_value(self.name, None)
        new_parent.component(self.name).lock_ref = 0

    def node_has_component_data(self, node: "HybridTreeNode") -> bool:
        return node.component_value(self.name) is not None

    def evict_component_from_internal_node(self, node: "HybridTreeNode") -> int:
        value = node.component_value(self.name)
        self.cache.req_to_token_pool.mamba_pool.free(value)
        freed = len(value)
        self.cache.component_evictable_size_[self.name] -= freed
        node.set_component_value(self.name, None)
        return freed

    def release_component_on_leaf_eviction(self, node: "HybridTreeNode") -> int:
        value = node.component_value(self.name)
        self.cache.req_to_token_pool.mamba_pool.free(value)
        return len(value)

    def acquire_component_lock(self, node: "HybridTreeNode") -> Any:
        value = node.component_value(self.name)
        if value is not None:
            if node.component(self.name).lock_ref == 0:
                self.cache.component_evictable_size_[self.name] -= len(value)
                self.cache.component_protected_size_[self.name] += len(value)
            node.component(self.name).lock_ref += 1
        return None

    def release_component_lock(self, node: "HybridTreeNode", handle: Any) -> None:
        value = node.component_value(self.name)
        if value is not None:
            assert node.component(self.name).lock_ref > 0
            if node.component(self.name).lock_ref == 1:
                self.cache.component_evictable_size_[self.name] += len(value)
                self.cache.component_protected_size_[self.name] -= len(value)
            node.component(self.name).lock_ref -= 1
