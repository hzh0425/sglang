from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.hybrid_cache.tree_component import (
    BASE_COMPONENT_NAME,
    ComponentInsertResult,
    LockHandle,
    TreeComponent,
)
from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_radix_cache import HybridTreeNode


class FullComponent(TreeComponent):
    @property
    def name(self) -> str:
        return BASE_COMPONENT_NAME

    def transform_key_for_match(self, key: RadixKey) -> Optional[RadixKey]:
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
        return node.full_value is not None

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
        return

    def get_tombstone_prefix_len_for_insert(
        self, total_prefix_len: int, new_key_len: int, params: InsertParams
    ) -> int:
        return 0

    def commit_insert_component_data(
        self, node: "HybridTreeNode", is_new_leaf: bool, params: InsertParams
    ) -> ComponentInsertResult:
        return ComponentInsertResult(reused_existing=not is_new_leaf)

    def redistribute_on_node_split(
        self, new_parent: "HybridTreeNode", child: "HybridTreeNode"
    ):
        return

    def node_has_component_data(self, node: "HybridTreeNode") -> bool:
        return node.full_value is not None

    def evict_component_from_internal_node(self, node: "HybridTreeNode") -> int:
        raise NotImplementedError("FullComponent internal eviction is handled by HybridRadixCache.")

    def release_component_on_leaf_eviction(self, node: "HybridTreeNode") -> int:
        return len(node.full_value)

    def acquire_component_lock(self, node: "HybridTreeNode") -> Any:
        return None

    def release_component_lock(self, node: "HybridTreeNode", handle: Any) -> None:
        return

    def export_public_lock_handle(self, handle: LockHandle) -> Any:
        return handle
