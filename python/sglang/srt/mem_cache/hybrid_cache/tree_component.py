from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_radix_cache import HybridRadixCache, HybridTreeNode


BASE_COMPONENT_NAME = "full"

_LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)
_COMPONENT_UUID_COUNTER = 1


@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ComponentInsertResult:
    reused_existing: bool = False


@dataclasses.dataclass
class LockHandle:
    component_handles: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def legacy_swa_uuid(self) -> Any:
        return self.component_handles.get("swa")


def get_last_access_time() -> float64:
    global _LAST_ACCESS_TIME_COUNTER_FLOAT
    ret = _LAST_ACCESS_TIME_COUNTER_FLOAT
    _LAST_ACCESS_TIME_COUNTER_FLOAT += 1.0
    return ret


def gen_component_uuid() -> int:
    global _COMPONENT_UUID_COUNTER
    _COMPONENT_UUID_COUNTER += 1
    return _COMPONENT_UUID_COUNTER


class TreeComponent(ABC):
    def __init__(self, cache: "HybridRadixCache"):
        self.cache = cache

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def transform_key_for_match(self, key: RadixKey) -> Optional[RadixKey]: ...

    @abstractmethod
    def transform_key_value_for_insert(
        self, key: RadixKey, value: Optional[torch.Tensor]
    ) -> tuple[RadixKey, torch.Tensor]: ...

    @abstractmethod
    def init_match_walk_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def is_valid_match_endpoint(
        self, node: "HybridTreeNode", state: dict[str, Any]
    ) -> bool: ...

    @abstractmethod
    def compute_match_result_extras(
        self,
        params: MatchPrefixParams,
        last_node: "HybridTreeNode",
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def update_component_on_insert_overlap(
        self,
        node: "HybridTreeNode",
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None: ...

    @abstractmethod
    def get_tombstone_prefix_len_for_insert(
        self, total_prefix_len: int, new_key_len: int, params: InsertParams
    ) -> int: ...

    @abstractmethod
    def commit_insert_component_data(
        self, node: "HybridTreeNode", is_new_leaf: bool, params: InsertParams
    ) -> ComponentInsertResult: ...

    @abstractmethod
    def redistribute_on_node_split(
        self, new_parent: "HybridTreeNode", child: "HybridTreeNode"
    ): ...

    @abstractmethod
    def node_has_component_data(self, node: "HybridTreeNode") -> bool: ...

    @abstractmethod
    def evict_component_from_internal_node(self, node: "HybridTreeNode") -> int: ...

    @abstractmethod
    def release_component_on_leaf_eviction(self, node: "HybridTreeNode") -> int: ...

    def value_len(self, node: "HybridTreeNode") -> int:
        value = node.component_value(self.name)
        return len(value) if value is not None else 0

    def count_full_tokens_on_component_leaf_eviction(self) -> bool:
        return False

    @abstractmethod
    def acquire_component_lock(self, node: "HybridTreeNode") -> Any: ...

    @abstractmethod
    def release_component_lock(self, node: "HybridTreeNode", handle: Any) -> None: ...

    def export_public_lock_handle(self, handle: LockHandle) -> Any:
        return handle

    def import_public_lock_handle(self, handle: Any) -> LockHandle:
        if isinstance(handle, LockHandle):
            return handle
        return LockHandle(component_handles={self.name: handle})
