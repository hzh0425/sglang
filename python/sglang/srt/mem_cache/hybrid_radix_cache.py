from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch
from numpy import float64

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
    maybe_bigram_convert,
    page_align_keys,
)
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.utils import convert_to_bigram_key
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


BASE_COMPONENT_NAME = "full"


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


class HybridTreeNode:
    counter = 0
    last_access_time_counter_float = float64(1.0)
    component_uuid_counter = 1

    def __init__(self, component_names: list[str]):
        self.children = defaultdict(partial(HybridTreeNode, component_names))
        self.parent: HybridTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.component_names = list(component_names)
        self.component_data = {
            component_name: ComponentData() for component_name in self.component_names
        }
        self.last_access_time = get_last_access_time()
        self.host_value = None
        self.hit_count = 0
        self.lru_prev: dict[str, HybridTreeNode | None] = {
            component_name: None for component_name in self.component_names
        }
        self.lru_next: dict[str, HybridTreeNode | None] = {
            component_name: None for component_name in self.component_names
        }
        self.id = HybridTreeNode.counter
        HybridTreeNode.counter += 1

    def component(self, name: str) -> ComponentData:
        return self.component_data[name]

    @property
    def full_value(self) -> Optional[torch.Tensor]:
        return self.component(BASE_COMPONENT_NAME).value

    @full_value.setter
    def full_value(self, value: Optional[torch.Tensor]) -> None:
        self.component(BASE_COMPONENT_NAME).value = value

    def component_value(self, name: str) -> Optional[torch.Tensor]:
        return self.component(name).value

    def set_component_value(self, name: str, value: Optional[torch.Tensor]) -> None:
        self.component(name).value = value

    def __lt__(self, other: "HybridTreeNode"):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = HybridTreeNode.last_access_time_counter_float
    HybridTreeNode.last_access_time_counter_float += 1.0
    return ret


def gen_component_uuid() -> int:
    HybridTreeNode.component_uuid_counter += 1
    return HybridTreeNode.component_uuid_counter


class HybridLRUList:
    def __init__(self, component_name: str, component_names: list[str]):
        self.component_name = component_name
        self.head = HybridTreeNode(component_names)
        self.tail = HybridTreeNode(component_names)
        self.head.lru_next[component_name] = self.tail
        self.tail.lru_prev[component_name] = self.head
        self.cache: dict[int, HybridTreeNode] = {}

    def _add_node_after(self, old_node: HybridTreeNode, new_node: HybridTreeNode):
        component_name = self.component_name
        new_node.lru_prev[component_name] = old_node
        new_node.lru_next[component_name] = old_node.lru_next[component_name]
        old_node.lru_next[component_name].lru_prev[component_name] = new_node
        old_node.lru_next[component_name] = new_node

    def _add_node(self, node: HybridTreeNode):
        self._add_node_after(self.head, node)

    def _remove_node(self, node: HybridTreeNode):
        component_name = self.component_name
        node.lru_prev[component_name].lru_next[component_name] = node.lru_next[
            component_name
        ]
        node.lru_next[component_name].lru_prev[component_name] = node.lru_prev[
            component_name
        ]

    def insert_mru(self, node: HybridTreeNode):
        assert node.id not in self.cache
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: HybridTreeNode):
        assert node.id in self.cache
        del self.cache[node.id]
        self._remove_node(node)

    def reset_node_mru(self, node: HybridTreeNode):
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(
        self,
        node: HybridTreeNode,
        root_node: HybridTreeNode,
        should_include,
    ):
        prev_node = self.head
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def in_list(self, node: Optional[HybridTreeNode]):
        return node is not None and node.id in self.cache

    def get_prev_no_lock(self, node: HybridTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        x = node.lru_prev[self.component_name]
        while x.component(self.component_name).lock_ref > 0:
            x = x.lru_prev[self.component_name]
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: HybridTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        x = node.lru_prev[self.component_name]
        while x.component(self.component_name).lock_ref > 0 or len(x.children) > 0:
            x = x.lru_prev[self.component_name]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)


class TreeComponent(ABC):
    def __init__(self, cache: "HybridRadixCache"):
        self.cache = cache

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def preprocess_match_key(self, key: RadixKey) -> Optional[RadixKey]: ...

    @abstractmethod
    def preprocess_insert_key(
        self, key: RadixKey, value: Optional[torch.Tensor]
    ) -> tuple[RadixKey, torch.Tensor]: ...

    @abstractmethod
    def init_match_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def on_match_visit(
        self, node: HybridTreeNode, matched_seg_len: int, state: dict[str, Any]
    ) -> bool: ...

    @abstractmethod
    def on_match_finalize(
        self,
        params: MatchPrefixParams,
        last_node: HybridTreeNode,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def on_split_node(self, new_node: HybridTreeNode, child: HybridTreeNode): ...

    @abstractmethod
    def should_track_in_lru(self, node: HybridTreeNode) -> bool: ...

    @abstractmethod
    def handle_overlap(
        self,
        node: HybridTreeNode,
        prefix_len: int,
        total_prefix_length: int,
        update_after_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None: ...

    @abstractmethod
    def before_add_new_leaf(
        self,
        node: HybridTreeNode,
        total_prefix_length: int,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> tuple[HybridTreeNode, RadixKey, torch.Tensor]: ...

    @abstractmethod
    def finalize_new_leaf(
        self, leaf: HybridTreeNode, params: InsertParams
    ) -> ComponentInsertResult: ...

    @abstractmethod
    def finalize_existing_node(
        self, node: HybridTreeNode, params: InsertParams
    ) -> ComponentInsertResult: ...

    @abstractmethod
    def free_internal(self, node: HybridTreeNode) -> int: ...

    @abstractmethod
    def free_leaf(self, node: HybridTreeNode) -> int: ...

    def value_len(self, node: HybridTreeNode) -> int:
        value = node.component_value(self.name)
        return len(value) if value is not None else 0

    @abstractmethod
    def get_lock_handle_and_inc(self, node: HybridTreeNode) -> Any: ...

    @abstractmethod
    def dec_lock_ref(self, node: HybridTreeNode, handle: Any) -> None: ...

    def export_public_lock_handle(self, handle: LockHandle) -> Any:
        return handle

    def import_public_lock_handle(self, handle: Any) -> LockHandle:
        if isinstance(handle, LockHandle):
            return handle
        return LockHandle(component_handles={self.name: handle})


class MambaComponent(TreeComponent):
    @property
    def name(self) -> str:
        return "mamba"

    def preprocess_match_key(self, key: RadixKey) -> Optional[RadixKey]:
        if self.cache.disable or len(key) == 0:
            return None
        return key

    def preprocess_insert_key(
        self, key: RadixKey, value: Optional[torch.Tensor]
    ) -> tuple[RadixKey, torch.Tensor]:
        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        return key, value

    def init_match_state(self) -> dict[str, Any]:
        return {}

    def on_match_visit(
        self, node: HybridTreeNode, matched_seg_len: int, state: dict[str, Any]
    ) -> bool:
        return node.component_value(self.name) is not None

    def on_match_finalize(
        self,
        params: MatchPrefixParams,
        last_node: HybridTreeNode,
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

    def on_split_node(self, new_node: HybridTreeNode, child: HybridTreeNode):
        new_node.set_component_value(self.name, None)
        new_node.component(self.name).lock_ref = 0

    def should_track_in_lru(self, node: HybridTreeNode) -> bool:
        return node.component_value(self.name) is not None

    def handle_overlap(
        self,
        node: HybridTreeNode,
        prefix_len: int,
        total_prefix_length: int,
        update_after_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None:
        return

    def before_add_new_leaf(
        self,
        node: HybridTreeNode,
        total_prefix_length: int,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> tuple[HybridTreeNode, RadixKey, torch.Tensor]:
        return node, key, value

    def finalize_new_leaf(
        self, leaf: HybridTreeNode, params: InsertParams
    ) -> ComponentInsertResult:
        assert params.mamba_value is not None
        leaf.set_component_value(self.name, params.mamba_value)
        self.cache.lru_lists[self.name].insert_mru(leaf)
        self.cache.component_evictable_size_[self.name] += len(params.mamba_value)
        return ComponentInsertResult()

    def finalize_existing_node(
        self, node: HybridTreeNode, params: InsertParams
    ) -> ComponentInsertResult:
        assert params.mamba_value is not None
        if node.component_value(self.name) is None:
            node.set_component_value(self.name, params.mamba_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(params.mamba_value)
            node.last_access_time = get_last_access_time()
            return ComponentInsertResult(reused_existing=False)
        self.cache.lru_lists[self.name].reset_node_mru(node)
        node.last_access_time = get_last_access_time()
        return ComponentInsertResult(reused_existing=True)

    def free_internal(self, node: HybridTreeNode) -> int:
        value = node.component_value(self.name)
        self.cache.req_to_token_pool.mamba_pool.free(value)
        freed = len(value)
        self.cache.component_evictable_size_[self.name] -= freed
        node.set_component_value(self.name, None)
        return freed

    def free_leaf(self, node: HybridTreeNode) -> int:
        value = node.component_value(self.name)
        self.cache.req_to_token_pool.mamba_pool.free(value)
        return len(value)

    def get_lock_handle_and_inc(self, node: HybridTreeNode) -> Any:
        value = node.component_value(self.name)
        if value is not None:
            if node.component(self.name).lock_ref == 0:
                self.cache.component_evictable_size_[self.name] -= len(value)
                self.cache.component_protected_size_[self.name] += len(value)
            node.component(self.name).lock_ref += 1
        return None

    def dec_lock_ref(self, node: HybridTreeNode, handle: Any) -> None:
        value = node.component_value(self.name)
        if value is not None:
            assert node.component(self.name).lock_ref > 0
            if node.component(self.name).lock_ref == 1:
                self.cache.component_evictable_size_[self.name] += len(value)
                self.cache.component_protected_size_[self.name] -= len(value)
            node.component(self.name).lock_ref -= 1


class SWAComponent(TreeComponent):
    @property
    def name(self) -> str:
        return "swa"

    def preprocess_match_key(self, key: RadixKey) -> Optional[RadixKey]:
        key, _ = maybe_bigram_convert(self.cache.is_eagle, key)
        if self.cache.disable or len(key) == 0:
            return None
        if self.cache.page_size != 1:
            page_aligned_len = len(key) // self.cache.page_size * self.cache.page_size
            key = key[:page_aligned_len]
        return key

    def preprocess_insert_key(
        self, key: RadixKey, value: Optional[torch.Tensor]
    ) -> tuple[RadixKey, torch.Tensor]:
        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        return maybe_bigram_convert(self.cache.is_eagle, key, value)

    def init_match_state(self) -> dict[str, Any]:
        return {"match_len_since_release": float("inf")}

    def on_match_visit(
        self, node: HybridTreeNode, matched_seg_len: int, state: dict[str, Any]
    ) -> bool:
        if node.component_value(self.name) is None:
            state["match_len_since_release"] = 0
            return False
        state["match_len_since_release"] += len(node.full_value)
        return state["match_len_since_release"] >= self.cache.sliding_window_size

    def on_match_finalize(
        self,
        params: MatchPrefixParams,
        last_node: HybridTreeNode,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> dict[str, Any]:
        return {}

    def on_split_node(self, new_node: HybridTreeNode, child: HybridTreeNode):
        child_value = child.component_value(self.name)
        new_node.set_component_value(
            self.name,
            new_node.full_value.clone() if child_value is not None else None,
        )
        new_node.component(self.name).lock_ref = child.component(self.name).lock_ref
        if "component_uuid" in child.component(self.name).metadata:
            new_node.component(self.name).metadata["component_uuid"] = child.component(
                self.name
            ).metadata["component_uuid"]
            child.component(self.name).metadata.pop("component_uuid", None)

    def should_track_in_lru(self, node: HybridTreeNode) -> bool:
        return node.component_value(self.name) is not None

    def handle_overlap(
        self,
        node: HybridTreeNode,
        prefix_len: int,
        total_prefix_length: int,
        update_after_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None:
        if update_after_len >= total_prefix_length + prefix_len:
            return
        if node.component_value(self.name) is not None:
            self.cache.token_to_kv_pool_allocator.free(value_slice)
            return

        assert params.swa_evicted_seqlen % self.cache.page_size == 0
        assert node.component(self.name).lock_ref == 0

        if params.swa_evicted_seqlen <= total_prefix_length:
            self.cache.token_to_kv_pool_allocator.free(node.full_value[:prefix_len])
            node.full_value = value_slice.clone()
            node.set_component_value(self.name, node.full_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(node.component_value(self.name))
        elif params.swa_evicted_seqlen < total_prefix_length + prefix_len:
            start_update_idx = params.swa_evicted_seqlen - total_prefix_length
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

    def before_add_new_leaf(
        self,
        node: HybridTreeNode,
        total_prefix_length: int,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> tuple[HybridTreeNode, RadixKey, torch.Tensor]:
        if (
            params.swa_evicted_seqlen > total_prefix_length
            and params.swa_evicted_seqlen < total_prefix_length + len(key)
        ):
            tombstone_len = params.swa_evicted_seqlen - total_prefix_length
            node = self.cache._add_new_leaf(
                node,
                key[:tombstone_len],
                value[:tombstone_len],
                component_values={self.name: None},
            )
            key = key[tombstone_len:]
            value = value[tombstone_len:]
        return node, key, value

    def finalize_new_leaf(
        self, leaf: HybridTreeNode, params: InsertParams
    ) -> ComponentInsertResult:
        leaf.set_component_value(self.name, leaf.full_value)
        self.cache.lru_lists[self.name].insert_mru(leaf)
        self.cache.component_evictable_size_[self.name] += len(leaf.component_value(self.name))
        return ComponentInsertResult()

    def finalize_existing_node(
        self, node: HybridTreeNode, params: InsertParams
    ) -> ComponentInsertResult:
        return ComponentInsertResult(reused_existing=True)

    def free_internal(self, node: HybridTreeNode) -> int:
        self.cache.token_to_kv_pool_allocator.free_swa(node.full_value)
        freed = len(node.full_value)
        self.cache.component_evictable_size_[self.name] -= freed
        node.set_component_value(self.name, None)
        return freed

    def free_leaf(self, node: HybridTreeNode) -> int:
        return len(node.component_value(self.name))

    def get_lock_handle_and_inc(self, node: HybridTreeNode) -> Any:
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

    def dec_lock_ref(self, node: HybridTreeNode, handle: Any) -> None:
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


@dataclasses.dataclass(frozen=True)
class HybridTreeSpec:
    component_names: tuple[str, ...]
    public_lock_component_name: Optional[str] = None


def build_hybrid_tree_spec(
    params: "CacheInitParams",
    component_names: Optional[tuple[str, ...]] = None,
) -> HybridTreeSpec:
    if component_names is None:
        component_names = tuple(getattr(params, "hybrid_tree_components", ()) or ())
    if not component_names:
        if isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            component_names = ("swa",)
        elif isinstance(params.req_to_token_pool, HybridReqToTokenPool):
            component_names = ("mamba",)
        else:
            raise ValueError("Can not infer hybrid tree components from params.")
    public_lock_component_name = component_names[0] if len(component_names) == 1 else None
    return HybridTreeSpec(
        component_names=component_names,
        public_lock_component_name=public_lock_component_name,
    )


COMPONENT_REGISTRY = {
    "mamba": MambaComponent,
    "swa": SWAComponent,
}


class HybridRadixCache(BasePrefixCache):
    def __init__(
        self,
        params: "CacheInitParams",
        component_names: tuple[str, ...],
        public_lock_component_name: Optional[str] = None,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        self.is_eagle = params.is_eagle
        self.sliding_window_size = params.sliding_window_size

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        self.component_names = [BASE_COMPONENT_NAME, *component_names]
        self.component_order = list(component_names)
        self.public_lock_component_name = public_lock_component_name
        self.components = {
            name: COMPONENT_REGISTRY[name](self) for name in self.component_order
        }
        if self.is_eagle:
            self.key_convert_fn = convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key
        self.reset()

    def reset(self) -> None:
        self.root_node = HybridTreeNode(self.component_names)
        self.root_node.key = RadixKey([], None)
        self.root_node.full_value = []
        for component_name in self.component_names:
            self.root_node.component(component_name).lock_ref = 1
        self.full_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.component_evictable_size_ = {name: 0 for name in self.component_order}
        self.component_protected_size_ = {name: 0 for name in self.component_order}
        self._usage_counters_dirty = False
        self.lru_lists = {
            component_name: HybridLRUList(component_name, self.component_names)
            for component_name in self.component_names
        }

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        raise NotImplementedError(
            "Generic HybridRadixCache requires a request lifecycle adapter. "
            f"Configured components={self.component_order}."
        )

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        raise NotImplementedError(
            "Generic HybridRadixCache requires a request lifecycle adapter. "
            f"Configured components={self.component_order}."
        )

    def _mark_usage_counters_dirty(self) -> None:
        self._usage_counters_dirty = True

    def _refresh_usage_counters(self) -> None:
        if not self._usage_counters_dirty:
            return

        full_evictable_size = 0
        full_protected_size = 0
        component_evictable_size = {name: 0 for name in self.component_order}
        component_protected_size = {name: 0 for name in self.component_order}
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())

            full_len = len(node.full_value)
            if node.component(BASE_COMPONENT_NAME).lock_ref > 0:
                full_protected_size += full_len
            else:
                full_evictable_size += full_len

            for component_name in self.component_order:
                value = node.component_value(component_name)
                if value is None:
                    continue
                if node.component(component_name).lock_ref > 0:
                    component_protected_size[component_name] += len(value)
                else:
                    component_evictable_size[component_name] += len(value)

        self.full_evictable_size_ = full_evictable_size
        self.full_protected_size_ = full_protected_size
        self.component_evictable_size_ = component_evictable_size
        self.component_protected_size_ = component_protected_size
        self._usage_counters_dirty = False

    def supports_swa(self) -> bool:
        return "swa" in self.components

    def supports_mamba(self) -> bool:
        return "mamba" in self.components

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        for component in self.components.values():
            processed_key = component.preprocess_match_key(key)
            if processed_key is None:
                return MatchResult(
                    device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
                    last_device_node=self.root_node,
                    last_host_node=self.root_node,
                )
            key = processed_key
        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        for component in self.components.values():
            key, value = component.preprocess_insert_key(key, value)
        prefix_len, component_results = self._insert_helper(self.root_node, key, value, params)
        self._mark_usage_counters_dirty()
        mamba_result = component_results.get("mamba", ComponentInsertResult())
        return InsertResult(
            prefix_len=prefix_len,
            mamba_exist=mamba_result.reused_existing,
        )

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], HybridTreeNode, int]:
        node = self.root_node
        child_key = self.get_child_key_fn(key)
        value: list[torch.Tensor] = []
        best_value_len = 0
        best_node = node
        component_states = {
            name: component.init_match_state()
            for name, component in self.components.items()
        }
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                node = self._split_node(child.key, child, prefix_len)
                value.append(node.full_value)
                if all(
                    self.components[name].on_match_visit(
                        node, prefix_len, component_states[name]
                    )
                    for name in self.component_order
                ):
                    best_value_len = len(value)
                    best_node = node
                break
            value.append(child.full_value)
            node = child
            if all(
                self.components[name].on_match_visit(
                    node, prefix_len, component_states[name]
                )
                for name in self.component_order
            ):
                best_value_len = len(value)
                best_node = node
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return value, best_node, best_value_len

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        last_node: HybridTreeNode,
        best_value_len: int,
    ) -> MatchResult:
        node_update = last_node
        self.lru_lists[BASE_COMPONENT_NAME].reset_node_and_parents_mru(
            node_update, self.root_node, lambda node: True
        )
        for component_name, component in self.components.items():
            self.lru_lists[component_name].reset_node_and_parents_mru(
                node_update, self.root_node, component.should_track_in_lru
            )
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        extras = {}
        for component in self.components.values():
            extras.update(
                component.on_match_finalize(params, last_node, value, best_value_len)
            )

        value = value[:best_value_len]
        if value:
            device_indices = torch.cat(value)
        else:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=device_indices,
            last_device_node=last_node,
            last_host_node=last_node,
            mamba_branching_seqlen=extras.get("mamba_branching_seqlen"),
        )

    def _split_node(
        self, key: RadixKey, child: HybridTreeNode, split_len: int
    ) -> HybridTreeNode:
        new_node = HybridTreeNode(self.component_names)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.full_value = child.full_value[:split_len].clone()
        new_node.component(BASE_COMPONENT_NAME).lock_ref = child.component(
            BASE_COMPONENT_NAME
        ).lock_ref
        for component in self.components.values():
            component.on_split_node(new_node, child)

        self.lru_lists[BASE_COMPONENT_NAME].remove_node(child)
        for component_name, component in self.components.items():
            if component.should_track_in_lru(child):
                self.lru_lists[component_name].remove_node(child)

        child.parent = new_node
        child.key = child.key[split_len:]
        child.full_value = child.full_value[split_len:].clone()
        for component_name in self.component_order:
            value = child.component_value(component_name)
            if component_name == "swa" and value is not None:
                child.set_component_value(component_name, child.full_value)
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self.lru_lists[BASE_COMPONENT_NAME].insert_mru(new_node)
        self.lru_lists[BASE_COMPONENT_NAME].insert_mru(child)
        for component_name, component in self.components.items():
            if component.should_track_in_lru(new_node):
                self.lru_lists[component_name].insert_mru(new_node)
            if component.should_track_in_lru(child):
                self.lru_lists[component_name].insert_mru(child)
        child.last_access_time = get_last_access_time()
        return new_node

    def _touch_node(self, node: HybridTreeNode):
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.lru_lists[BASE_COMPONENT_NAME].reset_node_mru(node)
            for component_name, component in self.components.items():
                if component.should_track_in_lru(node):
                    self.lru_lists[component_name].reset_node_mru(node)

    def _add_new_leaf(
        self,
        parent: HybridTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        component_values: Optional[dict[str, Optional[torch.Tensor]]] = None,
    ) -> HybridTreeNode:
        new_node = HybridTreeNode(self.component_names)
        new_node.parent = parent
        new_node.key = key
        new_node.full_value = value.clone()
        for component_name, component_value in (component_values or {}).items():
            new_node.set_component_value(component_name, component_value)
        parent.children[self.get_child_key_fn(key)] = new_node
        self.lru_lists[BASE_COMPONENT_NAME].insert_mru(new_node)
        self.full_evictable_size_ += len(value)
        return new_node

    def _insert_helper(
        self,
        node: HybridTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> tuple[int, dict[str, ComponentInsertResult]]:
        self._touch_node(node)
        if len(key) == 0:
            return 0, {name: ComponentInsertResult(reused_existing=True) for name in self.component_order}

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = self.key_match_fn(node.key, key)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            for component in self.components.values():
                component.handle_overlap(
                    node,
                    prefix_len,
                    total_prefix_length,
                    params.prev_prefix_len,
                    value[:prefix_len],
                    params,
                )

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            for component in self.components.values():
                node, key, value = component.before_add_new_leaf(
                    node, total_prefix_length, key, value, params
                )
            if len(key):
                new_node = self._add_new_leaf(node, key, value)
                results = {
                    name: component.finalize_new_leaf(new_node, params)
                    for name, component in self.components.items()
                }
            else:
                results = {
                    name: ComponentInsertResult(reused_existing=True)
                    for name in self.component_order
                }
        else:
            results = {
                name: component.finalize_existing_node(node, params)
                for name, component in self.components.items()
            }
        return total_prefix_length, results

    def _delete_leaf(self, node: HybridTreeNode) -> tuple[int, dict[str, int]]:
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node
        full_len = len(node.full_value)
        component_lens = {}
        self.full_evictable_size_ -= full_len
        for component_name, component in self.components.items():
            component_len = component.value_len(node)
            component_lens[component_name] = component_len
            if component_len:
                self.component_evictable_size_[component_name] -= component_len
        return full_len, component_lens

    def _delete_tombstone_leaf(self, node: HybridTreeNode) -> int:
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node
        full_len = len(node.full_value)
        self.full_evictable_size_ -= full_len
        return full_len

    def _iteratively_delete_tombstone_leaf(
        self, node: HybridTreeNode
    ) -> tuple[HybridTreeNode, int]:
        full_num_evicted = 0
        while node.parent != self.root_node and len(node.parent.children) == 0:
            if any(
                node.parent.component_value(component_name) is not None
                for component_name in self.component_order
            ):
                break
            if node.parent.component(BASE_COMPONENT_NAME).lock_ref > 0:
                break
            assert all(
                node.parent.component(component_name).lock_ref == 0
                for component_name in self.component_order
            )
            self.token_to_kv_pool_allocator.free(node.parent.full_value)
            full_num_evicted += len(node.parent.full_value)
            self.lru_lists[BASE_COMPONENT_NAME].remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent
        return node, full_num_evicted

    def _evict_leaf_node(
        self, node: HybridTreeNode, trigger_component_name: str
    ) -> tuple[int, dict[str, int], HybridTreeNode, Optional[HybridTreeNode]]:
        assert node.component(BASE_COMPONENT_NAME).lock_ref == 0
        assert all(
            node.component(component_name).lock_ref == 0
            for component_name in self.component_order
            if node.component_value(component_name) is not None
        )
        self.token_to_kv_pool_allocator.free(node.full_value)
        full_num_evicted = len(node.full_value)
        component_num_evicted = {}
        for component_name, component in self.components.items():
            if node.component_value(component_name) is None:
                component_num_evicted[component_name] = 0
                continue
            component_num_evicted[component_name] = component.free_leaf(node)
            node.set_component_value(component_name, None)

        if trigger_component_name == BASE_COMPONENT_NAME:
            next_node = self.lru_lists[BASE_COMPONENT_NAME].get_prev_leaf_no_lock(node)
        else:
            next_node = self.lru_lists[trigger_component_name].get_prev_no_lock(node)

        self.lru_lists[BASE_COMPONENT_NAME].remove_node(node)
        for component_name, component in self.components.items():
            if component.should_track_in_lru(node):
                self.lru_lists[component_name].remove_node(node)
        self._delete_leaf(node)
        node, extra_full = self._iteratively_delete_tombstone_leaf(node)
        full_num_evicted += extra_full
        return full_num_evicted, component_num_evicted, node, next_node

    @property
    def cache_req_mamba_pool(self):
        return self.req_to_token_pool.mamba_pool

    def _component_request(self, params: EvictParams, component_name: str) -> int:
        if component_name == "mamba":
            return params.mamba_num
        if component_name == "swa":
            return params.swa_num_tokens
        return 0

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        full_num_evicted = 0
        component_num_evicted = {name: 0 for name in self.component_order}

        if params.num_tokens > 0:
            x = self.lru_lists[BASE_COMPONENT_NAME].get_leaf_lru_no_lock()
            while full_num_evicted < params.num_tokens and self.lru_lists[BASE_COMPONENT_NAME].in_list(x):
                full_delta, component_delta, x, x_next = self._evict_leaf_node(
                    x, BASE_COMPONENT_NAME
                )
                full_num_evicted += full_delta
                for component_name in self.component_order:
                    component_num_evicted[component_name] += component_delta[component_name]
                if len(x.parent.children) == 0:
                    x_next = self.lru_lists[BASE_COMPONENT_NAME].get_leaf_lru_no_lock()
                x = x_next

        for component_name, component in self.components.items():
            component_request = self._component_request(params, component_name)
            if component_num_evicted[component_name] >= component_request:
                continue
            x = self.lru_lists[component_name].get_lru_no_lock()
            while (
                component_num_evicted[component_name] < component_request
                and self.lru_lists[component_name].in_list(x)
            ):
                assert x.component_value(component_name) is not None
                if len(x.children) > 0:
                    component_num_evicted[component_name] += component.free_internal(x)
                    x_next = self.lru_lists[component_name].get_prev_no_lock(x)
                    self.lru_lists[component_name].remove_node(x)
                else:
                    full_delta, component_delta, _, x_next = self._evict_leaf_node(
                        x, component_name
                    )
                    full_num_evicted += full_delta
                    for name in self.component_order:
                        component_num_evicted[name] += component_delta[name]
                x = x_next

        self.update_eviction_metrics(
            full_num_evicted + sum(component_num_evicted.values()), start_time
        )
        self._mark_usage_counters_dirty()
        return EvictResult(
            num_tokens_evicted=full_num_evicted,
            swa_num_tokens_evicted=component_num_evicted.get("swa", 0),
            mamba_num_evicted=component_num_evicted.get("mamba", 0),
        )

    def inc_lock_ref(self, node: HybridTreeNode):
        if self.disable:
            return None
        handle = LockHandle()
        for component_name, component in self.components.items():
            handle.component_handles[component_name] = component.get_lock_handle_and_inc(node)
        cur = node
        while cur != self.root_node:
            if cur.component(BASE_COMPONENT_NAME).lock_ref == 0:
                self.full_evictable_size_ -= len(cur.full_value)
                self.full_protected_size_ += len(cur.full_value)
            cur.component(BASE_COMPONENT_NAME).lock_ref += 1
            cur = cur.parent
        self._mark_usage_counters_dirty()
        if self.public_lock_component_name is not None:
            return self.components[self.public_lock_component_name].export_public_lock_handle(
                handle
            )
        return handle

    def dec_lock_ref(self, node: HybridTreeNode, handle: Any = None):
        if self.disable:
            return
        if self.public_lock_component_name is not None:
            normalized_handle = self.components[
                self.public_lock_component_name
            ].import_public_lock_handle(handle)
        elif isinstance(handle, LockHandle):
            normalized_handle = handle
        else:
            normalized_handle = LockHandle()
        for component_name, component in self.components.items():
            component.dec_lock_ref(
                node, normalized_handle.component_handles.get(component_name)
            )
        cur = node
        while cur != self.root_node:
            assert cur.component(BASE_COMPONENT_NAME).lock_ref > 0
            if cur.component(BASE_COMPONENT_NAME).lock_ref == 1:
                self.full_evictable_size_ += len(cur.full_value)
                self.full_protected_size_ -= len(cur.full_value)
            cur.component(BASE_COMPONENT_NAME).lock_ref -= 1
            cur = cur.parent
        self._mark_usage_counters_dirty()

    def full_evictable_size(self) -> int:
        self._refresh_usage_counters()
        return self.full_evictable_size_

    def full_protected_size(self) -> int:
        self._refresh_usage_counters()
        return self.full_protected_size_

    def swa_evictable_size(self) -> int:
        self._refresh_usage_counters()
        return self.component_evictable_size_.get("swa", 0)

    def mamba_evictable_size(self) -> int:
        self._refresh_usage_counters()
        return self.component_evictable_size_.get("mamba", 0)

    def swa_protected_size(self) -> int:
        self._refresh_usage_counters()
        return self.component_protected_size_.get("swa", 0)

    def mamba_protected_size(self) -> int:
        self._refresh_usage_counters()
        return self.component_protected_size_.get("mamba", 0)

    def total_size(self):
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            total_size += len(node.full_value)
            for component_name in self.component_order:
                value = node.component_value(component_name)
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: HybridTreeNode):
            for child in node.children.values():
                values.append(child.full_value)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(self, component_name: str) -> torch.Tensor:
        if component_name not in self.components:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: HybridTreeNode):
            value = node.component_value(component_name)
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten("mamba")

    def all_swa_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten("swa")

    def available_and_evictable_str(self) -> str:
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        lines = [
            f"Available full tokens: {full_available_size + self.full_evictable_size_} "
            f"(full_available_size={full_available_size} + full_evictable_size_={self.full_evictable_size_})"
        ]
        for component_name in self.component_order:
            if component_name == "swa":
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif component_name == "mamba":
                available_size = self.cache_req_mamba_pool.available_size()
            else:
                available_size = 0
            lines.append(
                f"Available {component_name}: {available_size + self.component_evictable_size_[component_name]} "
                f"(available_size={available_size} + component_evictable_size_={self.component_evictable_size_[component_name]})"
            )
        return "\n".join(lines) + "\n"

    def sanity_check(self):
        assert self.full_evictable_size_ >= 0
        assert self.full_protected_size_ >= 0
        for component_name in self.component_order:
            assert self.component_evictable_size_[component_name] >= 0
            assert self.component_protected_size_[component_name] >= 0

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{component_name}={'yes' if node.component_value(component_name) is not None else 'no'}"
                for component_name in self.component_order
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component(BASE_COMPONENT_NAME).lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))


class HybridMambaRadixCache(HybridRadixCache):
    def __init__(self, params: "CacheInitParams"):
        assert isinstance(
            params.token_to_kv_pool_allocator, TokenToKVPoolAllocator
        ) or isinstance(params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator)
        assert isinstance(params.req_to_token_pool, HybridReqToTokenPool)
        if not params.enable_mamba_extra_buffer:
            assert params.page_size == 1
        super().__init__(
            params,
            component_names=("mamba",),
            public_lock_component_name="mamba",
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_mamba_cache(req)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        if is_insert:
            cache_len = (
                req.mamba_last_track_seqlen
                if self.enable_mamba_extra_buffer
                else len(token_ids)
            )
            if cache_len is None:
                cache_len = 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]
            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
            else:
                page_aligned_len = len(kv_indices)
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
            assert cache_len == page_aligned_len, (
                f"It is required {cache_len=}, {page_aligned_len=}, {kv_committed_len=}, "
                f"{len(req.origin_input_ids)=}, {len(req.output_ids)=}"
            )
            if self.enable_mamba_extra_buffer:
                keep_idx = self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
                mamba_value = (
                    req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:
                keep_idx = None
                mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
            result = self.insert(
                InsertParams(
                    key=RadixKey(token_ids[:page_aligned_len], req.extra_key),
                    value=page_aligned_kv_indices,
                    mamba_value=mamba_value,
                )
            )
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : result.prefix_len]
            )
            mamba_exist = result.mamba_exist
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])
            mamba_exist = True
            keep_idx = None

        if mamba_exist:
            keep_idx = None
        free_mamba_cache = True if self.enable_mamba_extra_buffer else mamba_exist
        if free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req, mamba_ping_pong_track_buffer_to_keep=keep_idx
            )
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        def _skip(req: Req):
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

        token_ids = req.fill_ids
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None:
            return _skip(req)
        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
        assert page_aligned_len == len(kv_indices), (
            f"page_aligned_len != len(kv_indices), {page_aligned_len=}, "
            f"{len(kv_indices)=}, {cache_len=}, {self.page_size=}, {FLA_CHUNK_SIZE=}"
        )
        page_aligned_token_ids = token_ids[:page_aligned_len]
        if self.enable_mamba_extra_buffer:
            keep_idx = self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
            mamba_value = (
                req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
            )
        else:
            mamba_value = self.req_to_token_pool.get_mamba_indices(
                req.req_pool_idx
            ).unsqueeze(-1)
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)
        if mamba_value_forked is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            assert mamba_value_forked is not None, "Can not alloc mamba cache"
        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_forked,
            )
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : result.prefix_len]
        )
        if result.mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)
        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node


class HybridSWARadixCache(HybridRadixCache):
    def __init__(self, params: "CacheInitParams"):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        super().__init__(
            params,
            component_names=("swa",),
            public_lock_component_name="swa",
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        keys = self.key_convert_fn(token_ids)
        keys = page_align_keys(keys, self.page_size)
        page_aligned_len = len(keys)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(
            keys[:page_aligned_len], req.extra_key, is_bigram=self.is_eagle
        )
        if is_insert:
            self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    prev_prefix_len=req.cache_protected_len,
                    swa_evicted_seqlen=req.swa_evicted_seqlen,
                )
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : page_aligned_len]
            )
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        self.dec_lock_ref(req.last_node, req.swa_uuid_for_lock)
        req.swa_uuid_for_lock = None

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            req.prefix_indices = kv_indices
            return
        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        keys = self.key_convert_fn(token_ids)
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)
        self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                prev_prefix_len=req.cache_protected_len,
            )
        )
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )
        req.cache_protected_len = len(new_indices)
        self.dec_lock_ref(req.last_node, req.swa_uuid_for_lock)
        req.swa_uuid_for_lock = self.inc_lock_ref(new_last_node)
        req.prefix_indices = (
            torch.cat([new_indices, kv_indices[len(new_indices) :]])
            if len(new_indices) < len(kv_indices)
            else new_indices
        )
        req.last_node = new_last_node


def create_hybrid_radix_cache(
    params: "CacheInitParams",
    component_names: Optional[tuple[str, ...]] = None,
) -> HybridRadixCache:
    spec = build_hybrid_tree_spec(params, component_names=component_names)
    if spec.component_names == ("mamba",):
        return HybridMambaRadixCache(params)
    if spec.component_names == ("swa",):
        return HybridSWARadixCache(params)
    return HybridRadixCache(
        params,
        component_names=spec.component_names,
        public_lock_component_name=spec.public_lock_component_name,
    )
