from __future__ import annotations

import logging
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
    maybe_bigram_convert,
    page_align_keys,
)
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    ComponentData,
    ComponentType,
    FullComponent,
    MambaComponent,
    SWAComponent,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.utils import convert_to_bigram_key

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


class UnifiedTreeNode:
    counter = 0

    __slots__ = (
        "children",
        "parent",
        "key",
        "tree_components",
        "component_data",
        "last_access_time",
        "host_value",
        "hit_count",
        "lru_prev",
        "lru_next",
        "id",
        "_base_cd",
    )

    def __init__(self, tree_components: tuple[ComponentType, ...]):
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        self.component_data = {ct: ComponentData() for ct in tree_components}
        self._base_cd = self.component_data[BASE_COMPONENT_TYPE]
        self.last_access_time = get_and_increase_time_counter()
        self.host_value = None
        self.hit_count = 0
        self.lru_prev: dict[ComponentType, UnifiedTreeNode | None] = {
            ct: None for ct in tree_components
        }
        self.lru_next: dict[ComponentType, UnifiedTreeNode | None] = {
            ct: None for ct in tree_components
        }
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    def __lt__(self, other: UnifiedTreeNode):
        return self.last_access_time < other.last_access_time


class UnifiedLRUList:
    def __init__(
        self, component_type: ComponentType, tree_components: tuple[ComponentType, ...]
    ):
        self.component_type = component_type
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[component_type] = self.tail
        self.tail.lru_prev[component_type] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}

    def _add_node_after(self, prev_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        ct = self.component_type
        new_node.lru_prev[ct] = prev_node
        new_node.lru_next[ct] = prev_node.lru_next[ct]
        prev_node.lru_next[ct].lru_prev[ct] = new_node
        prev_node.lru_next[ct] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        self._add_node_after(self.head, node)

    def _remove_node(self, node: UnifiedTreeNode):
        ct = self.component_type
        node.lru_prev[ct].lru_next[ct] = node.lru_next[ct]
        node.lru_next[ct].lru_prev[ct] = node.lru_prev[ct]

    def insert_mru(self, node: UnifiedTreeNode):
        assert node.id not in self.cache
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        del self.cache[node.id]
        self._remove_node(node)

    def reset_node_mru(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ):
        prev_node = self.head
        ct = self.component_type
        while node != root_node:
            if node.component_data[ct].value is not None:
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def in_list(self, node: Optional[UnifiedTreeNode]):
        return node is not None and node.id in self.cache

    def get_prev_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        ct = self.component_type
        x = node.lru_prev[ct]
        while x.component_data[ct].lock_ref > 0:
            x = x.lru_prev[ct]
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        ct = self.component_type
        x = node.lru_prev[ct]
        while x.component_data[ct].lock_ref > 0 or len(x.children) > 0:
            x = x.lru_prev[ct]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}

logger = logging.getLogger(__name__)


class UnifiedRadixCache(BasePrefixCache):
    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle

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

        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        self.components: dict[ComponentType, TreeComponent] = {
            ct: COMPONENT_REGISTRY[ct](self, params) for ct in self.tree_components
        }
        self._precompute_hot_paths()
        if self.is_eagle:
            self.key_convert_fn = convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key
        self.reset()
        logger.info(f"Init Unified RadixTree with components {self.tree_components}")

    def _precompute_hot_paths(self):
        """Pre-compute filtered component lists so hot loops skip no-op components."""

        def _overrides(method_name: str) -> tuple[TreeComponent, ...]:
            base = getattr(TreeComponent, method_name)
            return tuple(
                c for c in self.components.values()
                if getattr(type(c), method_name) is not base
            )

        self._component_types_tuple = tuple(self.tree_components)
        self._all_components_with_ct = tuple(
            (ct, c) for ct, c in self.components.items()
        )
        self._overlap_components = _overrides("update_component_on_insert_overlap")
        self._skip_leaf_components = _overrides("should_skip_leaf_creation")
        self._commit_components = _overrides("commit_insert_component_data")
        self._finalize_match_components = _overrides("finalize_match_result")

        # When exactly one non-trivial component uses a simple "value is not
        # None" validator, store its ComponentType so _match_prefix_helper can
        # inline the check instead of creating lambdas / calling all().
        _always_true = FullComponent.create_match_validator
        non_trivial = [
            (ct, c) for ct, c in self.components.items()
            if type(c).create_match_validator is not _always_true
        ]
        if len(non_trivial) == 1 and non_trivial[0][1]._simple_match_validator:
            self._match_inline_ct = non_trivial[0][0]
        else:
            self._match_inline_ct = None

    def reset(self) -> None:
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.root_node.key = RadixKey([], None)
        self.root_node._base_cd.value = []
        for ct in self.tree_components:
            self.root_node.component(ct).lock_ref = 1
        self.component_evictable_size_ = {ct: 0 for ct in self.tree_components}
        self.component_protected_size_ = {ct: 0 for ct in self.tree_components}
        self.lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        key, _ = maybe_bigram_convert(self.is_eagle, key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)

        key, value = maybe_bigram_convert(self.is_eagle, key, value)
        result = self._insert_helper(self.root_node, key, value, params)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        for component in self.components.values():
            component.drive_eviction(params=params, tracker=tracker)

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def inc_lock_ref(self, node: UnifiedTreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self.components.values():
            result = component.acquire_component_lock(node=node, result=result)
        return result

    def dec_lock_ref(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        for component in self.components.values():
            component.release_component_lock(node=node, params=params)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self.components.values():
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        result = None
        insert_params = None

        if is_insert:
            insert_params = InsertParams(prev_prefix_len=req.cache_protected_len)

            # components prepare insert data + return effective cache_len
            effective_cache_len = len(token_ids)
            for comp in self.components.values():
                cl = comp.prepare_for_caching_req(
                    req=req,
                    insert_params=insert_params,
                    token_ids_len=len(token_ids),
                    is_finished=True,
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

            # Truncate if needed
            if effective_cache_len < len(token_ids):
                free_start = max(effective_cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[free_start:])
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            # Key convert + page align
            keys = self.key_convert_fn(token_ids)
            keys = page_align_keys(keys, self.page_size)
            page_aligned_len = len(keys)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
            radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # Free unaligned tail
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )

        # cleanup
        for comp in self.components.values():
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=result, insert_params=insert_params
            )

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        token_ids = req.fill_ids

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(prev_prefix_len=req.cache_protected_len)
        effective_cache_len = len(token_ids)
        for comp in self.components.values():
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        if effective_cache_len <= 0:
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self.components.values():
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        # Key convert + page align
        keys = self.key_convert_fn(token_ids[:effective_cache_len])
        keys = page_align_keys(keys, self.page_size)
        page_aligned_len = len(keys)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # Match prefix
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        lock_result = self.inc_lock_ref(new_last_node)

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock

        # cleanup
        for comp in self.components.values():
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                insert_result=result,
                insert_params=insert_params,
            )

    # ---- Internal Helpers ----

    def _match_prefix_helper_readonly(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], UnifiedTreeNode, int]:
        """Read-only version of _match_prefix_helper that does not split nodes.
        Only considers fully matched nodes, ignores partial matches.

        Not used yet; reserved for future read-only match operations."""
        node = self.root_node
        child_key = self.get_child_key_fn(key)
        value: list[torch.Tensor] = []
        best_value_len = 0
        best_node = node
        validators = {
            ct: component.create_match_validator()
            for ct, component in self.components.items()
        }

        def _update_best_if_valid(node):
            nonlocal best_value_len, best_node
            if all(validators[ct](node) for ct in self.components):
                best_value_len = len(value)
                best_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                # Read-only: do not split, ignore partial match and stop
                break
            value.append(child._base_cd.value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return value, best_node, best_value_len

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], UnifiedTreeNode, int]:
        node = self.root_node
        child_key = self.get_child_key_fn(key)
        value: list[torch.Tensor] = []
        best_value_len = 0
        best_node = node
        key_match = self.key_match_fn
        get_child = self.get_child_key_fn
        _inline_ct = self._match_inline_ct

        # Build the per-node validity check.  For configs with a single
        # simple validator (e.g. mamba) we inline a direct attribute check
        # to avoid lambda/all() overhead on every node visit.
        if _inline_ct is not None:
            is_valid = None  # sentinel — handled inline below
        else:
            _always_true = FullComponent.create_match_validator
            validators = tuple(
                component.create_match_validator()
                for ct, component in self.components.items()
                if type(component).create_match_validator is not _always_true
            )

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = key_match(child.key, key)
            if prefix_len < len(child.key):
                node = self._split_node(child.key, child, prefix_len)
                value.append(node._base_cd.value)
                if _inline_ct is not None:
                    if node.component_data[_inline_ct].value is not None:
                        best_value_len = len(value)
                        best_node = node
                elif not validators or all(v(node) for v in validators):
                    best_value_len = len(value)
                    best_node = node
                break
            value.append(child._base_cd.value)
            node = child
            if _inline_ct is not None:
                if node.component_data[_inline_ct].value is not None:
                    best_value_len = len(value)
                    best_node = node
            elif not validators or all(v(node) for v in validators):
                best_value_len = len(value)
                best_node = node
            key = key[prefix_len:]
            if len(key):
                child_key = get_child(key)
        return value, best_node, best_value_len

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        last_node: UnifiedTreeNode,
        best_value_len: int,
    ) -> MatchResult:
        node_update = last_node
        root = self.root_node
        for ct in self._component_types_tuple:
            self.lru_lists[ct].reset_node_and_parents_mru(
                node_update, root,
            )
        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        if best_value_len > 0:
            device_indices = torch.cat(value[:best_value_len])
        else:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=last_node,
            last_host_node=last_node,
        )

        for component in self._finalize_match_components:
            result = component.finalize_match_result(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_value_len,
            )
        return result

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        child_base = child._base_cd
        new_node._base_cd.value = child_base.value[:split_len].clone()

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]
        child_base.value = child_base.value[split_len:].clone()

        for component in self.components.values():
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._for_each_component_lru(new_node, UnifiedLRUList.insert_mru)
        self._for_each_component_lru(child, UnifiedLRUList.insert_mru)
        child.last_access_time = get_and_increase_time_counter()
        return new_node

    def _touch_node(self, node: UnifiedTreeNode):
        node.last_access_time = get_and_increase_time_counter()
        if node != self.root_node:
            component_data = node.component_data
            lru_lists = self.lru_lists
            for ct in self._component_types_tuple:
                if component_data[ct].value is not None:
                    UnifiedLRUList.reset_node_mru(lru_lists[ct], node)

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components)
        new_node.parent = parent
        new_node.key = key
        new_node._base_cd.value = value.clone()
        parent.children[self.get_child_key_fn(key)] = new_node
        self.lru_lists[BASE_COMPONENT_TYPE].insert_mru(new_node)
        self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)
        return new_node

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> InsertResult:
        self._touch_node(node)
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        overlap_components = self._overlap_components
        has_overlap_components = len(overlap_components) > 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = self.key_match_fn(node.key, key)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            if has_overlap_components:
                value_slice = value[:prefix_len]
                consumed_from = prefix_len
                for component in overlap_components:
                    comp_consumed_from = component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )
                    consumed_from = min(consumed_from, comp_consumed_from)
            else:
                consumed_from = prefix_len

            dup_start = max(0, params.prev_prefix_len - total_prefix_length)
            if dup_start < consumed_from:
                self.token_to_kv_pool_allocator.free(
                    value[dup_start:consumed_from]
                )

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        is_new_leaf = False
        # Create new leaf for remaining suffix
        if len(key):
            if self._skip_leaf_components and any(
                comp.should_skip_leaf_creation(
                    total_prefix_len=total_prefix_length,
                    key_len=len(key),
                    params=params,
                )
                for comp in self._skip_leaf_components
            ):
                # TODO: When leaf creation is skipped, We should release all component
                # resources here or propagate a flag so that
                # cleanup_after_caching_req can free them properly.
                self.token_to_kv_pool_allocator.free(value)
                return InsertResult(prefix_len=total_prefix_length)
            target_node = self._add_new_node(node, key, value)
            is_new_leaf = True
        else:
            target_node = node

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        result = InsertResult(prefix_len=total_prefix_length)
        for component in self._commit_components:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )
        return result

    def _cascade_evict(
        self,
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        tracker: dict[ComponentType, int],
    ):
        """Cascade eviction from trigger to lower-or-equal priority components.

        When a component evicts a node, all other components with equal or
        lower eviction_priority on the same node are also evicted.
        If the node is a leaf, it is removed from the tree and any
        resulting tombstone ancestors are cleaned up recursively."""
        is_leaf = len(node.children) == 0
        trigger_priority = trigger.eviction_priority(is_leaf)

        for comp in self.components.values():
            if comp.eviction_priority(is_leaf) <= trigger_priority:
                if comp is not trigger and comp.node_has_component_data(node):
                    assert node.component_data[comp.component_type].lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node, comp, is_leaf=is_leaf, tracker=tracker
                    )

        if is_leaf:
            self._remove_leaf_from_parent(node)
            self._iteratively_delete_tombstone_leaf(node, tracker)

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        is_leaf: bool,
        tracker: dict[ComponentType, int],
    ) -> int:
        freed = comp.evict_component(node, is_leaf=is_leaf)
        tracker[comp.component_type] += freed
        lru = self.lru_lists[comp.component_type]
        if lru.in_list(node):
            lru.remove_node(node)
        return freed

    def _iteratively_delete_tombstone_leaf(
        self, deleted_node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ):
        """After a leaf is removed, walk up the parent chain and delete
        any ancestor that is leaf node and has lost any component data (tombstoned)."""
        cur = deleted_node.parent
        components = self._all_components_with_ct
        while cur != self.root_node and len(cur.children) == 0:
            cd = cur.component_data
            has_tombstone = any(
                cd[ct].value is None for ct, _ in components
            )
            if not has_tombstone:
                break

            if any(
                cd[ct].lock_ref > 0
                for ct, _ in components
                if cd[ct].value is not None
            ):
                break

            for ct, comp in components:
                if cd[ct].value is not None:
                    self._evict_component_and_detach_lru(
                        cur, comp, is_leaf=True, tracker=tracker
                    )
            self._remove_leaf_from_parent(cur)
            cur = cur.parent

    def _for_each_component_lru(self, node: UnifiedTreeNode, lru_op):
        component_data = node.component_data
        for ct in self._component_types_tuple:
            if component_data[ct].value is not None:
                lru_op(self.lru_lists[ct], node)

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        swa = self.components.get(ComponentType.SWA)
        return swa.sliding_window_size if swa else None

    def supports_swa(self) -> bool:
        return ComponentType.SWA in self.components

    def supports_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    def evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)

    def protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.SWA, 0)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.MAMBA, 0)

    def swa_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.SWA, 0)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.MAMBA, 0)

    def total_size(self):
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            total_size += len(node.component(BASE_COMPONENT_TYPE).value)
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component(ct).value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                values.append(child.component(BASE_COMPONENT_TYPE).value)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(
        self, component_type: ComponentType
    ) -> torch.Tensor:
        if component_type not in self.components:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: UnifiedTreeNode):
            value = node.component(component_type).value
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.MAMBA)

    def all_swa_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.SWA)

    def available_and_evictable_str(self) -> str:
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.component_evictable_size_[BASE_COMPONENT_TYPE]
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_pool.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.component_evictable_size_[ct]} "
                f"(available_size={available_size} + component_evictable_size_={self.component_evictable_size_[ct]})"
            )
        return "\n".join(lines) + "\n"

    def sanity_check(self):
        for ct in self.tree_components:
            assert self.component_evictable_size_[ct] >= 0
            assert self.component_protected_size_[ct] >= 0

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{ct}={'yes' if node.component(ct).value is not None else 'no'}"
                for ct in self.tree_components
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component(BASE_COMPONENT_TYPE).lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))
