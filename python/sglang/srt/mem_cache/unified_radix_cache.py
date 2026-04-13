from __future__ import annotations

import logging
import threading
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
    InitLoadBackParams,
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
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    ComponentData,
    ComponentType,
    FullComponent,
    HiCachePhase,
    MambaComponent,
    SWAComponent,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.utils import convert_to_bigram_key

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


class UnifiedTreeNode:
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...]):
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.hash_value = None
        self.hit_count = 0
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * _NUM_COMPONENT_TYPES
        self.lru_next: list[UnifiedTreeNode | None] = [None] * _NUM_COMPONENT_TYPES
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    @property
    def backuped(self) -> bool:
        """Tree-level: has any host backup (Full or auxiliary components)."""
        return any(
            cd.host_value is not None for cd in self.component_data
        )

    @property
    def evicted(self) -> bool:
        """Tree-level: Full KV not on device (non-root with value=None)."""
        return (
            self.parent is not None
            and self.component_data[ComponentType.FULL].value is None
        )

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
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        if self.is_eagle:
            self.key_convert_fn = convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key

        self.tp_group = params.tp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )

        # HiCache D↔H defaults (overridden by _init_hicache)
        self.cache_controller = None
        self.write_through_threshold = 1
        self.enable_storage = False

        self.reset()
        logger.info(f"Init Unified RadixTree with components {self.tree_components}")

    def reset(self) -> None:
        if self.cache_controller is not None:
            # HiCache mode: flush L1 (device) only, preserve L2 (host) + tree
            self._reset_l1_only()
        else:
            # Non-HiCache: full reset
            self._reset_full()

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.root_node.key = RadixKey([], None)
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        for ct in self.tree_components:
            self.root_node.component_data[ct].lock_ref = 1
        self.component_evictable_size_ = {ct: 0 for ct in self.tree_components}
        self.component_protected_size_ = {ct: 0 for ct in self.tree_components}
        
        self.lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }
        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        self.host_lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }
        self.ongoing_write_through: dict[int, UnifiedTreeNode] = {}
        self.ongoing_load_back: dict[int, UnifiedTreeNode] = {}

    def _reset_l1_only(self) -> None:
        """Flush L1 (device) cache only; preserve L2 (host) and tree structure.

        After this call every tree node becomes evicted (device value = None)
        but keeps its component host_value so that subsequent match_prefix can still
        traverse the tree and init_load_back can restore data from host.
        """
        # 1. Drain pending async operations
        self.writing_check(write_back=True)
        if self.cache_controller is not None:
            self.cache_controller.ack_write_queue.clear()
            self.cache_controller.ack_load_queue.clear()
            self.cache_controller.write_queue.clear()
            self.cache_controller.load_queue.clear()
        self.ongoing_write_through.clear()
        self.ongoing_load_back.clear()

        # 2. Walk tree: release all device resources, keep component host_values
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            for ct in self.tree_components:
                cd = node.component_data[ct]
                if node is not self.root_node:
                    cd.value = None  # Release device reference
                cd.lock_ref = 1 if node is self.root_node else 0
                # cd.host_value / cd.host_lock_ref are preserved
            stack.extend(node.children.values())

        # Root keeps its empty-list value marker
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []

        # 3. Reset bookkeeping
        self.component_evictable_size_ = {ct: 0 for ct in self.tree_components}
        self.component_protected_size_ = {ct: 0 for ct in self.tree_components}
        self.lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }

        # 4. Reset device leaf set
        self.evictable_device_leaves.clear()

        # 5. Rebuild host leaf sets (all non-root backuped nodes are now evictable host leaves)
        self.evictable_host_leaves = set()
        self._rebuild_host_leaf_sets()

        # 6. Rebuild host LRU lists for extra components
        self.host_lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }
        self._rebuild_host_lru_lists()

        logger.info(
            "UnifiedRadixCache L1-only reset completed: "
            "tree structure and L2 host data preserved"
        )

    def _rebuild_host_leaf_sets(self) -> None:
        """Rebuild evictable_host_leaves after L1-only reset."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node:
                self._update_host_leaf_status(node)
            stack.extend(node.children.values())

    def _rebuild_host_lru_lists(self) -> None:
        """Rebuild host_lru_lists for extra components after L1-only reset.
        Walks the tree and adds nodes with host component data to the
        appropriate host LRU list."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node:
                for ct in self.tree_components:
                    if ct == BASE_COMPONENT_TYPE:
                        continue  # Full uses evictable_host_leaves, not host LRU
                    cd = node.component_data[ct]
                    if cd.host_value is not None:
                        self.host_lru_lists[ct].insert_mru(node)
            stack.extend(node.children.values())

    def _init_hicache(
        self, server_args: ServerArgs, params: CacheInitParams
    ) -> None:
        """Initialize HiCache D↔H infrastructure.

        Called by the scheduler when ``enable_hierarchical_cache`` is set.
        Creates host pools, assembles HostPoolGroup, and creates
        HybridCacheController for asynchronous D↔H transfers.
        """
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.memory_pool_host import (
            HostPoolGroup,
            MambaPoolHost,
            MHATokenToKVPoolHost,
            MLATokenToKVPoolHost,
            PoolEntry,
        )

        # Direct IO layout fixup (must happen before pool creation)
        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, "
                    "switching to page first direct layout"
                )

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()

        # === 1. Host Pool creation ===
        if isinstance(kvcache, HybridLinearKVPool):
            full_kv_pool = kvcache.full_kv_pool
            use_mla = kvcache.use_mla
        else:
            full_kv_pool = kvcache
            from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

            use_mla = isinstance(full_kv_pool, MLATokenToKVPool)

        kv_host_pool_cls = MLATokenToKVPoolHost if use_mla else MHATokenToKVPoolHost
        self.full_kv_pool_host = kv_host_pool_cls(
            full_kv_pool,
            server_args.hicache_ratio,
            server_args.hicache_size,
            self.page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )

        # Mamba host pool (only when MAMBA component is present)
        if ComponentType.MAMBA in self.components:
            mamba_comp = self.components[ComponentType.MAMBA]
            self.mamba_pool_host = MambaPoolHost(
                self.req_to_token_pool.mamba_pool,
                server_args.hicache_ratio,
                server_args.hicache_size,
                allocator_type=server_args.hicache_storage_backend,
                layout=server_args.hicache_mem_layout,
            )
            mamba_comp._mamba_pool_host = self.mamba_pool_host
            mamba_comp._hicache_enabled = True

        # === 2. HostPoolGroup assembly (layer mapping + evict callbacks) ===
        full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
        mamba_layer_mapping = (
            dict(self.req_to_token_pool.mamba_map)
            if hasattr(self.req_to_token_pool, "mamba_map")
            else {}
        )
        transfer_layer_num = len(
            set(full_layer_mapping) | set(mamba_layer_mapping)
        )

        def kv_layer_mapper(layer_id: int) -> Optional[int]:
            if not 0 <= layer_id < transfer_layer_num:
                return None
            return full_layer_mapping.get(layer_id)

        def mamba_layer_mapper(layer_id: int) -> Optional[int]:
            if not 0 <= layer_id < transfer_layer_num:
                return None
            return mamba_layer_mapping.get(layer_id)

        entries: list[PoolEntry] = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=self.full_kv_pool_host,
                device_pool=full_kv_pool,
                layer_mapper=kv_layer_mapper,
                is_primary_index_anchor=True,
            ),
        ]
        if ComponentType.MAMBA in self.components:
            entries.append(
                PoolEntry(
                    name=PoolName.MAMBA,
                    host_pool=self.mamba_pool_host,
                    device_pool=self.req_to_token_pool.mamba_pool,
                    layer_mapper=mamba_layer_mapper,
                    host_evict_fn=lambda n: self._evict_host(n, ComponentType.MAMBA),
                    device_evict_fn=lambda n: self.evict(EvictParams(mamba_num=n)),
                )
            )
        self.host_pool_group = HostPoolGroup(entries)

        # === 3. HybridCacheController creation ===
        self.load_cache_event = threading.Event()
        self.cache_controller = HybridCacheController(
            self.token_to_kv_pool_allocator,
            self.host_pool_group,
            self.page_size,
            params.tp_cache_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=None,  # Phase 2: server_args.hicache_storage_backend
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            transfer_layer_num=transfer_layer_num,
        )

        # Register layer-transfer counter for per-layer async loading
        self.req_to_token_pool.register_layer_transfer_counter(
            self.cache_controller.layer_done_counter
        )
        kvcache.register_layer_transfer_counter(
            self.cache_controller.layer_done_counter
        )

        # === 4. State initialization ===
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.enable_storage = False  # Phase 2: server_args.hicache_storage_backend is not None

        # Enable HiCache on components
        self.components[ComponentType.FULL]._hicache_enabled = True

        logger.info(
            f"HiCache D↔H initialized: "
            f"host_pool_size={self.host_pool_group.size}, "
            f"write_policy={server_args.hicache_write_policy}, "
            f"tp_world_size={self.tp_world_size}, "
            f"transfer_layer_num={transfer_layer_num}"
        )

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

        for component in self._components_tuple:
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
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)
        return result

    def dec_lock_ref(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
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
            for comp in self._components_tuple:
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
            for comp in self._components_tuple:
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
        for comp in self._components_tuple:
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
        insert_params = InsertParams(prev_prefix_len=req.cache_protected_len, chunked=chunked)
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
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
            for comp in self._components_tuple:
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
        for comp in self._components_tuple:
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
        validators = tuple(
            comp.create_match_validator() for comp in self._components_tuple
        )

        def _update_best_if_valid(node):
            nonlocal best_value_len, best_node
            if all(v(node) for v in validators):
                best_value_len = len(value)
                best_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                break

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                # Read-only: do not split, ignore partial match and stop
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
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
        validators = tuple(
            comp.create_match_validator() for comp in self._components_tuple
        )

        def _update_best_if_valid(node):
            nonlocal best_value_len, best_node
            if all(v(node) for v in validators):
                best_value_len = len(value)
                best_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                break

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                if child.evicted:
                    break
                node = self._split_node(child.key, child, prefix_len)
                value.append(node.component_data[BASE_COMPONENT_TYPE].value)
                _update_best_if_valid(node)
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return value, best_node, best_value_len

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        last_node: UnifiedTreeNode,
        best_value_len: int,
    ) -> MatchResult:
        node_update = last_node
        for comp in self._components_tuple:
            self.lru_lists[comp.component_type].reset_node_and_parents_mru(
                node_update, self.root_node, comp.node_has_component_data
            )
        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        # Walk up to find last_device_node (first non-evicted ancestor)
        last_device_node = last_node
        while last_device_node is not self.root_node and last_device_node.evicted:
            last_device_node = last_device_node.parent

        # Walk up to find last_host_node (first ancestor with host backup)
        last_host_node = last_node
        while last_host_node is not self.root_node and not last_host_node.backuped:
            last_host_node = last_host_node.parent

        if best_value_len > 0:
            device_indices = torch.cat(value[:best_value_len])
        else:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=last_device_node,
            last_host_node=last_host_node,
            host_hit_length=0,
        )

        for component in self._components_tuple:
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

        child_full_value = child.component_data[BASE_COMPONENT_TYPE].value
        if child_full_value is not None:
            new_node.component_data[BASE_COMPONENT_TYPE].value = (
                child_full_value[:split_len].clone()
            )

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]
        if child_full_value is not None:
            child.component_data[BASE_COMPONENT_TYPE].value = (
                child_full_value[split_len:].clone()
            )

        for component in self._components_tuple:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._for_each_component_lru(new_node, UnifiedLRUList.insert_mru)
        self._for_each_component_lru(child, UnifiedLRUList.insert_mru)
        child.last_access_time = get_and_increase_time_counter()

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        return new_node

    def _touch_node(self, node: UnifiedTreeNode):
        node.last_access_time = get_and_increase_time_counter()
        if node != self.root_node:
            self._for_each_component_lru(node, UnifiedLRUList.reset_node_mru)

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].value = value.clone()
        parent.children[self.get_child_key_fn(key)] = new_node
        self.lru_lists[BASE_COMPONENT_TYPE].insert_mru(new_node)
        self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        return new_node

    def _unevict_node_on_insert(
        self, node: UnifiedTreeNode, fresh_value: torch.Tensor
    ) -> None:
        """Restore an evicted node's Full device value from fresh KV indices
        during insert."""
        ct = BASE_COMPONENT_TYPE
        cd = node.component_data[ct]
        assert cd.value is None
        n = len(fresh_value)
        cd.value = fresh_value.clone()
        self.lru_lists[ct].insert_mru(node)
        self.component_evictable_size_[ct] += n
        self._update_evictable_leaf_sets(node)
        if node.parent is not None:
            self._update_evictable_leaf_sets(node.parent)

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
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = self.key_match_fn(node.key, key)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            if node.evicted:
                self._unevict_node_on_insert(node, value[:prefix_len])
            else:
                value_slice = value[:prefix_len]
                consumed_from = prefix_len
                # Let each component claim ownership of overlapping KV slots
                for component in self._components_tuple:
                    comp_consumed_from = component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )
                    consumed_from = min(consumed_from, comp_consumed_from)

                dup_start = max(0, params.prev_prefix_len - total_prefix_length)
                if dup_start < consumed_from:
                    self.token_to_kv_pool_allocator.free(
                        value_slice[dup_start:consumed_from]
                    )

                self._inc_hit_count(node, params.chunked)
                total_prefix_length += prefix_len

            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        is_new_leaf = False
        # Create new leaf for remaining suffix
        if len(key):
            if any(
                comp.should_skip_leaf_creation(
                    total_prefix_len=total_prefix_length,
                    key_len=len(key),
                    params=params,
                )
                for comp in self._components_tuple
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
        for component in self._components_tuple:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )
        if is_new_leaf:
            self._inc_hit_count(target_node, params.chunked)
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

        for comp in self._components_tuple:
            if comp.eviction_priority(is_leaf) <= trigger_priority:
                if comp is not trigger and comp.node_has_component_data(node):
                    assert node.component_data[comp.component_type].lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node, comp, is_leaf=is_leaf, tracker=tracker
                    )

        if is_leaf:
            parent = node.parent
            self._remove_leaf_from_parent(node)
            self._update_evictable_leaf_sets(parent)
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
        while cur != self.root_node and len(cur.children) == 0:
            # Convention: if ANY component has lost its device value, the entire
            # node is considered tombstoned and should be cascade-deleted,
            # because a partially-evicted node can no longer serve as a valid
            # match boundary.
            has_tombstone = any(
                not comp.node_has_component_data(cur)
                for comp in self.components.values()
            )
            if not has_tombstone:
                break

            # Don't delete nodes that are still locked (device or host).
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0
                for cd in cur.component_data
            ):
                break

            for comp in self.components.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur, comp, is_leaf=True, tracker=tracker
                    )
            self.evictable_host_leaves.discard(cur)
            self._remove_leaf_from_parent(cur)
            parent = cur.parent
            self._update_evictable_leaf_sets(parent)
            cur = parent

    def _for_each_component_lru(self, node: UnifiedTreeNode, lru_op):
        for ct in self.tree_components:
            if node.component_data[ct].value is not None:
                lru_op(self.lru_lists[ct], node)

    # ---- HiCache: Evict Helpers ----

    def _update_device_leaf_status(self, node: UnifiedTreeNode) -> None:
        """Update whether a node qualifies as an evictable device leaf for Full."""
        ct = BASE_COMPONENT_TYPE
        cd = node.component_data[ct]
        if node is self.root_node or cd.value is None or cd.lock_ref > 0:
            self.evictable_device_leaves.discard(node)
            return
        for child in node.children.values():
            if child.component_data[ct].value is not None:
                self.evictable_device_leaves.discard(node)
                return
        self.evictable_device_leaves.add(node)

    def _update_host_leaf_status(self, node: UnifiedTreeNode) -> None:
        """Update whether a node qualifies as an evictable host leaf."""
        if not node.evicted or not node.backuped or node is self.root_node:
            self.evictable_host_leaves.discard(node)
            return
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            self.evictable_host_leaves.discard(node)
            return
        for child in node.children.values():
            if child.evicted and child.backuped:
                self.evictable_host_leaves.discard(node)
                return
        self.evictable_host_leaves.add(node)

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """Convenience: update both device and host leaf status for a node."""
        self._update_device_leaf_status(node)
        self._update_host_leaf_status(node)

    def _evict_to_host(self, node: UnifiedTreeNode) -> None:
        """GPU→CPU demotion: release all device resources, node stays in tree."""
        assert not node.evicted and node.backuped
        for comp in self._components_tuple:
            if comp.node_has_component_data(node):
                comp.evict_component(node, is_leaf=False)
                lru = self.lru_lists[comp.component_type]
                if lru.in_list(node):
                    lru.remove_node(node)

        # After device eviction, add to host LRU for extra components
        # TODO5: 为什么需要这段？我看 HiMambaTree 里面没有?
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            cd = node.component_data[ct]
            if cd.host_value is not None:
                host_lru = self.host_lru_lists[ct]
                if not host_lru.in_list(node):
                    host_lru.insert_mru(node)
        self._update_evictable_leaf_sets(node)
        self._update_evictable_leaf_sets(node.parent)

    def _evict_device_leaf(self, node: UnifiedTreeNode) -> int:
        """Evict a device leaf node, choosing the right strategy:

        - backuped: demote to host via _evict_to_host (node stays in tree)
        - not backuped + write_back: write_backup first, then demote
        - not backuped + write_through: Cascade evict all components
        """
        num_full = len(node.component_data[BASE_COMPONENT_TYPE].value)
        if not node.backuped:
            if (
                self.cache_controller is not None
                and self.cache_controller.write_policy == "write_back"
            ):
                self.write_backup(node, write_back=True)
                self._evict_to_host(node)
                return num_full
            else:
                # TODO7: 这里跟 HiMambaRadixTree 的 evict_regular 是完全对其的吗？有漏的吗？
                # Cascade: evict trigger (Full) first, then cascade to others.
                trigger = self.components[BASE_COMPONENT_TYPE]
                tracker = {ct: 0 for ct in self.tree_components}
                self._evict_component_and_detach_lru(
                    node, trigger, is_leaf=True, tracker=tracker
                )
                self._cascade_evict(node, trigger, tracker)
                return tracker[BASE_COMPONENT_TYPE]
        self._evict_to_host(node)
        return num_full

    def _evict_host_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> int:
        """Atomically evict all components' host resources on a host leaf (C1)."""
        # TODO: 这段跟 Himamba Tree 也没有对其？没有正确 assert，没有正确更新 components lru?
        assert node.evicted and node.backuped
        total_freed = 0
        for comp in self._components_tuple:
            freed = comp.evict_component(node, is_leaf=True)
            tracker[comp.component_type] += freed
            total_freed += freed
        self.evictable_host_leaves.discard(node)
        self._remove_leaf_from_parent(node)
        self._iteratively_delete_tombstone_leaf(node, tracker)
        return total_freed

    # ---- HiCache: Backup / Restore ----

    def write_backup(
        self, node: UnifiedTreeNode, write_back: bool = False
    ) -> int:
        """Backup a node's data from device to host (D→H).
        Returns the number of host indices written, or 0 on failure."""
        if self.cache_controller is None:
            return 0

        # Backup invariant (write-through): parent must be backuped first
        if not write_back and (
            node.parent is not self.root_node and not node.parent.backuped
        ):
            return 0

        # Refresh host LRU position for extra components if already backed up
        # TODO：分析下这段为什么需要？
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            cd = node.component_data[ct]
            if cd.value is not None and cd.host_value is not None:
                host_lru = self.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.reset_node_mru(node)

        # Build per-component transfer descriptors
        transfers: list = []
        for comp in self._components_tuple:
            t = comp.build_hicache_transfers(node, HiCachePhase.BACKUP)
            if t:
                transfers.extend(t)

        # Write to host — retry after evicting host on first failure
        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        host_indices = self.cache_controller.write(
            device_value, node_id=node.id, extra_pools=transfers or None
        )
        if host_indices is None:
            self._evict_host(len(device_value))
            # TODO: 这里为什么需要再次 build_hicache_transfers?
            # Rebuild transfers (previous alloc may have been rolled back)
            transfers = []
            for comp in self._components_tuple:
                t = comp.build_hicache_transfers(node, HiCachePhase.BACKUP)
                if t:
                    transfers.extend(t)
            host_indices = self.cache_controller.write(
                device_value, node_id=node.id, extra_pools=transfers or None
            )
        if host_indices is None:
            return 0

        # Commit: store host indices and let components do bookkeeping
        node.component_data[BASE_COMPONENT_TYPE].host_value = host_indices.clone()
        for comp in self._components_tuple:
            comp.commit_hicache_transfer(
                node, HiCachePhase.BACKUP, transfers=transfers
            )

        # Async tracking
        self.ongoing_write_through[node.id] = node
        if not write_back:
            self.inc_lock_ref(node)
        return len(host_indices)

    def _evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> None:
        """Evict host resources for a specific component to free host pool space."""
        tracker: dict[ComponentType, int] = {ct: 0 for ct in self.tree_components}
        comp = self.components.get(component_type)
        if comp is not None:
            comp.drive_host_eviction(num_tokens, tracker)

    def _inc_hit_count(self, node: UnifiedTreeNode, chunked: bool = False) -> None:
        """Increment hit count; trigger write_backup when threshold reached."""
        if self.cache_controller is None:
            return
        if node.evicted or chunked:
            return
        if self.cache_controller.write_policy == "write_back":
            return
        node.hit_count += 1
        if not node.backuped and node.hit_count >= self.write_through_threshold:
            self.write_backup(node)

    def load_back(
        self,
        node: UnifiedTreeNode,
        mem_quota: Optional[int] = None,
        req=None,
    ) -> Optional[torch.Tensor]:
        """Load evicted KV data from host back to device (H→D).
        Pure orchestrator — component-specific logic is in
        build_hicache_transfers(RESTORE) / commit_hicache_transfer(RESTORE).
        Returns device indices on success, None if load is skipped/failed."""
        if self.cache_controller is None:
            return None

        last_hit_node = node
        nodes_to_load: list[UnifiedTreeNode] = []
        while node.evicted:
            assert node.backuped, f"No backup on evicted node {node.id}"
            nodes_to_load.insert(0, node)
            node = node.parent
        ancestor_node = node

        result = self.inc_lock_ref(ancestor_node)
        delta = result.delta or 0

        # Build per-component restore transfers
        transfers: list = []
        for comp in self._components_tuple:
            t = comp.build_hicache_transfers(
                last_hit_node, HiCachePhase.RESTORE, req=req
            )
            if t:
                transfers.extend(t)

        # Compute total host indices for the controller
        kv_transfers = [t for t in transfers if t.name == PoolName.KV]
        full_host_indices = (
            kv_transfers[0].host_indices if kv_transfers else
            torch.empty((0,), dtype=torch.int64, device="cpu")
        )

        logger.info(
            "load_back: kv_host_len=%d, nodes_to_load=%d, node_id=%d",
            len(full_host_indices),
            len(nodes_to_load),
            last_hit_node.id,
        )

        # Load from host to device — retry after device eviction
        full_device_indices = self.cache_controller.load(
            host_indices=full_host_indices,
            node_id=last_hit_node.id,
            extra_pools=transfers or None,
        )
        if full_device_indices is None:
            if len(full_host_indices) > 0:
                self.evict(EvictParams(num_tokens=len(full_host_indices)))
            # Rebuild transfers after eviction freed space
            transfers = []
            for comp in self._components_tuple:
                t = comp.build_hicache_transfers(
                    last_hit_node, HiCachePhase.RESTORE, req=req
                )
                if t:
                    transfers.extend(t)
            kv_transfers = [t for t in transfers if t.name == PoolName.KV]
            full_host_indices = (
                kv_transfers[0].host_indices if kv_transfers else
                torch.empty((0,), dtype=torch.int64, device="cpu")
            )
            full_device_indices = self.cache_controller.load(
                host_indices=full_host_indices,
                node_id=last_hit_node.id,
                extra_pools=transfers or None,
            )
        self.dec_lock_ref(ancestor_node)
        if full_device_indices is None:
            return None

        # Let components finalize restore (restore device values, LRU, etc.)
        for comp in self._components_tuple:
            comp.commit_hicache_transfer(
                last_hit_node,
                HiCachePhase.RESTORE,
                req=req,
                transfers=transfers,
                nodes_to_load=nodes_to_load,
            )

        self._update_evictable_leaf_sets(ancestor_node)
        self.inc_lock_ref(last_hit_node)
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        return full_device_indices

    # ---- HiCache: Async Event Management ----

    def writing_check(self, write_back: bool = False) -> None:
        """Poll write-through completions."""
        cc = self.cache_controller
        if cc is None:
            return

        if write_back:
            # Blocking: wait for all pending write-backs
            while self.ongoing_write_through:
                for _, finish_event, ack_list in cc.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        self.ongoing_write_through.pop(ack_id, None)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in cc.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1

        # TP sync: MIN across all ranks for consistent tree updates
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        finish_count = int(queue_size.item())

        # Process completed acks
        while finish_count > 0:
            _, finish_event, ack_list = cc.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                node = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(node)
            finish_count -= 1

    def loading_check(self) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None or not self.ongoing_load_back:
            return
        finish_count = 0
        for _, finish_event, ack_list in cc.ack_load_queue:
            if not finish_event.query():
                break
            finish_count += 1
            for ack_id in ack_list:
                node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(node)
        del cc.ack_load_queue[:finish_count]

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, UnifiedTreeNode]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node) tuple."""
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        req = params.req

        if last_node.evicted or params.host_hit_length > 0:
            logger.info(
                "init_load_back triggered: node_id=%d, host_hit_length=%d",
                last_node.id,
                params.host_hit_length,
            )
            loading_values = self.load_back(last_node, mem_quota, req=req)
            if loading_values is not None:
                logger.info(
                    "init_load_back success: loaded %d tokens for node %d",
                    len(loading_values),
                    last_node.id,
                )
                return loading_values, last_node

            # Fallback: walk up to non-evicted ancestor
            while last_node is not self.root_node and last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        self.writing_check()
        self.loading_check()

    def flush_write_through_acks(self) -> None:
        """Flush pending write-through acknowledgements."""
        self.writing_check()

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

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
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component_data[ct].value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                v = child.component_data[BASE_COMPONENT_TYPE].value
                if v is not None:
                    values.append(v)
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
            value = node.component_data[component_type].value
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

    def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def sanity_check(self):
        """Thorough sanity check: verify LRU membership, lock state, linked-list
        integrity, and evictable sizes for every component.
        Expensive — use only in tests or idle checks."""
        try:
            # 1. Collect all nodes from tree
            all_nodes = self._collect_all_nodes()

            for ct in self.tree_components:
                # 2. Basic size invariants
                assert (
                    self.component_evictable_size_[ct] >= 0
                ), f"component_evictable_size_[{ct}] = {self.component_evictable_size_[ct]} < 0"
                assert (
                    self.component_protected_size_[ct] >= 0
                ), f"component_protected_size_[{ct}] = {self.component_protected_size_[ct]} < 0"

                # 3. Verify LRU membership: tree nodes with data == LRU cache entries
                lru = self.lru_lists[ct]
                tree_ids = {
                    n.id
                    for n in all_nodes
                    if n != self.root_node and n.component_data[ct].value is not None
                }
                lru_ids = set(lru.cache.keys())
                assert tree_ids == lru_ids, (
                    f"[{ct}] LRU membership mismatch: "
                    f"in_tree_not_lru={tree_ids - lru_ids}, "
                    f"in_lru_not_tree={lru_ids - tree_ids}"
                )

                # 4. Walk LRU doubly-linked list: verify structural integrity
                #    and that all nodes are unlocked (idle check)
                visited = set()
                x = lru.head.lru_next[ct]
                prev = lru.head
                while x != lru.tail:
                    assert (
                        x.lru_prev[ct] == prev
                    ), f"[{ct}] broken prev link at node {x.id}"
                    assert (
                        x.id in lru.cache
                    ), f"[{ct}] node {x.id} in linked list but not in cache dict"
                    assert x.id not in visited, f"[{ct}] cycle detected at node {x.id}"
                    assert x.component_data[ct].lock_ref == 0, (
                        f"[{ct}] node {x.id} should not be locked when idle, "
                        f"lock_ref={x.component_data[ct].lock_ref}"
                    )
                    visited.add(x.id)
                    prev = x
                    x = x.lru_next[ct]
                assert len(visited) == len(lru.cache), (
                    f"[{ct}] linked list has {len(visited)} nodes, "
                    f"cache dict has {len(lru.cache)}"
                )

                # 5. Verify evictable size by walking unlocked LRU nodes
                recomputed = 0
                x = lru.get_lru_no_lock()
                while lru.in_list(x):
                    v = x.component_data[ct].value
                    recomputed += len(v) if v is not None else 0
                    x = lru.get_prev_no_lock(x)
                assert self.component_evictable_size_[ct] == recomputed, (
                    f"[{ct}] evictable_size_={self.component_evictable_size_[ct]} "
                    f"!= recomputed={recomputed}"
                )

        except Exception as e:
            logger.error(f"Unified RadixTree sanity check failed: {e}")
            self.pretty_print()
            raise

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{ct}={'yes' if node.component_data[ct].value is not None else 'no'}"
                for ct in self.tree_components
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component_data[BASE_COMPONENT_TYPE].lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))
