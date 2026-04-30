from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
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


logger = logging.getLogger(__name__)


def _debug_len(value) -> str:
    return "none" if value is None else str(len(value))


class SWAComponent(TreeComponent):
    """Sliding window attention component.

    Each SWA node stores translated SWA pool indices as its component
    value, independent of the full attention indices on the same tree node.
    When SWA data is evicted from an internal node the node is tombstoned
    — its SWA component value becomes None while the full attention
    value stays intact.
    """

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        assert isinstance(
            cache.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), f"SWAComponent requires SWATokenToKVPoolAllocator, got {type(cache.token_to_kv_pool_allocator)}"
        super().__init__(cache, params)
        self.sliding_window_size = params.sliding_window_size
        # HiCache state: set to host SWA pool when HiCache enabled
        self._swa_kv_pool_host = None

    component_type = ComponentType.SWA

    def _debug_node_state(self, node: UnifiedTreeNode) -> str:
        full = node.component_data[BASE_COMPONENT_TYPE]
        swa = node.component_data[self.component_type]
        parent_id = None if node.parent is None else node.parent.id
        key_len = 0 if node.key is None else len(node.key)
        return (
            f"node_id={node.id}, key_len={key_len}, parent={parent_id}, "
            f"children={len(node.children)}, evicted={node.evicted}, "
            f"full_dev={_debug_len(full.value)}, full_host={_debug_len(full.host_value)}, "
            f"full_lock={full.lock_ref}, swa_dev={_debug_len(swa.value)}, "
            f"swa_host={_debug_len(swa.host_value)}, swa_lock={swa.lock_ref}, "
            f"swa_host_lock={swa.host_lock_ref}, swa_uuid={swa.metadata.get('uuid')}, "
            f"dev_lru={self.cache.lru_lists[self.component_type].in_list(node)}, "
            f"host_lru={self.cache.host_lru_lists[self.component_type].in_list(node)}, "
            f"device_leaf={node in self.cache.evictable_device_leaves}, "
            f"host_leaf={node in self.cache.evictable_host_leaves}"
        )

    def _log_if_gap_created(self, node: UnifiedTreeNode, op: str) -> None:
        full = node.component_data[BASE_COMPONENT_TYPE]
        swa = node.component_data[self.component_type]
        if (
            full.value is not None
            and swa.value is None
            and swa.host_value is None
            and not swa.metadata.get("debug_gap_logged", False)
        ):
            swa.metadata["debug_gap_logged"] = True
            logger.warning("swa gap created: op=%s, %s", op, self._debug_node_state(node))

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full_indices
        )

    def _restore_device_value(self, node: UnifiedTreeNode, value: torch.Tensor) -> None:
        ct = self.component_type
        node.component_data[ct].value = value
        node.component_data[ct].metadata.pop("debug_gap_logged", None)
        host_lru = self.cache.host_lru_lists[ct]
        if host_lru.in_list(node):
            host_lru.remove_node(node)
        self.cache.lru_lists[ct].insert_mru(node)
        self.cache.component_evictable_size_[ct] += len(value)

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        sliding_window_size = self.sliding_window_size
        ct = self.component_type
        state = {"len": float("inf")}

        def validator(node: UnifiedTreeNode) -> bool:
            cd = node.component_data[ct]
            if cd.value is None and (not node.evicted or cd.host_value is None):
                state["len"] = 0
                return False
            state["len"] += len(node.key)
            return state["len"] >= sliding_window_size

        return validator

    def _collect_load_back_nodes(
        self, node: UnifiedTreeNode
    ) -> tuple[list[torch.Tensor], list[UnifiedTreeNode], int]:
        ct = self.component_type
        collected_leaf_first: list[torch.Tensor] = []
        nodes_leaf_first: list[UnifiedTreeNode] = []
        n_swa = 0
        covered = 0
        cur = node

        while cur is not self.cache.root_node and covered < self.sliding_window_size:
            cd = cur.component_data[ct]
            if cd.value is not None:
                covered += len(cd.value)
                cur = cur.parent
                continue
            if not cur.evicted or cd.host_value is None:
                break
            collected_leaf_first.append(cd.host_value)
            nodes_leaf_first.append(cur)
            n_swa += len(cd.host_value)
            covered += len(cd.host_value)
            cur = cur.parent

        return collected_leaf_first, nodes_leaf_first, n_swa

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        return result

    def _update_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        mapping = getattr(
            self.cache.token_to_kv_pool_allocator, "full_to_swa_index_mapping", None
        )
        if mapping is not None:
            mapping[full_indices.to(torch.int64)] = swa_indices.to(torch.int64)

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

        is_tombstone = node.component_data[self.component_type].value is None
        if not is_tombstone:
            return prefix_len

        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            node.component_data[self.component_type].lock_ref == 0
        ), f"tombstone {self.component_type} lock_ref should be 0, node {node.id}"
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{self.component_type}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        if swa_evicted_seqlen <= total_prefix_len:
            # Branch 1: entire value_slice is within SWA window — recover
            self.cache.token_to_kv_pool_allocator.free(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice.clone()
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)
            return 0
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # Branch 2: value_slice[start_idx:] is within SWA window — partial recover
            start_idx = swa_evicted_seqlen - total_prefix_len
            self.cache.token_to_kv_pool_allocator.free(
                node.component_data[BASE_COMPONENT_TYPE].value[start_idx:]
            )
            new_parent = self.cache._split_node(node.key, node, start_idx)
            self._log_if_gap_created(new_parent, "overlap_split_parent")
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice[
                start_idx:
            ].clone()
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)
            return start_idx
        else:
            # Branch 3: entire value_slice is outside SWA window — not consumed
            self._log_if_gap_created(node, "overlap_skip")
            return prefix_len

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        return params.swa_evicted_seqlen >= total_prefix_len + key_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        # _unevict_node_on_insert already wrote the request's fresh KV slice
        # into the base value. We just need to rebuild SWA from that slice for
        # the in-window portion. There is no old SWA slot to free here.
        ct = self.component_type
        if node.component_data[ct].value is not None:
            return
        assert (
            node.component_data[ct].lock_ref == 0
        ), f"tombstone {ct} lock_ref should be 0 on unevict, node {node.id}"
        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{ct}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        full_value = node.component_data[BASE_COMPONENT_TYPE].value
        if swa_evicted_seqlen <= total_prefix_len:
            swa_value = self._translate_full_to_swa(full_value)
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            start_idx = swa_evicted_seqlen - total_prefix_len
            new_parent = self.cache._split_node(node.key, node, start_idx)
            self._log_if_gap_created(new_parent, "unevict_split_parent")
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            swa_value = self._translate_full_to_swa(full_value)
        else:
            self._log_if_gap_created(node, "unevict_skip")
            return
        self._restore_device_value(node, swa_value)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        if not is_new_leaf:
            return

        node_start = result.prefix_len
        split_pos = params.swa_evicted_seqlen - node_start

        if split_pos <= 0:
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
        elif split_pos < len(node.key):
            # Node straddles the SWA eviction boundary
            # Split into parent (tombstone, no SWA) and child (with SWA)
            # After _split_node, `node` becomes the child
            new_parent = self.cache._split_node(node.key, node, split_pos)
            self._log_if_gap_created(new_parent, "insert_split_parent")
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
        else:
            self._log_if_gap_created(node, "insert_skip")

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component_data[self.component_type].lock_ref = child.component_data[
            self.component_type
        ].lock_ref

        child_swa_value = child.component_data[self.component_type].value
        if child_swa_value is not None:
            split_len = len(new_parent.key)
            new_parent.component_data[self.component_type].value = child_swa_value[
                :split_len
            ].clone()
            child.component_data[self.component_type].value = child_swa_value[
                split_len:
            ].clone()
        else:
            new_parent.component_data[self.component_type].value = None

        child_swa_host_value = child.component_data[self.component_type].host_value
        if child_swa_host_value is not None:
            split_len = len(new_parent.key)
            new_parent.component_data[self.component_type].host_value = (
                child_swa_host_value[:split_len].clone()
            )
            child.component_data[self.component_type].host_value = child_swa_host_value[
                split_len:
            ].clone()
            host_lru = self.cache.host_lru_lists[self.component_type]
            if new_parent.component_data[self.component_type].value is None:
                host_lru.insert_mru(new_parent)
            if (
                child.component_data[self.component_type].value is None
                and not host_lru.in_list(child)
            ):
                host_lru.insert_mru(child)

        # parent inherits the swa_uuid from child for swa lock ref
        new_parent.component_data[self.component_type].metadata["uuid"] = (
            child.component_data[self.component_type].metadata.get("uuid")
        )
        child.component_data[self.component_type].metadata.pop("uuid", None)

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            # Pass full indices to free_swa so slots with no SWA pair are
            # skipped. Freeing swa_value directly would double free those
            # entries since they all map to the same sentinel slot.
            self.cache.token_to_kv_pool_allocator.free_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            freed = len(cd.value)
            self.cache.component_evictable_size_[ct] -= freed
            cd.value = None
            if target is EvictLayer.DEVICE:
                self._log_if_gap_created(node, "device_evict")

        # Host layer
        host_lru = self.cache.host_lru_lists[ct]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._swa_kv_pool_host is not None:
                self._swa_kv_pool_host.free(cd.host_value)
            cd.host_value = None
            if host_lru.in_list(node):
                host_lru.remove_node(node)
            if EvictLayer.DEVICE not in target:
                self._log_if_gap_created(node, "host_evict")

        # After device tombstone: if host_value remains, move into host LRU
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
        ):
            if not host_lru.in_list(node):
                host_lru.insert_mru(node)

        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 1

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.swa_num_tokens
        ct = self.component_type
        lru = self.cache.lru_lists[ct]
        x = lru.get_lru_no_lock()
        while tracker[ct] < request and x is not None and lru.in_list(x):
            assert x.component_data[ct].value is not None
            if x in self.cache.evictable_device_leaves:
                # D-leaf: atomic eviction of all components
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_device_leaf(x, tracker)
                if not lru.in_list(x_next):
                    x_next = lru.get_lru_no_lock()
                x = x_next
            else:
                # Internal: tombstone SWA + cascade
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        ct = self.component_type
        root = self.cache.root_node
        sliding_window_size = self.sliding_window_size
        swa_lock_size = 0
        swa_uuid_for_lock = None

        # Tombstoned nodes (cd.value is None) have no SWA chunk to protect
        # skip them and keep walking up. This path is hit when HiCache
        # backs up a FULL present internal node whose SWA was already evicted.
        cur = node
        while cur != root and swa_lock_size < sliding_window_size:
            comp = cur.component_data[ct]
            if comp.value is None:
                cur = cur.parent
                continue
            if comp.lock_ref == 0:
                key_len = len(cur.key)
                self.cache.component_evictable_size_[ct] -= key_len
                self.cache.component_protected_size_[ct] += key_len
            comp.lock_ref += 1
            swa_lock_size += len(cur.key)
            if swa_lock_size >= sliding_window_size:
                if comp.metadata.get("uuid") is None:
                    comp.metadata["uuid"] = next_component_uuid()
                swa_uuid_for_lock = comp.metadata["uuid"]
            cur = cur.parent

        result.swa_uuid_for_lock = swa_uuid_for_lock
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        ct = self.component_type
        root = self.cache.root_node
        swa_uuid_for_lock = params.swa_uuid_for_lock if params else None
        dec_swa = True
        dec_tokens = 0
        dec_nodes: list[str] = []
        stopped_by_uuid = False

        # lock_ref == 0 means acquire_component_lock skipped this node
        # (tombstone at acquire time) or load_back revived a tombstone between
        # acquire and release. Either way, there is nothing for us to undo here.
        cur = node
        while cur != root and dec_swa:
            comp = cur.component_data[ct]
            if comp.lock_ref == 0:
                cur = cur.parent
                continue
            if comp.lock_ref == 1:
                key_len = len(cur.key)
                self.cache.component_evictable_size_[ct] += key_len
                self.cache.component_protected_size_[ct] -= key_len
            comp.lock_ref -= 1
            dec_tokens += len(cur.key)
            if len(dec_nodes) < 8:
                dec_nodes.append(
                    f"{cur.id}:{len(cur.key)}:{comp.lock_ref}:"
                    f"{comp.metadata.get('uuid')}"
                )
            if swa_uuid_for_lock and comp.metadata.get("uuid") == swa_uuid_for_lock:
                stopped_by_uuid = True
                dec_swa = False
            cur = cur.parent

        if (
            (swa_uuid_for_lock is None and dec_tokens >= self.sliding_window_size)
            or (
                swa_uuid_for_lock is not None
                and dec_tokens > 0
                and not stopped_by_uuid
            )
        ):
            logger.warning(
                "swa suspicious release: boundary_uuid=%s, stopped_by_uuid=%s, "
                "dec_tokens=%d, window=%d, dec_path=%s, %s",
                swa_uuid_for_lock,
                stopped_by_uuid,
                dec_tokens,
                self.sliding_window_size,
                dec_nodes,
                self._debug_node_state(node),
            )

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

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            if cd.value is None:
                return None
            # cd.value already holds SWA-pool indices (translated at insert time).
            # Host pool indexing wants int64.
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=cd.value.to(torch.int64),
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            collected_leaf_first, nodes_leaf_first, n_swa = (
                self._collect_load_back_nodes(node)
            )
            if not collected_leaf_first:
                return None
            collected_leaf_first.reverse()
            nodes_leaf_first.reverse()
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.cat(collected_leaf_first),
                    device_indices=None,
                    swa_suffix_tokens=n_swa,
                    nodes_to_load=nodes_leaf_first,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                cd = node.component_data[ct]
                if cd.host_value is None:
                    cd.host_value = transfers[0].host_indices.clone()
                cd.metadata.pop("debug_gap_logged", None)
            return

        if phase == CacheTransferPhase.LOAD_BACK:
            if not transfers or transfers[0].device_indices is None:
                return
            xfer = transfers[0]
            device_indices = xfer.device_indices
            assert (
                len(device_indices) == xfer.swa_suffix_tokens
            ), (
                "SWA loadback device indices mismatch: "
                f"{len(device_indices)=}, {xfer.swa_suffix_tokens=}"
            )
            offset = 0
            for n in xfer.nodes_to_load or []:
                cd_n = n.component_data[ct]
                n_tokens = len(cd_n.host_value)
                full_value = n.component_data[BASE_COMPONENT_TYPE].value
                assert (
                    full_value is not None and len(full_value) == n_tokens
                ), f"SWA loadback requires matching Full device value on node {n.id}"
                swa_value = device_indices[offset : offset + n_tokens].clone()
                self._update_full_to_swa_mapping(full_value, swa_value)
                self._restore_device_value(
                    n,
                    swa_value,
                )
                offset += n_tokens
            assert offset == xfer.swa_suffix_tokens
            return

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict SWA host resources.
        Internal nodes: private tombstone (free SWA host only).
        Host leaves: atomic eviction via _evict_host_leaf."""
        ct = self.component_type
        host_lru = self.cache.host_lru_lists[ct]
        x = host_lru.get_lru_no_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            x_next = host_lru.get_prev_no_lock(x)
            cd = x.component_data[ct]
            if x in self.cache.evictable_host_leaves:
                self.cache._evict_host_leaf(x, tracker)
            else:
                assert cd.host_value is not None
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)
            x = x_next
