from __future__ import annotations

import json
import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, List, NamedTuple, Optional

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.external_cache_controller import (
    BackupRequest,
    BackupResult,
    BaseExternalCacheController,
    ExternalCacheProgress,
    ExternalCacheTreeOps,
    LoadBackRequest,
    LoadBackResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()


class CacheOperation(BaseCacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[list[PoolTransfer]]:
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {}
        for op in ops:
            for t in op.pool_transfers or []:
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=ts[0].name,
                host_indices=cat_or_none(t.host_indices for t in ts),
                device_indices=cat_or_none(t.device_indices for t in ts),
                keys=[k for t in ts if t.keys for k in t.keys] or None,
                hit_policy=ts[0].hit_policy,
                indices_from_pool=ts[0].indices_from_pool,
            )
            for ts in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        self.pool_transfers_done = not bool(pool_transfers)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class OngoingWriteThrough(NamedTuple):
    """Tracks an in-flight D2H write-through operation."""

    node: Any
    lock_params: Any
    publish_nodes: list[Any]


class HybridCacheController(BaseHiCacheController, BaseExternalCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
    ):
        startup_storage_backend = storage_backend
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        self.ongoing_write_through: dict[int, Any] = {}
        self.ongoing_load_back: dict[int, Any] = {}
        self.ongoing_backup: dict[int, Any] = {}
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.ongoing_prefetch: dict[str, Any] = {}
        self._active_cache: Any = None
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def _start_storage_threads(self):
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                path = storage_backend_extra_config[1:]
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb" if ext == ".toml" else "r") as f:
                    if ext == ".json":
                        extra_config = json.load(f)
                    elif ext == ".toml":
                        import tomllib

                        extra_config = tomllib.load(f)
                    elif ext in (".yaml", ".yml"):
                        import yaml

                        extra_config = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file {path} (config format: {ext})"
                        )
            else:
                extra_config = json.loads(storage_backend_extra_config)

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        self.storage_backend.clear()
        return True

    def _init_extra_host_mem_release_queues(self) -> None:
        self.extra_host_mem_release_queues = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ):
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            if (
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        super().reset()
        self.ongoing_write_through.clear()
        self.ongoing_load_back.clear()
        self.ongoing_backup.clear()
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.ongoing_prefetch.clear()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def shutdown(self) -> None:
        if self.enable_storage:
            self._stop_storage_threads()

    def _cache_from_tree_ops(self, tree_ops: ExternalCacheTreeOps) -> Any:
        cache = getattr(tree_ops, "cache", None)
        if cache is None:
            raise TypeError("HybridCacheController requires UnifiedRadixCache tree ops")
        return cache

    def _get_node(self, tree_ops: ExternalCacheTreeOps, node_id: int) -> Any:
        get_node = getattr(tree_ops, "get_node", None)
        if get_node is None:
            raise TypeError("HybridCacheController requires node-resolving tree ops")
        return get_node(node_id)

    def write_backup(
        self, request: BackupRequest, tree_ops: ExternalCacheTreeOps
    ) -> BackupResult:
        cache = self._cache_from_tree_ops(tree_ops)
        self._active_cache = cache
        node = self._get_node(tree_ops, request.node_id)

        if not request.write_back and (
            node.parent is not cache.root_node and not node.parent.backuped
        ):
            parent_result = self.write_backup(
                BackupRequest(node_id=node.parent.id), tree_ops
            )
            if parent_result.backed_up_tokens <= 0:
                return BackupResult()

        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in cache._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                node, CacheTransferPhase.BACKUP_HOST
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers
        sidecar_xfers = cache._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

        kv_tokens = len(device_value)
        host_avail = self.mem_pool_host.available_size()
        if host_avail < kv_tokens:
            needed = kv_tokens - host_avail
            evicted = cache.evict_host(needed)
            if evicted < needed:
                return BackupResult()

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        host_indices = self.write(
            device_value, node_id=node.id, extra_pools=aux_xfers or None
        )
        if host_indices is None:
            return BackupResult()

        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        cache.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[kv_xfer],
        )
        for ct, xfers in comp_xfers.items():
            cache.components[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=xfers,
            )

        lock_params = None
        if not request.write_back:
            lock_params = cache.inc_lock_ref(node).to_dec_params()
        self.track_write_through_node(node, lock_params)
        return BackupResult(backed_up_tokens=len(host_indices))

    def track_write_through_node(self, node: Any, lock_params: Any) -> None:
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = OngoingWriteThrough(
            node, lock_params, [node]
        )

    def replace_pending_write_through_node(
        self, old_node: Any, new_nodes: list[Any]
    ) -> None:
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return

        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node, lock_params, publish_nodes = pending
        updated_nodes = []
        replaced = False
        for node in publish_nodes:
            if node is old_node:
                updated_nodes.extend(new_nodes)
                replaced = True
            else:
                updated_nodes.append(node)

        if not replaced:
            return

        for node in new_nodes:
            node.write_through_pending_id = ack_id
        self.ongoing_write_through[ack_id] = OngoingWriteThrough(
            lock_node,
            lock_params,
            updated_nodes,
        )

    def load_back(
        self, request: LoadBackRequest, tree_ops: ExternalCacheTreeOps
    ) -> LoadBackResult:
        cache = self._cache_from_tree_ops(tree_ops)
        self._active_cache = cache
        best_match_node = self._get_node(tree_ops, request.node_id)
        req = getattr(tree_ops, "current_req", None)
        start_time = time.perf_counter()
        host_anchor_params = cache.inc_host_lock_ref(best_match_node).to_dec_params()
        kv_xfer = cache.components[BASE_COMPONENT_TYPE].build_hicache_transfers(
            best_match_node, CacheTransferPhase.LOAD_BACK
        )[0]

        result = cache.inc_lock_ref(best_match_node)
        ancestor_lock_params = result.to_dec_params()
        kv_tokens = len(kv_xfer.host_indices)

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in cache._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                best_match_node, CacheTransferPhase.LOAD_BACK, req=req
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers
        sidecar_xfers = cache._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )

        if (kv_tokens < cache.load_back_threshold and not comp_xfers) or (
            request.mem_quota is not None and kv_tokens > request.mem_quota + result.delta
        ):
            cache.dec_lock_ref(best_match_node, ancestor_lock_params)
            cache.dec_host_lock_ref(best_match_node, host_anchor_params)
            return LoadBackResult()

        if cache.supports_swa():
            avail = cache.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = cache.token_to_kv_pool_allocator.available_size()
        if avail < kv_tokens:
            needed = kv_tokens - avail
            evict_result = cache.evict(EvictParams(num_tokens=needed))
            if evict_result.num_tokens_evicted < needed:
                cache.dec_lock_ref(best_match_node, ancestor_lock_params)
                cache.dec_host_lock_ref(best_match_node, host_anchor_params)
                return LoadBackResult()

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        device_indices = self.load(
            host_indices=kv_xfer.host_indices,
            node_id=best_match_node.id,
            extra_pools=aux_xfers or None,
        )

        cache.dec_lock_ref(best_match_node, ancestor_lock_params)
        if device_indices is None:
            cache.dec_host_lock_ref(best_match_node, host_anchor_params)
            return LoadBackResult()

        kv_xfer.device_indices = device_indices
        cache.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            best_match_node,
            CacheTransferPhase.LOAD_BACK,
            [kv_xfer],
        )
        for node in kv_xfer.nodes_to_load or ():
            cache._record_store_event(node, medium=StorageMedium.GPU)
        for ct, xfers in comp_xfers.items():
            cache.components[ct].commit_hicache_transfer(
                best_match_node,
                CacheTransferPhase.LOAD_BACK,
                xfers,
            )

        cache._update_evictable_leaf_sets(best_match_node)
        self.ongoing_load_back[best_match_node.id] = (
            best_match_node,
            cache.inc_lock_ref(best_match_node).to_dec_params(),
            host_anchor_params,
        )

        if cache.metrics_collector is not None:
            cache.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            cache.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return LoadBackResult(loaded=True, device_indices=device_indices)

    def finish_write_through_ack(
        self,
        ack_id: int,
        *,
        publish_host_node: Callable[[Any], None],
        release_node_lock: Callable[[Any, Any], None],
        enqueue_storage_backup: Optional[Callable[[Any], None]] = None,
    ) -> None:
        lock_node, lock_params, publish_nodes = self.ongoing_write_through.pop(ack_id)
        for node in publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            publish_host_node(node)
        if lock_params is not None:
            release_node_lock(lock_node, lock_params)
        if enqueue_storage_backup is not None:
            for node in publish_nodes:
                enqueue_storage_backup(node)

    def finish_load_back_ack(
        self,
        ack_id: int,
        *,
        release_node_lock: Callable[[Any, Any], None],
        release_host_lock: Callable[[Any, Any], None],
    ) -> None:
        node, lock_params, host_lock_params = self.ongoing_load_back.pop(ack_id)
        release_node_lock(node, lock_params)
        release_host_lock(node, host_lock_params)

    def _decrement_prefetch_tokens_occupied(self, num_tokens: int) -> None:
        self.prefetch_tokens_occupied -= num_tokens
        if self.prefetch_tokens_occupied < 0:
            self.prefetch_tokens_occupied = 0

    def finish_prefetch_progress(
        self,
        req_id: str,
        *,
        completed_tokens: int,
        min_completed_tokens: int,
        matched_tokens: int,
        loaded_tokens: int,
        release_host_lock: Callable[[Any, Any], None],
    ) -> int:
        (
            last_host_node,
            prefetch_key,
            host_indices,
            _operation,
            anchor_lock_params,
            _comp_xfers,
        ) = self.ongoing_prefetch.pop(req_id)
        self.mem_pool_host.free(host_indices[:matched_tokens])
        self.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        release_host_lock(last_host_node, anchor_lock_params)
        self._decrement_prefetch_tokens_occupied(len(prefetch_key))
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_tokens
        return self.prefetch_tokens_occupied

    def mark_prefetch_terminated(self, req_id: str) -> None:
        info = self.ongoing_prefetch.get(req_id)
        if info is None:
            return
        operation = info.operation if hasattr(info, "operation") else info[3]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def release_aborted_prefetch(
        self,
        req_id: str,
        *,
        barrier_attn_groups: Callable[[], None],
        release_host_lock: Callable[[Any, Any], None],
    ) -> None:
        self.prefetch_loaded_tokens_by_reqid.pop(req_id, None)
        if req_id not in self.ongoing_prefetch:
            return

        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.terminate_prefetch(operation)
        barrier_attn_groups()
        release_host_lock(last_host_node, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.append_host_mem_release(
            host_indices=host_indices[:completed_tokens],
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
        )
        self._decrement_prefetch_tokens_occupied(len(prefetch_key))

    def revoke_prefetch(
        self,
        req_id: str,
        *,
        release_host_lock: Callable[[Any, Any], None],
    ) -> bool:
        info = self.ongoing_prefetch.pop(req_id, None)
        if info is None:
            return False
        (
            last_host_node,
            prefetch_key,
            _host_indices,
            _operation,
            anchor_lock_params,
            comp_xfers,
        ) = info
        self.append_host_mem_release(
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers]
        )
        release_host_lock(last_host_node, anchor_lock_params)
        self._decrement_prefetch_tokens_occupied(len(prefetch_key))
        return True

    def poll(
        self, tree_ops: ExternalCacheTreeOps, *, wait: bool = False
    ) -> ExternalCacheProgress:
        cache = self._cache_from_tree_ops(tree_ops)
        self._active_cache = cache
        if wait:
            self._poll_write_acks(cache, wait=True)
            return ExternalCacheProgress()

        completed_writes = self._poll_write_acks(cache)
        completed_loads = self._poll_load_acks(cache)
        completed_storage_ops = 0
        if self.enable_storage:
            completed_storage_ops = self._drain_storage_control_queues(cache)
        return ExternalCacheProgress(
            completed_writes=completed_writes,
            completed_loads=completed_loads,
            completed_storage_ops=completed_storage_ops,
        )

    def _poll_write_acks(self, cache: Any, *, wait: bool = False) -> int:
        if wait:
            while self.ongoing_write_through:
                for _, finish_event, ack_list in self.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        if ack_id in self.ongoing_write_through:
                            self.finish_write_through_ack(
                                ack_id,
                                publish_host_node=lambda node: cache._record_store_event(
                                    node, medium=StorageMedium.CPU
                                ),
                                release_node_lock=cache.dec_lock_ref,
                                enqueue_storage_backup=(
                                    self.write_backup_storage
                                    if cache.enable_storage
                                    else None
                                ),
                            )
                self.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return 0

        finish_count = 0
        if cache.pp_rank == 0:
            for _, finish_event, _ in self.ack_write_queue:
                if not finish_event.query():
                    break
                finish_count += 1

        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        cache._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        completed = 0
        while finish_count > 0:
            _, finish_event, ack_list = self.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self.finish_write_through_ack(
                    ack_id,
                    publish_host_node=lambda node: cache._record_store_event(
                        node, medium=StorageMedium.CPU
                    ),
                    release_node_lock=cache.dec_lock_ref,
                    enqueue_storage_backup=(
                        self.write_backup_storage if cache.enable_storage else None
                    ),
                )
                completed += 1
            finish_count -= 1
        return completed

    def _poll_load_acks(self, cache: Any) -> int:
        finish_count = 0
        if cache.pp_rank == 0:
            for _, finish_event, _ in self.ack_load_queue:
                if not finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        cache._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        completed = 0
        while finish_count > 0:
            _, finish_event, ack_list = self.ack_load_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self.finish_load_back_ack(
                    ack_id,
                    release_node_lock=cache.dec_lock_ref,
                    release_host_lock=cache.dec_host_lock_ref,
                )
                completed += 1
            finish_count -= 1
        return completed

    def _drain_storage_control_queues(self, cache: Any) -> int:
        extra_release_queues = getattr(self, "extra_host_mem_release_queues", {})
        extra_pool_names = list(extra_release_queues)
        local_qsize_list = [
            self.prefetch_revoke_queue.qsize(),
            self.ack_backup_queue.qsize(),
            self.host_mem_release_queue.qsize(),
            *[
                extra_release_queues[pool_name].qsize()
                for pool_name in extra_pool_names
            ],
        ]
        qsizes = torch.tensor(
            local_qsize_list,
            dtype=torch.int,
        )
        cache._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)
        qsize_list = list(map(int, qsizes.tolist()))
        n_revoke, n_backup, n_release = qsize_list[:3]
        extra_counts = {
            pool_name: qsize_list[3 + idx]
            for idx, pool_name in enumerate(extra_pool_names)
        }
        return self._drain_storage_control_queues_impl(
            cache,
            n_revoke,
            n_backup,
            n_release,
            extra_counts,
            log_metrics=True,
        )

    def _drain_storage_control_queues_impl(
        self,
        cache: Any,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> int:
        def _drain_queue(q: Queue, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        drained_total = 0
        for req_id in _drain_queue(self.prefetch_revoke_queue, n_revoke):
            if self.revoke_prefetch(
                req_id,
                release_host_lock=cache.dec_host_lock_ref,
            ):
                drained_total += 1

        for operation in _drain_queue(self.ack_backup_queue, n_backup):
            drained_total += 1
            entry = self.ongoing_backup.pop(operation.id, None)
            if entry is not None:
                node, lock_params = entry
                cache.dec_host_lock_ref(node, lock_params)
            if (
                log_metrics
                and cache.enable_storage_metrics
                and cache.storage_metrics_collector is not None
            ):
                cache.storage_metrics_collector.log_backuped_tokens(
                    operation.completed_tokens
                )

        host_indices_list = []
        for host_indices in _drain_queue(self.host_mem_release_queue, n_release):
            drained_total += 1
            host_indices_list.append(host_indices)
        if host_indices_list:
            self.mem_pool_host.free(torch.cat(host_indices_list, dim=0))

        if extra_release_counts:
            for pool_name, limit in extra_release_counts.items():
                release_queue = self.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                host_indices_list = []
                for host_indices in _drain_queue(release_queue, limit):
                    drained_total += 1
                    host_indices_list.append(host_indices)
                if host_indices_list:
                    entry = self.mem_pool_host.entry_map.get(pool_name)
                    if entry is not None:
                        entry.host_pool.free(torch.cat(host_indices_list, dim=0))
        return drained_total

    def write_backup_storage(self, node: Any) -> None:
        if not self.enable_storage or not node.backuped:
            return

        cache = self._active_cache
        prefix_keys = None
        if cache.hicache_storage_pass_prefix_keys:
            prefix_keys = node.get_prefix_hash_values(node.parent)

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in cache._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                node,
                CacheTransferPhase.BACKUP_STORAGE,
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers

        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=node.component_data[BASE_COMPONENT_TYPE].host_value,
            keys=node.hash_value,
        )
        sidecar_xfers = cache._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, comp_xfers
        )
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        operation_id = self.write_storage(
            node.component_data[BASE_COMPONENT_TYPE].host_value,
            node.key.token_ids,
            node.hash_value,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        self.ongoing_backup[operation_id] = (
            node,
            cache.inc_host_lock_ref(node).to_dec_params(),
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: Any,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
    ) -> None:
        cache = self._active_cache
        if not cache.enable_storage:
            return

        extra_key = last_host_node.key.extra_key if last_host_node.key else None
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=cache.is_eagle,
        ).page_aligned(cache.page_size)
        prefetch_length = len(prefetch_key)
        if prefetch_length < cache.prefetch_threshold or self.prefetch_rate_limited():
            return

        anchor_lock_params = cache.inc_host_lock_ref(last_host_node).to_dec_params()
        host_indices = self.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            cache.evict_host(prefetch_length)
            host_indices = self.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            available_size = self.mem_pool_host.available_size()
            prefetch_length = available_size - (available_size % cache.page_size)
            if prefetch_length >= cache.prefetch_threshold:
                prefetch_key = prefetch_key[:prefetch_length]
                host_indices = self.mem_pool_host.alloc(prefetch_length)
            else:
                cache.dec_host_lock_ref(last_host_node, anchor_lock_params)
                return
        if host_indices is None:
            cache.dec_host_lock_ref(last_host_node, anchor_lock_params)
            return

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        alloc_failed = False
        for comp in cache._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                last_host_node,
                CacheTransferPhase.PREFETCH,
                token_ids=prefetch_key.token_ids,
                prefetch_tokens=len(prefetch_key),
                last_hash=last_hash,
            )
            if transfers == []:
                alloc_failed = True
                break
            if transfers:
                comp_xfers[comp.component_type] = transfers
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        sidecar_xfers = cache._build_sidecar_transfers(
            CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
        )
        if alloc_failed:
            self.append_host_mem_release(
                host_indices=host_indices,
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            cache.dec_host_lock_ref(last_host_node, anchor_lock_params)
            return

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        operation = self.prefetch(
            req_id,
            host_indices,
            prefetch_key,
            last_hash,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        self.prefetch_tokens_occupied += len(prefetch_key)

    def _prefetch_timeout_check_linear_func(
        self, cache: Any, operation: PrefetchOperation
    ) -> bool:
        return (
            time.monotonic() - operation.start_time
            > cache.prefetch_timeout_base
            + len(operation.hash_value) * cache.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, cache: Any, operation: PrefetchOperation) -> bool:
        if cache.prefetch_stop_policy == "best_effort":
            return True

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * cache.page_size
            )

        if cache.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif cache.prefetch_stop_policy == "timeout":
            can_terminate = completed or self._prefetch_timeout_check_linear_func(
                cache, operation
            )
        else:
            return True
        if (
            completed
            and getattr(operation, "pool_transfers", None)
            and not getattr(operation, "pool_transfers_done", True)
        ):
            can_terminate = False

        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        cache._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        return can_terminate or operation_terminated

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        cache = self._active_cache
        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return True
        if not self.can_terminate_prefetch(cache, operation):
            return False

        completed_tokens, hash_value = self.terminate_prefetch(operation)
        min_completed_tokens = completed_tokens
        hit_pages = operation.pool_storage_result.extra_pool_hit_pages
        if cache.tp_world_size > 1:
            sidecar_pools = [t.name for xfers in comp_xfers.values() for t in xfers]
            packed = torch.tensor(
                [completed_tokens] + [hit_pages.get(p, 0) for p in sidecar_pools],
                dtype=torch.int,
            )
            cache._all_reduce_attn_groups(packed, torch.distributed.ReduceOp.MIN)
            min_completed_tokens = int(packed[0].item())
            for i, p in enumerate(sidecar_pools, start=1):
                hit_pages[p] = int(packed[i].item())

        fetched_key = prefetch_key[:min_completed_tokens]
        insert_result = cache._insert_helper_host(
            last_host_node,
            fetched_key,
            host_indices[:min_completed_tokens],
            hash_value[: min_completed_tokens // cache.page_size],
        )

        for ct, xfers in comp_xfers.items():
            cache.components[ct].commit_hicache_transfer(
                last_host_node,
                CacheTransferPhase.PREFETCH,
                xfers,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )

        loaded_from_storage = min_completed_tokens - insert_result.prefix_len
        occupied = self.finish_prefetch_progress(
            req_id,
            completed_tokens=completed_tokens,
            min_completed_tokens=min_completed_tokens,
            matched_tokens=insert_result.prefix_len,
            loaded_tokens=loaded_from_storage,
            release_host_lock=cache.dec_host_lock_ref,
        )
        logger.info(
            "HiCache prefetch success req=%s completed_local=%d completed_synced=%d matched=%d loaded=%d tail_release=%d occupied=%d",
            req_id,
            completed_tokens,
            min_completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            completed_tokens - min_completed_tokens,
            occupied,
        )
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return True

    def abort_request(self, request_id: str, tree_ops: ExternalCacheTreeOps) -> None:
        self._active_cache = self._cache_from_tree_ops(tree_ops)
        self.release_aborted_prefetch(
            request_id,
            barrier_attn_groups=self._active_cache._barrier_attn_groups,
            release_host_lock=self._active_cache.dec_host_lock_ref,
        )

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=True,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        op = CacheOperation.merge_ops(self.write_queue)
        # Page-first write-back JIT kernels can keep destination host indices on CPU.
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            host_indices = op.host_indices
            device_indices = op.device_indices
            resolved_pool_transfers = op.pool_transfers
        else:
            host_indices, device_indices, resolved_pool_transfers = (
                self.move_hybrid_indices(op)
            )
        self.write_queue.clear()
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                self.io_backend,
                pool_transfers=resolved_pool_transfers,
            )
            if self.has_draft and host_indices.numel() > 0:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            self._record_transfer_indices_on_stream(
                self.write_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        need_load_kv = host_indices.numel() > 0

        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None

        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=False,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                full_allocator.free(device_indices)
            return None

        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                    pool_transfers=resolved_pool_transfers,
                )
                if (
                    self.has_draft
                    and host_indices.numel() > 0
                    and i < self.mem_pool_host_draft.layer_num
                ):
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            self._record_transfer_indices_on_stream(
                self.load_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def begin_pending_loads(self) -> int:
        return self.start_loading()

    def _record_transfer_indices_on_stream(
        self,
        stream: torch.Stream,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)
        for transfer in pool_transfers or []:
            if transfer.host_indices is not None and transfer.host_indices.is_cuda:
                transfer.host_indices.record_stream(stream)
            if transfer.device_indices is not None and transfer.device_indices.is_cuda:
                transfer.device_indices.record_stream(stream)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        hash_value = []
        for start in range(0, len(operation.token_ids), self.page_size):
            last_hash = self.get_hash_str(
                operation.token_ids[start : start + self.page_size], last_hash
            )
            hash_value.append(last_hash)

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation):
        # KV pools first — determines actual completed page count
        super()._page_transfer(operation)

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        kv_completed_pages = operation.completed_tokens // self.page_size
        if operation.pool_transfers and kv_completed_pages == len(operation.hash_value):
            self._sync_trailing_keys(
                operation.pool_transfers, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)
        operation.pool_transfers_done = True

    def _page_backup(self, operation):
        # Backup extra pools
        if operation.pool_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        # Backup kv pools
        super()._page_backup(operation)

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            if transfer.indices_from_pool != PoolName.KV:
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        """
        for transfer in pool_transfers:
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Auto-alloc host or device indices for PoolTransfers where they are None."""
        if not extra_pools:
            return None
        # (pool, free_fn, indices) for atomic rollback on failure.
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                if alloc_host:
                    prev_pool.host_indices = None
                else:
                    prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if alloc_host:
                if pool.host_indices is not None or pool.device_indices is None:
                    continue
                alloc_fn = entry.host_pool.alloc
                free_fn = entry.host_pool.free
                evict_fn = entry.host_evict_fn
                size = len(pool.device_indices)
            else:
                if pool.device_indices is not None or pool.host_indices is None:
                    continue
                # device_alloc_fn / device_free_fn override entry.device_pool's
                # methods for pools whose device_pool is a raw KV pool (layout)
                # rather than an allocator (e.g. SWA).
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                free_fn = entry.device_free_fn or entry.device_pool.free
                evict_fn = entry.device_evict_fn
                size = len(pool.host_indices)
            indices = alloc_fn(size)
            if indices is None and evict_fn:
                evict_fn(size)
                indices = alloc_fn(size)
            if indices is None:
                # Atomic rollback: free everything we successfully allocated.
                rollback_allocated()
                return None
            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # Assign indices to deferred pools from their source.
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices
        return extra_pools
