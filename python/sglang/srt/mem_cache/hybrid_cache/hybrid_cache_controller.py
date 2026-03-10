from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

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
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    StorageTransfer,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()


@dataclass
class AuxiliaryPageBinding:
    policy: str = "all_pages"
    trailing_pages: int = 0
    page_indices: Optional[list[int]] = None


@dataclass
class AuxiliaryLoadResult:
    loaded_names: set[str]
    hit_page_count_by_name: dict[str, int]

    @classmethod
    def empty(cls) -> "AuxiliaryLoadResult":
        return cls(set(), {})

    def merge(self, other: Optional["AuxiliaryLoadResult"]) -> None:
        if other is None:
            return
        self.loaded_names.update(other.loaded_names)
        for name, hit_count in other.hit_page_count_by_name.items():
            self.hit_page_count_by_name[name] = max(
                self.hit_page_count_by_name.get(name, 0),
                hit_count,
            )


@dataclass
class AuxiliaryTransfer:
    name: str
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    binding: Optional[AuxiliaryPageBinding] = None


class CacheOperation(BaseCacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.auxiliary_transfers = auxiliary_transfers

    @staticmethod
    def merge_auxiliary_transfers(
        ops: List["CacheOperation"],
    ) -> Optional[list[AuxiliaryTransfer]]:
        grouped: dict[str, dict[str, Any]] = {}
        for op in ops:
            for transfer in op.auxiliary_transfers or []:
                merged = grouped.setdefault(
                    transfer.name,
                    {
                        "host_indices": [],
                        "device_indices": [],
                        "page_indices": [],
                        "has_page_indices": False,
                        "policy": (
                            "all_pages"
                            if transfer.binding is None
                            else transfer.binding.policy
                        ),
                        "trailing_pages": (
                            0
                            if transfer.binding is None
                            else transfer.binding.trailing_pages
                        ),
                    },
                )
                current_policy = (
                    "all_pages" if transfer.binding is None else transfer.binding.policy
                )
                if merged["policy"] != current_policy:
                    raise ValueError(
                        f"Conflicting binding policies for {transfer.name}: "
                        f"{merged['policy']} vs {current_policy}"
                    )
                current_trailing_pages = (
                    0 if transfer.binding is None else transfer.binding.trailing_pages
                )
                if merged["trailing_pages"] != current_trailing_pages:
                    raise ValueError(
                        f"Conflicting trailing_pages for {transfer.name}: "
                        f"{merged['trailing_pages']} vs {current_trailing_pages}"
                    )
                if transfer.host_indices is not None:
                    merged["host_indices"].append(transfer.host_indices)
                if transfer.device_indices is not None:
                    merged["device_indices"].append(transfer.device_indices)
                page_indices = (
                    None if transfer.binding is None else transfer.binding.page_indices
                )
                if page_indices is not None:
                    merged["page_indices"].extend(int(x) for x in page_indices)
                    merged["has_page_indices"] = True
        if not grouped:
            return None

        transfers = []
        for name, merged in grouped.items():
            transfers.append(
                AuxiliaryTransfer(
                    name=name,
                    host_indices=(
                        torch.cat(merged["host_indices"])
                        if merged["host_indices"]
                        else None
                    ),
                    device_indices=(
                        torch.cat(merged["device_indices"])
                        if merged["device_indices"]
                        else None
                    ),
                    binding=AuxiliaryPageBinding(
                        policy=merged["policy"],
                        trailing_pages=merged["trailing_pages"],
                        page_indices=(
                            merged["page_indices"]
                            if merged["has_page_indices"]
                            else None
                        ),
                    ),
                )
            )
        return transfers

    @staticmethod
    def merge_ops(ops: List["CacheOperation"]) -> "CacheOperation":
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
            auxiliary_transfers=CacheOperation.merge_auxiliary_transfers(ops),
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
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.auxiliary_transfers = auxiliary_transfers
        self.auxiliary_load_result = AuxiliaryLoadResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
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
            auxiliary_transfers=auxiliary_transfers,
        )

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


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        transfer_layer_num: Optional[int] = None,
    ):
        self.transfer_layer_num = transfer_layer_num
        startup_storage_backend = storage_backend
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            pp_rank=pp_rank,
            pp_size=pp_size,
        )
        self.transfer_layer_num = self.transfer_layer_num or self.layer_num
        if self.transfer_layer_num != self.layer_done_counter.num_layers:
            self.layer_done_counter = LayerDoneCounter(self.transfer_layer_num)
            self.mem_pool_device.register_layer_transfer_counter(
                self.layer_done_counter
            )
        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

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
        self.page_get_func = self._page_get_v2
        self.page_set_func = self._page_set_v2
        logger.info("Using pool-based interface for hybrid storage operations")

    def reset(self):
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                auxiliary_transfers=auxiliary_transfers,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
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
                auxiliary_transfers=op.auxiliary_transfers,
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        auxiliary_transfers: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        if callable(auxiliary_transfers):
            auxiliary_transfers = auxiliary_transfers(device_indices)
        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                auxiliary_transfers=auxiliary_transfers,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(op)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.transfer_layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                    auxiliary_transfers=op.auxiliary_transfers,
                )
                producer_event.complete(i)
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            auxiliary_transfers=auxiliary_transfers,
        )
        self.prefetch_queue.put(operation)
        return operation

    def _build_storage_transfers(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
    ) -> list[StorageTransfer]:
        transfers = [StorageTransfer("kv", keys, host_indices)]
        if not auxiliary_transfers:
            return transfers
        for transfer in auxiliary_transfers:
            if transfer.host_indices is None:
                continue
            page_indices = (
                None if transfer.binding is None else transfer.binding.page_indices
            )
            if page_indices is None:
                aux_keys = keys
            else:
                aux_keys = [keys[i] for i in page_indices if 0 <= i < len(keys)]
            if not aux_keys:
                continue
            transfers.append(
                StorageTransfer(transfer.name, aux_keys, transfer.host_indices)
            )
        return transfers

    def _page_get_v2(
        self,
        operation,
        hash_values,
        host_indices,
        extra_info=None,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
    ):
        results = self.storage_backend.batch_get_v2(
            self._build_storage_transfers(
                hash_values,
                host_indices,
                auxiliary_transfers=auxiliary_transfers,
            ),
            extra_info,
        )
        if extra_info and extra_info.extra_info:
            operation.auxiliary_load_result.merge(
                AuxiliaryLoadResult(
                    loaded_names=set(extra_info.extra_info.get("loaded_names", [])),
                    hit_page_count_by_name=dict(
                        extra_info.extra_info.get("hit_page_count_by_name", {})
                    ),
                )
            )
        inc = 0
        kv_results = results.get("kv", [])
        for ok, key in zip(kv_results, hash_values):
            if not ok:
                logger.warning(
                    "Prefetch operation %s failed to retrieve page %s.",
                    operation.request_id,
                    key,
                )
                break
            inc += self.page_size
        operation.increment(inc)

    def _page_transfer(self, operation):
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), self.storage_batch_size):
            batch_hashes = operation.hash_value[i : i + self.storage_batch_size]
            batch_page_count = len(batch_hashes)
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + batch_page_count) * self.page_size
            ]
            prev_completed_tokens = operation.completed_tokens
            self.page_get_func(
                operation,
                batch_hashes,
                batch_host_indices,
                HiCacheStorageExtraInfo(prefix_keys=prefix_keys),
                self._slice_auxiliary_transfers_for_batch(
                    operation.auxiliary_transfers, i, batch_page_count
                ),
            )
            if (
                operation.completed_tokens
                != prev_completed_tokens + batch_page_count * self.page_size
            ):
                operation.mark_terminate()
                break
            if prefix_keys:
                prefix_keys += batch_hashes

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
        constraints = self._build_auxiliary_hit_constraints(
            operation.auxiliary_transfers
        )
        hit_page_num = (
            self.storage_backend.batch_exists_v2(hash_value, constraints, extra_info)
            if constraints
            else self.storage_backend.batch_exists(hash_value, extra_info)
        )
        return hash_value[:hit_page_num], hit_page_num * self.page_size

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            auxiliary_transfers=auxiliary_transfers,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _page_set_v2(
        self,
        hash_values,
        host_indices,
        extra_info=None,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]] = None,
    ) -> bool:
        results = self.storage_backend.batch_set_v2(
            self._build_storage_transfers(
                hash_values,
                host_indices,
                auxiliary_transfers=auxiliary_transfers,
            ),
            extra_info,
        )
        return all(all(pool_results) for pool_results in results.values())

    def _page_backup(self, operation):
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), self.storage_batch_size):
            batch_hashes = operation.hash_value[i : i + self.storage_batch_size]
            batch_page_count = len(batch_hashes)
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + batch_page_count) * self.page_size
            ]
            success = self.page_set_func(
                batch_hashes,
                batch_host_indices,
                HiCacheStorageExtraInfo(prefix_keys=prefix_keys),
                self._slice_auxiliary_transfers_for_batch(
                    operation.auxiliary_transfers, i, batch_page_count
                ),
            )
            if not success:
                logger.warning(
                    "Write page to storage: %s pages failed.", len(batch_hashes)
                )
                break
            if prefix_keys:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * batch_page_count

    def _slice_auxiliary_transfers_for_batch(
        self,
        auxiliary_transfers: Optional[list[AuxiliaryTransfer]],
        batch_page_start: int,
        batch_page_count: int,
    ) -> Optional[list[AuxiliaryTransfer]]:
        if not auxiliary_transfers:
            return None
        batch_page_end = batch_page_start + batch_page_count
        batch_transfers = []
        for transfer in auxiliary_transfers:
            page_indices = (
                None if transfer.binding is None else transfer.binding.page_indices
            )
            if page_indices is None:
                batch_transfers.append(transfer)
                continue
            local_page_indices = [
                int(page_index) - batch_page_start
                for page_index in page_indices
                if batch_page_start <= int(page_index) < batch_page_end
            ]
            if local_page_indices:
                batch_transfers.append(
                    AuxiliaryTransfer(
                        name=transfer.name,
                        host_indices=transfer.host_indices,
                        device_indices=transfer.device_indices,
                        binding=AuxiliaryPageBinding(
                            policy=transfer.binding.policy,
                            trailing_pages=transfer.binding.trailing_pages,
                            page_indices=local_page_indices,
                        ),
                    )
                )
        return batch_transfers or None

    def _build_auxiliary_hit_constraints(
        self, auxiliary_transfers: Optional[list[AuxiliaryTransfer]]
    ) -> Optional[list[dict[str, Any]]]:
        if not auxiliary_transfers:
            return None
        constraints = []
        for transfer in auxiliary_transfers:
            policy = (
                "all_pages" if transfer.binding is None else transfer.binding.policy
            )
            constraint = {"name": transfer.name, "policy": policy}
            if policy == "trailing_pages":
                constraint["trailing_pages"] = max(
                    1, int(transfer.binding.trailing_pages)
                )
            elif policy != "all_pages":
                raise ValueError(f"Unsupported auxiliary hit policy: {policy}")
            constraints.append(constraint)
        return constraints or None
