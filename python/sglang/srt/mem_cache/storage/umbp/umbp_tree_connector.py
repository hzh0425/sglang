from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass, replace
from queue import Empty, Queue
from typing import Any

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_mappings import (
    DevicePoolGroup,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache_connector_mixin import UnifiedTreeConnector

logger = logging.getLogger(__name__)

# Keep every control-plane RPC comfortably below gRPC's message-size limit.
# This is a logical-page count; object batches contain CHUNK_PAGES * layers.
CHUNK_PAGES = 64


def _resolve_umbp_pool_group(
    kvcache: Any, page_size: int, req_to_token_pool: Any
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    if not isinstance(kvcache, DSATokenToKVPool):
        raise TypeError("Direct UMBP connector currently supports DSA KV pools only.")
    if kvcache.page_size != page_size:
        raise ValueError(
            "DSA KV page size must match the tree page size: "
            f"{kvcache.page_size} != {page_size}."
        )
    return resolve_hybrid_device_pool_group(kvcache, page_size, req_to_token_pool)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._producer_index = -1
        self.consumer_index = -1
        self._futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self._producer_index += 1
        self._futures[self._producer_index] = [Future() for _ in range(self.num_layers)]
        return self._producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self._futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self._futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self._futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            raise RuntimeError("UMBP layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self._futures.pop(index, None)

    def reset(self) -> None:
        self._producer_index = -1
        self.consumer_index = -1
        self._futures.clear()


@dataclass
class _LayerObjectPlan:
    name: PoolName
    keys: list[str]
    ptrs: list[int]
    sizes: list[int]
    num_layers: int

    @property
    def num_pages(self) -> int:
        return len(self.keys) // self.num_layers

    def layer_meta(self, layer: int):
        return (
            self.keys[layer :: self.num_layers],
            self.ptrs[layer :: self.num_layers],
            self.sizes[layer :: self.num_layers],
        )


def _sort_by_device_address(
    keys: list[str], ptrs: list[int], sizes: list[int]
) -> tuple[list[str], list[int], list[int]]:
    """Reorder an offload batch so objects go out in GPU-address order.

    The storage tier allocates slots in the order objects arrive, so the send
    order decides the storage-side layout. Objects are built page-major
    (`_object_keys` / `get_page_buffer_meta`), which scatters one layer's pages
    across the tier with a stride of `num_layers`; the layer-wise load then
    reads exactly that strided set and cannot coalesce anything, even though
    its GPU side is ~98% contiguous.

    Sorting by GPU address groups objects by layer buffer and orders pages
    within each layer, so the tier's slots end up mirroring the GPU layout and
    both sides of a later load become contiguous.

    Sorting by address rather than building an explicit layer-major permutation
    keeps this adaptive to the real allocation layout. The key strings are
    unchanged and travel with their pointers -- `lookup` builds its own
    page-major query list, which must stay page-major because
    `batch_exists_consecutive` counts an object prefix and the caller divides
    that by `num_layers` to get whole pages.
    """
    if len(keys) < 2:
        return keys, ptrs, sizes
    order = sorted(range(len(ptrs)), key=ptrs.__getitem__)
    return (
        [keys[i] for i in order],
        [ptrs[i] for i in order],
        [sizes[i] for i in order],
    )


def _config_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"UMBP connector config {key!r} must be boolean, got {value!r}.")


def _parse_storage_extra_config(raw_config):
    # Keep the connector module importable in CPU-only unit tests. The hybrid
    # controller imports device-specific memory-pool modules transitively.
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )

    extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
        raw_config
    )
    return extra_config


class UMBPTreeConnector(UnifiedTreeConnector):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        _storage=None,
    ):
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        pool_group = _resolve_umbp_pool_group(
            kvcache, self.page_size, params.req_to_token_pool
        )
        self.pools = pool_group.entry_map
        self.num_layers = pool_group.num_layers
        if self.num_layers == 0 or any(
            len(pool.kv_buffer) != self.num_layers for pool in self.pools.values()
        ):
            raise ValueError("UMBP KV and INDEXER pools must have equal layer counts.")

        tp_rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=params.tp_cache_group)

        extra_config = _parse_storage_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        extra_config = dict(extra_config)
        standalone_requested = bool(
            extra_config.get("standalone_address")
            or os.getenv("UMBP_STANDALONE_ADDRESS")
        )
        if "ssd_enabled" in extra_config and _config_bool(
            extra_config["ssd_enabled"], "ssd_enabled"
        ):
            raise ValueError(
                "Direct UMBP requires ssd_enabled=false because its GPU path "
                "cannot use the corresponding host-memory fallback."
            )
        extra_config["ssd_enabled"] = False

        if "cache_remote_fetches" in extra_config and _config_bool(
            extra_config["cache_remote_fetches"], "cache_remote_fetches"
        ):
            raise ValueError(
                "Direct UMBP requires cache_remote_fetches=false because its GPU "
                "path cannot use the corresponding host-memory fallback."
            )
        if standalone_requested:
            extra_config.pop("cache_remote_fetches", None)
        else:
            extra_config["cache_remote_fetches"] = False

        probe = torch.arange(self.page_size, dtype=torch.int64)
        min_object_size = min(
            min(pool.get_page_buffer_meta(probe)[1]) for pool in self.pools.values()
        )
        if standalone_requested:
            extra_config.pop("dram_page_size", None)
        else:
            dram_page_size = int(extra_config.get("dram_page_size", min_object_size))
            if not 0 < dram_page_size <= min_object_size:
                raise ValueError(
                    "Direct UMBP requires 0 < dram_page_size <= the smallest "
                    f"per-layer object ({min_object_size} bytes), got {dram_page_size}."
                )
            extra_config["dram_page_size"] = dram_page_size

        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=server_args.model_path,
            extra_config=extra_config,
        )

        if _storage is None:
            from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

            self.storage = UMBPStore(storage_config, mem_pool_host=None)
        else:
            self.storage = _storage

        try:
            client = self.storage.client
            mode = client.get_deployment_mode()
            mode_type = type(mode)
            if mode not in {
                mode_type.Distributed,
                mode_type.StandaloneProcess,
            }:
                raise ValueError(
                    "Direct UMBP supports only Distributed or StandaloneProcess "
                    f"deployment modes, got {mode!r}."
                )
            self.deployment_mode = mode
            self._standalone_process_mode = mode == mode_type.StandaloneProcess
            if getattr(self.storage, "_disable_zero_copy_register", False):
                raise ValueError(
                    "Direct UMBP cannot disable zero-copy memory registration."
                )

            self.storage.mem_pool_host = pool_group
            self.storage._kv_anchor_is_logical = True
            self.storage.registered_pools = self.pools
            self.storage.mla_suffix = (
                f"tp{tp_rank}_cp{params.attn_cp_rank}_pp{params.pp_rank}"
            )
            self._register_buffers()
        except BaseException:
            self.storage.close()
            raise

        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        self._pending: dict[str, list[PoolTransfer]] = {}
        self._load_queue: Queue[tuple[int, list[_LayerObjectPlan]] | None] = Queue()
        self._offload_queue: Queue[list[PoolTransfer] | None] = Queue()
        self._offload_results: Queue[bool] = Queue()
        self._stats = {"lookup": 0, "load": 0, "offload": 0}
        self._load_thread = threading.Thread(
            target=self._load_thread_func,
            daemon=True,
            name=f"umbp-layerwise-tp{tp_rank}",
        )
        self._offload_thread = threading.Thread(
            target=self._offload_thread_func,
            daemon=True,
            name=f"umbp-offload-tp{tp_rank}",
        )
        self._closed = False
        self._load_thread.start()
        self._offload_thread.start()

    def _register_buffers(self) -> None:
        seen = set()
        self._registered: list[tuple[int, int]] = []
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                if not self.storage.client.register_memory(*allocation):
                    raise RuntimeError(
                        "Failed to register a GPU KV buffer with UMBP: "
                        f"ptr=0x{allocation[0]:x}, size={allocation[1]}."
                    )
                self._registered.append(allocation)

    def _expand(self, transfers: list[PoolTransfer]) -> list[PoolTransfer]:
        kv = next(
            (transfer for transfer in transfers if transfer.name == PoolName.KV),
            None,
        )
        if kv is None or not kv.keys:
            return []
        return [
            replace(
                kv,
                name=name,
                host_indices=kv.device_indices,
                keys=list(kv.keys),
                indices_from_pool=None,
            )
            for name in self.pools
        ]

    def _object_keys(self, transfer: PoolTransfer) -> list[str]:
        page_keys, multiplier = self.storage._get_hybrid_page_component_keys(
            list(transfer.keys or []), transfer
        )
        if multiplier != 1:
            raise ValueError(
                f"Direct UMBP requires one page key per pool, got multiplier="
                f"{multiplier} for {transfer.name}."
            )
        return [
            f"{page_key}_L{layer}"
            for page_key in page_keys
            for layer in range(self.num_layers)
        ]

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self._expand(transfers)
        if not expanded:
            return []

        hit_pages = None
        chunk_objects = CHUNK_PAGES * self.num_layers
        for transfer in expanded:
            object_keys = self._object_keys(transfer)
            pages = 0
            for start in range(0, len(object_keys), chunk_objects):
                block = object_keys[start : start + chunk_objects]
                consecutive = int(self.storage.client.batch_exists_consecutive(block))
                if not 0 <= consecutive <= len(block):
                    raise RuntimeError(
                        "UMBP returned an invalid consecutive-exists count: "
                        f"{consecutive} for {len(block)} keys."
                    )
                pages += consecutive // self.num_layers
                if consecutive < len(block):
                    break
            hit_pages = pages if hit_pages is None else min(hit_pages, pages)

        self._stats["lookup"] += 1
        if not hit_pages:
            return []
        logger.debug("Unified tree UMBP lookup hit: rid=%s pages=%d", rid, hit_pages)
        return list(range(1, hit_pages + 1))

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        expanded = self._expand(transfers)
        if not expanded:
            return False
        if rid in self._pending:
            raise RuntimeError(f"UMBP load for rid={rid} is already queued.")
        self._pending[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> None:
        self._pending.pop(rid, None)

    def start_layer_wise_loading(self) -> int:
        if not self._pending:
            return -1
        pending = self._pending
        self._pending = {}
        plans = self._build_load_plans(list(pending.values()))
        counter_index = self.layer_done_counter.update_producer()
        self._load_queue.put((counter_index, plans))
        self._stats["load"] += len(pending)
        return counter_index

    def _build_load_plans(
        self, request_transfers: list[list[PoolTransfer]]
    ) -> list[_LayerObjectPlan]:
        grouped: dict[PoolName, list[PoolTransfer]] = {}
        for transfers in request_transfers:
            for transfer in transfers:
                grouped.setdefault(transfer.name, []).append(transfer)

        plans = []
        for name, transfers in grouped.items():
            keys: list[str] = []
            ptrs: list[int] = []
            sizes: list[int] = []
            for transfer in transfers:
                transfer_keys = self._object_keys(transfer)
                transfer_ptrs, transfer_sizes = self.pools[name].get_page_buffer_meta(
                    transfer.host_indices
                )
                if not len(transfer_keys) == len(transfer_ptrs) == len(transfer_sizes):
                    raise ValueError(
                        f"Layer-wise UMBP metadata mismatch for pool {name}: "
                        f"keys={len(transfer_keys)} ptrs={len(transfer_ptrs)} "
                        f"sizes={len(transfer_sizes)}."
                    )
                keys.extend(transfer_keys)
                ptrs.extend(transfer_ptrs)
                sizes.extend(transfer_sizes)
            plans.append(_LayerObjectPlan(name, keys, ptrs, sizes, self.num_layers))

        if not plans or not plans[0].keys:
            raise ValueError("Layer-wise UMBP load has no object keys.")
        num_pages = plans[0].num_pages
        if any(plan.num_pages != num_pages for plan in plans):
            raise ValueError("Layer-wise UMBP pools must cover the same page set.")
        return plans

    def _load_thread_func(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                counter_index, plans = task
                self._run_layer_wise_batch(counter_index, plans)
            finally:
                self._load_queue.task_done()

    def _run_layer_wise_batch(
        self, counter_index: int, plans: list[_LayerObjectPlan]
    ) -> None:
        try:
            max_objects_per_call = CHUNK_PAGES * self.num_layers
            for layer in range(self.num_layers):
                for plan in plans:
                    keys, ptrs, sizes = plan.layer_meta(layer)
                    for start in range(0, len(keys), max_objects_per_call):
                        chunk_keys = keys[start : start + max_objects_per_call]
                        results = list(
                            self.storage.client.batch_get_into_ptr(
                                chunk_keys,
                                ptrs[start : start + max_objects_per_call],
                                sizes[start : start + max_objects_per_call],
                            )
                        )
                        if len(results) != len(chunk_keys) or not all(results):
                            raise RuntimeError(
                                f"UMBP get failed for pool={plan.name}, layer={layer}: "
                                f"success={sum(bool(value) for value in results)}/"
                                f"{len(chunk_keys)}."
                            )
                self.layer_done_counter.complete(counter_index, layer)
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("UMBP layer-wise load batch failed")

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self._expand(transfers)
        if not expanded:
            return False
        self._offload_queue.put(expanded)
        return True

    def _offload_thread_func(self) -> None:
        while True:
            task = self._offload_queue.get()
            try:
                if task is None:
                    return
                try:
                    success = self._run_offload(task)
                except BaseException:
                    logger.exception("UMBP offload failed")
                    success = False
                self._offload_results.put(success)
            finally:
                self._offload_queue.task_done()

    def _run_offload(self, expanded: list[PoolTransfer]) -> bool:
        self._wait_for_device()
        chunk_objects = CHUNK_PAGES * self.num_layers
        for transfer in expanded:
            keys = self._object_keys(transfer)
            ptrs, sizes = self.pools[transfer.name].get_page_buffer_meta(
                transfer.host_indices
            )
            if not len(keys) == len(ptrs) == len(sizes):
                raise ValueError(
                    f"UMBP offload metadata mismatch for pool {transfer.name}."
                )
            keys, ptrs, sizes = _sort_by_device_address(keys, ptrs, sizes)
            for start in range(0, len(keys), chunk_objects):
                chunk_keys = keys[start : start + chunk_objects]
                results = list(
                    self.storage.client.batch_put_from_ptr(
                        chunk_keys,
                        ptrs[start : start + chunk_objects],
                        sizes[start : start + chunk_objects],
                    )
                )
                if len(results) != len(chunk_keys) or not all(results):
                    logger.warning(
                        "UMBP offload failed: pool=%s object_range=[%d,%d) "
                        "success=%d/%d returned=%d",
                        transfer.name,
                        start,
                        min(start + chunk_objects, len(keys)),
                        sum(bool(value) for value in results),
                        len(chunk_keys),
                        len(results),
                    )
                    return False

        self._stats["offload"] += 1
        return True

    def num_completed_offloads(self) -> int:
        return self._offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self._offload_results.get_nowait()

    def _wait_for_device(self) -> None:
        device = next(
            (
                buffer.device
                for pool in self.pools.values()
                for buffer in pool.get_hybrid_pool_buffer()
                if buffer.device.type == "cuda"
            ),
            None,
        )
        if device is not None:
            torch.cuda.synchronize(device)

    def reset(self) -> None:
        self._pending.clear()
        self._load_queue.join()
        self._offload_queue.join()
        while True:
            try:
                self._offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        for thread, queue in (
            (self._offload_thread, self._offload_queue),
            (self._load_thread, self._load_queue),
        ):
            if thread.is_alive():
                queue.put(None)
                thread.join()
        if self._standalone_process_mode and self._registered:
            # StandaloneProcess deregistration is client-wide; one call tears
            # down every registered region. Keep the GPU tensors alive until
            # the synchronous RPC has completed successfully.
            self.storage.client.deregister_memory(self._registered[0][0])
        logger.info("Unified tree UMBP stats: %s", self._stats)
        self.storage.close()
        self._closed = True
