from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
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


def _ordered_layers(entry) -> list[int]:
    component_lengths = {len(component) for component in entry.components}
    if len(component_lengths) != 1:
        raise ValueError(
            f"UMBP pool {entry.name} components have different layer counts."
        )
    pool_layer_count = component_lengths.pop()
    if pool_layer_count != len(entry.layer_mapping):
        raise ValueError(
            f"UMBP pool {entry.name} has {pool_layer_count} buffers per component "
            f"but {len(entry.layer_mapping)} mapped layers."
        )
    by_buffer = {
        buffer_index: logical_layer
        for logical_layer, buffer_index in entry.layer_mapping.items()
    }
    if sorted(by_buffer) != list(range(pool_layer_count)):
        raise ValueError(
            f"UMBP pool {entry.name} layer mapping is not a contiguous bijection."
        )
    return [by_buffer[index] for index in range(pool_layer_count)]


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
    logical_pages: int
    component_count: int
    pool_layer_count: int

    def layer_meta(self, layer: int):
        return (
            self.keys[layer :: self.pool_layer_count],
            self.ptrs[layer :: self.pool_layer_count],
            self.sizes[layer :: self.pool_layer_count],
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
        self.pool_group = _resolve_umbp_pool_group(
            kvcache, self.page_size, params.req_to_token_pool
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        if self.num_layers <= 0:
            raise ValueError("UMBP requires at least one logical layer.")
        self.pool_layers = {
            name: _ordered_layers(entry) for name, entry in self.pools.items()
        }
        invalid_layers = {
            name: [layer for layer in layers if not 0 <= layer < self.num_layers]
            for name, layers in self.pool_layers.items()
        }
        invalid_layers = {
            name: layers for name, layers in invalid_layers.items() if layers
        }
        if invalid_layers:
            raise ValueError(
                f"UMBP pool mappings contain out-of-range logical layers: {invalid_layers}."
            )

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

        min_object_size = min(
            min(
                pool.get_page_buffer_meta(
                    torch.arange(pool.page_size, dtype=torch.int64)
                )[1]
            )
            for pool in self.pools.values()
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

            self.storage.mem_pool_host = self.pool_group
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

    def _object_keys_for_pages(
        self, page_keys: list[str], transfer: PoolTransfer
    ) -> tuple[list[str], int]:
        component_keys, multiplier = self.storage._get_hybrid_page_component_keys(
            page_keys, transfer
        )
        component_count = len(self.pools[transfer.name].components)
        if multiplier != component_count:
            raise ValueError(
                f"UMBP pool {transfer.name} produced {multiplier} key components "
                f"for {component_count} buffer components."
            )
        layers = self.pool_layers[transfer.name]
        return (
            [
                f"{component_key}_L{logical_layer}"
                for component_key in component_keys
                for logical_layer in layers
            ],
            multiplier,
        )

    def _object_keys(self, transfer: PoolTransfer) -> list[str]:
        keys, _ = self._object_keys_for_pages(list(transfer.keys or []), transfer)
        return keys

    def _page_exists(self, page_keys: list[str], transfer: PoolTransfer) -> list[bool]:
        component_count = len(self.pools[transfer.name].components)
        pool_layer_count = len(self.pool_layers[transfer.name])
        objects_per_page = component_count * pool_layer_count
        max_objects = CHUNK_PAGES * self.num_layers
        pages_per_call = max(1, max_objects // objects_per_page)

        page_exists = []
        for start in range(0, len(page_keys), pages_per_call):
            chunk_pages = page_keys[start : start + pages_per_call]
            object_keys, _ = self._object_keys_for_pages(chunk_pages, transfer)
            exists = list(self.storage.client.batch_exists(object_keys))
            if len(exists) != len(object_keys):
                raise RuntimeError(
                    f"UMBP exists result-size mismatch for pool {transfer.name}: "
                    f"expected={len(object_keys)} actual={len(exists)}."
                )
            page_exists.extend(
                all(exists[index : index + objects_per_page])
                for index in range(0, len(exists), objects_per_page)
            )
        return page_exists

    @staticmethod
    def _apply_hit_policy(
        valid_pages: list[int], page_exists: list[bool], transfer: PoolTransfer
    ) -> list[int]:
        present_prefix = [0]
        for present in page_exists:
            present_prefix.append(present_prefix[-1] + int(present))

        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            return [end for end in valid_pages if present_prefix[end] == end]
        if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            trailing = max(1, len(transfer.keys or ()))
            return [
                end
                for end in valid_pages
                if present_prefix[end] - present_prefix[max(0, end - trailing)]
                == end - max(0, end - trailing)
            ]
        raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys or [])
        if not page_keys:
            return []

        valid_pages = list(range(1, len(page_keys) + 1))
        for transfer in expanded:
            page_exists = self._page_exists(page_keys, transfer)
            valid_pages = self._apply_hit_policy(valid_pages, page_exists, transfer)
            if not valid_pages:
                break

        self._stats["lookup"] += 1
        if valid_pages:
            logger.debug(
                "Unified tree UMBP lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                valid_pages[-1],
                len(valid_pages),
            )
        return valid_pages

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers)
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
            logical_pages = 0
            component_count = None
            for transfer in transfers:
                page_keys = list(transfer.keys or [])
                transfer_keys, multiplier = self._object_keys_for_pages(
                    page_keys, transfer
                )
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
                logical_pages += len(page_keys)
                if component_count is None:
                    component_count = multiplier
                elif component_count != multiplier:
                    raise ValueError(
                        f"UMBP pool {name} changed component count within one load."
                    )
            pool_layer_count = len(self.pool_layers[name])
            component_count = component_count or 0
            expected = logical_pages * component_count * pool_layer_count
            if not len(keys) == len(ptrs) == len(sizes) == expected:
                raise ValueError(
                    f"Layer-wise UMBP plan mismatch for pool {name}: "
                    f"keys={len(keys)} ptrs={len(ptrs)} sizes={len(sizes)} "
                    f"expected={expected}."
                )
            plans.append(
                _LayerObjectPlan(
                    name,
                    keys,
                    ptrs,
                    sizes,
                    logical_pages,
                    component_count,
                    pool_layer_count,
                )
            )

        if not plans or not plans[0].keys:
            raise ValueError("Layer-wise UMBP load has no object keys.")
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
            by_layer: dict[int, list[tuple[_LayerObjectPlan, int]]] = defaultdict(list)
            for plan in plans:
                for local_layer, logical_layer in enumerate(
                    self.pool_layers[plan.name]
                ):
                    by_layer[logical_layer].append((plan, local_layer))

            max_objects_per_call = CHUNK_PAGES * self.num_layers
            for logical_layer in range(self.num_layers):
                for plan, local_layer in by_layer.get(logical_layer, ()):
                    keys, ptrs, sizes = plan.layer_meta(local_layer)
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
                self.layer_done_counter.complete(counter_index, logical_layer)
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("UMBP layer-wise load batch failed")

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
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
        for transfer in expanded:
            keys, component_count = self._object_keys_for_pages(
                list(transfer.keys or []), transfer
            )
            ptrs, sizes = self.pools[transfer.name].get_page_buffer_meta(
                transfer.host_indices
            )
            if not len(keys) == len(ptrs) == len(sizes):
                raise ValueError(
                    f"UMBP offload metadata mismatch for pool {transfer.name}."
                )
            keys, ptrs, sizes = _sort_by_device_address(keys, ptrs, sizes)
            chunk_objects = (
                CHUNK_PAGES * len(self.pool_layers[transfer.name]) * component_count
            )
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
