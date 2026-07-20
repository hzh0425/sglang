from __future__ import annotations

import hashlib
import json
import logging
import threading
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

from sglang.srt.mem_cache.external_cache_connector import (
    ExternalCacheConnector,
    ExternalCacheHit,
    ExternalStoreCompletion,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.utils import get_hash_str

if TYPE_CHECKING:
    from sglang.srt.mem_cache.external_cache_pool import ExternalPoolStack
    from sglang.srt.mem_cache.radix_cache import RadixKey

logger = logging.getLogger(__name__)


def _stable_namespace(model_config, server_args, page_size: int, tp_size: int) -> str:
    hf_config = getattr(model_config, "hf_config", None)
    config_identity = None
    to_diff_dict = getattr(hf_config, "to_diff_dict", None)
    if callable(to_diff_dict):
        try:
            config_identity = to_diff_dict()
        except Exception:
            config_identity = None
    payload = {
        "model_path": getattr(server_args, "model_path", None),
        "revision": getattr(server_args, "revision", None),
        "model_config": config_identity,
        "kv_cache_dtype": str(getattr(server_args, "kv_cache_dtype", None)),
        "page_size": page_size,
        "tp_size": tp_size,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _extra_key_tag(extra_key: object) -> str:
    encoded = json.dumps(extra_key, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(encoded.encode()).hexdigest()


class MooncakeConnector(ExternalCacheConnector):
    def __init__(
        self,
        *,
        pool_stack: ExternalPoolStack,
        model_config,
        server_args,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        attn_cp_rank: int,
        attn_cp_size: int,
        _storage=None,
    ):
        self.page_size = pool_stack.anchor.page_size
        self.pool_stack = pool_stack
        namespace = _stable_namespace(
            model_config, server_args, self.page_size, tp_size
        )
        backend_tag = (
            f"{namespace}_tp{tp_rank}of{tp_size}"
            f"_cp{attn_cp_rank}of{attn_cp_size}"
        )
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            getattr(server_args, "hicache_storage_backend_extra_config", None)
        )
        extra_config["extra_backend_tag"] = backend_tag
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=getattr(server_args, "model_path", None),
            extra_config=extra_config,
        )

        if _storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(storage_config, pool_stack.anchor)
        else:
            self.storage = _storage
            self.storage.storage_config = storage_config

        self.storage.mem_pool_host = pool_stack.anchor
        self.storage.registered_pools = dict(pool_stack.pools)
        self._registered_allocations: list[tuple[int, int]] = []
        if _storage is None:
            self._register_device_allocations()

        max_workers = int(getattr(server_args, "mooncake_store_workers", 4) or 4)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mooncake-unified-store"
        )
        self._futures: dict[int, Future[bool]] = {}
        self._future_lock = threading.Lock()
        self._next_operation_id = 1
        self._closed = False

    def _register_device_allocations(self) -> None:
        seen: set[tuple[int, int]] = set()
        for pool in self.pool_stack.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                ret = self.storage.store.register_buffer(*allocation)
                if ret != 0:
                    raise RuntimeError(
                        "Failed to register GPU cache allocation with Mooncake, "
                        f"error code: {ret}"
                    )
                self._registered_allocations.append(allocation)

    def _page_keys(self, key: RadixKey) -> list[str]:
        hashes = get_hash_str(key.raw_token_ids(), page_size=self.page_size)
        if not isinstance(hashes, list):
            hashes = [hashes]
        tag = _extra_key_tag(key.extra_key)
        return [f"{tag}@{page_hash}" for page_hash in hashes]

    @staticmethod
    def _extra_transfers(
        transfers: Sequence[PoolTransfer],
    ) -> list[PoolTransfer]:
        return [transfer for transfer in transfers if transfer.name != PoolName.KV]

    @staticmethod
    def _keys_for_transfer(transfer: PoolTransfer, page_keys: list[str]) -> list[str]:
        if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            trailing = max(1, len(transfer.keys) if transfer.keys else 1)
            return page_keys[-trailing:]
        return page_keys

    def _resolve_transfers(
        self,
        page_keys: list[str],
        transfers: Sequence[PoolTransfer],
    ) -> list[PoolTransfer]:
        by_name = {transfer.name: transfer for transfer in transfers}
        resolved = []
        for transfer in self._extra_transfers(transfers):
            source = transfer
            if transfer.indices_from_pool is not None:
                source = by_name[transfer.indices_from_pool]
            resolved.append(
                replace(
                    transfer,
                    host_indices=source.device_indices,
                    keys=self._keys_for_transfer(transfer, page_keys),
                )
            )
        return resolved

    def query(
        self,
        key: RadixKey,
        local_tokens: int,
        transfers: Sequence[PoolTransfer],
    ) -> ExternalCacheHit:
        if local_tokens < 0 or local_tokens % self.page_size != 0:
            raise ValueError(f"local_tokens must be page-aligned, got {local_tokens}")
        page_keys = self._page_keys(key)
        candidates = page_keys[local_tokens // self.page_size :]
        if not candidates:
            return ExternalCacheHit([], 0)

        query_transfers = [
            replace(
                transfer,
                keys=self._keys_for_transfer(transfer, candidates),
            )
            for transfer in self._extra_transfers(transfers)
        ]
        result = self.storage.batch_exists_v2(candidates, query_transfers)
        hit_pages = min(result.kv_hit_pages, len(candidates))
        return ExternalCacheHit(
            page_keys=candidates[:hit_pages],
            hit_tokens=hit_pages * self.page_size,
        )

    @staticmethod
    def _all_transfers_succeeded(
        results: dict, transfers: Sequence[PoolTransfer]
    ) -> bool:
        return bool(results) and all(
            transfer.name in results
            and len(results[transfer.name]) == len(transfer.keys)
            and all(results[transfer.name])
            for transfer in transfers
        )

    def load(
        self,
        hit: ExternalCacheHit,
        transfers: Sequence[PoolTransfer],
    ) -> bool:
        if not hit.page_keys:
            return False
        resolved = self._resolve_transfers(hit.page_keys, transfers)
        results = self.storage.batch_get_v2(resolved)
        return self._all_transfers_succeeded(results, resolved)

    def _store(self, key: RadixKey, transfers: Sequence[PoolTransfer]) -> bool:
        page_keys = self._page_keys(key)
        resolved = self._resolve_transfers(page_keys, transfers)
        results = self.storage.batch_set_v2(resolved)
        return self._all_transfers_succeeded(results, resolved)

    def store_async(
        self,
        key: RadixKey,
        transfers: Sequence[PoolTransfer],
    ) -> int:
        with self._future_lock:
            if self._closed:
                raise RuntimeError("MooncakeConnector is closed")
            operation_id = self._next_operation_id
            self._next_operation_id += 1
            key_snapshot = type(key)(
                array("q", key.raw_token_ids()),
                key.extra_key,
                is_bigram=key.is_bigram,
            )
            transfer_snapshot = tuple(replace(transfer) for transfer in transfers)
            self._futures[operation_id] = self._executor.submit(
                self._store, key_snapshot, transfer_snapshot
            )
        return operation_id

    def poll_completed(self) -> list[ExternalStoreCompletion]:
        with self._future_lock:
            done = [
                (operation_id, future)
                for operation_id, future in self._futures.items()
                if future.done()
            ]
            for operation_id, _ in done:
                del self._futures[operation_id]

        completions = []
        for operation_id, future in done:
            try:
                success = bool(future.result())
            except Exception:
                logger.exception("Mooncake store operation %d failed", operation_id)
                success = False
            completions.append(ExternalStoreCompletion(operation_id, success))
        return completions

    def wait_for_all_stores(self) -> list[ExternalStoreCompletion]:
        with self._future_lock:
            futures = list(self._futures.values())
        for future in futures:
            try:
                future.result()
            except Exception:
                pass
        return self.poll_completed()

    def reset(self) -> None:
        self.wait_for_all_stores()

    def close(self) -> None:
        with self._future_lock:
            if self._closed:
                return
            self._closed = True
        self.wait_for_all_stores()
        self._executor.shutdown(wait=True)
        self.storage.close()
