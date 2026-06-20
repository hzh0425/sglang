# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RustUnifiedRadixCache: Python orchestrator over the Rust radix cache.

Coordinates three pieces:
    1. Rust RadixCache (sglang.srt.mem_cache._mem_cache_core.RustPageRadixCacheWrapper) — owns
       tree state, root handle, and lock_ref accounting. Handles all page sizes
       (`page_size >= 1`); `page_size=1` uses one-element page keys.
    2. Python ReqToTokenPool — owns per-request kv-index storage; unchanged.
    3. Python {Token,Paged}TokenToKVPoolAllocator — owns slot allocation and
       per-token KV cache references; unchanged.

Drop-in replacement for `sglang.srt.mem_cache.radix_cache.RadixCache` for the
v1 supported configuration:
    * Full attention only through the registered backend (no SWA, Mamba, or HiCache).
    * page_size >= 1.
    * LRU eviction only.
    * No EAGLE bigram, no `enable_kv_cache_events`, no TTL eviction.
    * No insert priority (LRU ignores it).

Unsupported features raise `RadixCacheInfraPyError`. Construction-time
rejections fail fast at process start; per-call rejections fail the call
without corrupting cache state.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

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
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

ComponentType: Any = None
RadixCacheInfraPyError: Any = None
RadixCacheRuntimePyError: Any = None
RustBigramRadixCacheWrapper: Any = None
RustPageRadixCacheWrapper: Any = None
_NATIVE_SYMBOLS_LOADED = False


def _load_native_symbols() -> None:
    """Load the PyO3 extension only when the Rust backend is selected."""
    global ComponentType
    global RadixCacheInfraPyError
    global RadixCacheRuntimePyError
    global RustBigramRadixCacheWrapper
    global RustPageRadixCacheWrapper
    global _NATIVE_SYMBOLS_LOADED

    if _NATIVE_SYMBOLS_LOADED:
        return

    try:
        from sglang.srt.mem_cache._mem_cache_core import (
            ComponentType as NativeComponentType,
            RadixCacheInfraPyError as NativeRadixCacheInfraPyError,
            RadixCacheRuntimePyError as NativeRadixCacheRuntimePyError,
            RustBigramRadixCacheWrapper as NativeRustBigramRadixCacheWrapper,
            RustPageRadixCacheWrapper as NativeRustPageRadixCacheWrapper,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "sglang.srt.mem_cache._mem_cache_core":
            raise ModuleNotFoundError(
                "RustUnifiedRadixCache requires native extension "
                "sglang.srt.mem_cache._mem_cache_core. Install SGLang with "
                "`python -m pip install -e python` or build the package before "
                "using `--radix-cache-backend rust_unified`."
            ) from exc
        raise

    ComponentType = NativeComponentType
    RadixCacheInfraPyError = NativeRadixCacheInfraPyError
    RadixCacheRuntimePyError = NativeRadixCacheRuntimePyError
    RustBigramRadixCacheWrapper = NativeRustBigramRadixCacheWrapper
    RustPageRadixCacheWrapper = NativeRustPageRadixCacheWrapper
    _NATIVE_SYMBOLS_LOADED = True


# Initial capacity hint for the Rust tree node pool. The pool grows on demand;
# this just avoids early reallocations during warmup.
_DEFAULT_INIT_NODE_CAPACITY = 1024


class RustUnifiedRadixCache(BasePrefixCache):
    """Python orchestrator routing tree ops to the Rust radix cache while
    keeping Python ownership of `req_to_token_pool` and the allocator.

    The `req.last_node` field carries an opaque integer (the Rust NodeIdx)
    instead of a Python `TreeNode`. External code that previously read
    `req.last_node.X` attributes will break — only HiCache / Mamba / LMCache
    paths do this today, all out of v1 scope.
    """

    def __init__(self, params: CacheInitParams):
        _load_native_symbols()

        # Required fields per `PrefixCacheTrait`. External code reads these
        # directly (e.g. observability `available_and_evictable_str`).
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable_finished_insert = params.disable_finished_insert
        self.sliding_window_size = params.sliding_window_size
        self.is_eagle = params.is_eagle
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        server_args = get_global_server_args()
        # Enable Mamba if the scheduler passed in a HybridReqToTokenPool.
        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            self.mamba_cache_chunk_size: Optional[int] = (
                server_args.mamba_cache_chunk_size
            )
        else:
            self.mamba_cache_chunk_size = None
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        # Confirm the radix cache related setups are supported.
        self._reject_unsupported(params)
        if self.enable_hierarchical_cache:
            self._reject_unsupported_hicache(server_args)

        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        if self.token_to_kv_pool_allocator is not None:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")
        self._debug_trace = os.environ.get("SGLANG_RUST_URT_DEBUG", "") == "1"
        device_str = self._device_to_rust_str(self.device)
        self.cache_controller: Any = None
        self.sidecar_pool_specs: list[Any] = []
        # Pick the concrete wrapper class at construction. The two share an
        # identical Python surface (constructor signature + every method
        # signature), so no per-method dispatch is needed downstream — the
        # bigram wrapper internally builds `(t[i], t[i+1])` overlap pairs
        # from the raw 1-D `int64` keys it receives. Pre-call trimming in
        # Python (page-align in raw-token space) would silently corrupt
        # the bigram count for the EAGLE path; we drop all such trims
        # below and let Rust own page-alignment in atom units (= bigram
        # pairs when `is_eagle`).
        wrapper_cls = (
            RustBigramRadixCacheWrapper if self.is_eagle else RustPageRadixCacheWrapper
        )
        self._rust_radix: Any = wrapper_cls(
            device=device_str,
            page_size=self.page_size,
            init_node_capacity=_DEFAULT_INIT_NODE_CAPACITY,
            sliding_window_size=self.sliding_window_size,
            mamba_cache_chunk_size=self.mamba_cache_chunk_size,
            enable_hicache=self.enable_hierarchical_cache,
            hicache_write_back=(
                server_args.hicache_write_policy == "write_back"
                if self.enable_hierarchical_cache
                else False
            ),
        )
        self.root_node = self._rust_radix.default_root_idx()
        # Scheduler idle checks can run before the second-phase HiCache setup
        # populates these fields. Keep the attributes present even when the
        # controller is not initialized yet.
        self.ongoing_write_through: dict[int, Any] = {}
        self.ongoing_load_back: dict[int, tuple[list[int], int]] = {}
        self.enable_storage = False
        # Cache a single empty tensor for the disabled-cache match-result
        # path so we don't allocate per call. Callers must not mutate it
        # (empty tensors aren't typically mutated, so this is safe by
        # convention).
        self._empty_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        self._empty_host_indices = torch.empty((0,), dtype=torch.int64, device="cpu")

    def _debug_key(self, token_ids: list[int] | tuple[int, ...]) -> str:
        if not getattr(self, "_debug_trace", False):
            return ""
        head = list(token_ids[:4])
        tail = list(token_ids[-4:]) if len(token_ids) >= 4 else list(token_ids)
        return f"len={len(token_ids)} head={head} tail={tail}"

    def _debug_allocator_state(self, label: str) -> None:
        if not getattr(self, "_debug_trace", False):
            return
        allocator = self.token_to_kv_pool_allocator
        full_available = None
        swa_available = None
        if allocator is not None:
            if hasattr(allocator, "full_available_size"):
                full_available = allocator.full_available_size()
            elif hasattr(allocator, "available_size"):
                full_available = allocator.available_size()
            if hasattr(allocator, "swa_available_size"):
                swa_available = allocator.swa_available_size()
        logger.info(
            "rust_urt_debug sizes %s full_avail=%s full_evict=%s full_prot=%s "
            "swa_avail=%s swa_evict=%s swa_prot=%s active_nodes=%s",
            label,
            full_available,
            self.full_evictable_size(),
            self.full_protected_size(),
            swa_available,
            self.swa_evictable_size(),
            self.swa_protected_size(),
            self._rust_radix.active_tree_node_count(),
        )

    def _reject_unsupported(self, params: CacheInitParams) -> None:
        if params.eviction_policy.lower() != "lru":
            raise RadixCacheInfraPyError(
                f"RustUnifiedRadixCache: eviction_policy={params.eviction_policy!r} "
                "not supported, only 'lru'"
            )
        if params.cache_ttl_seconds is not None:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: cache_ttl_seconds not supported"
            )
        if params.enable_kv_cache_events:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: enable_kv_cache_events=True not supported"
            )
        if params.tree_components is not None:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: tree_components not supported"
            )

    # HiCache (host tier) restrictions — OSS has no equivalent gate.
    def _reject_unsupported_hicache(self, server_args: Any) -> None:
        # Device <-> host only (no L3 storage backend yet).
        if server_args.hicache_storage_backend is not None:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache HiCache: storage backend (L3) is not supported yet"
            )
        if server_args.hicache_write_policy not in (
            "write_through",
            "write_through_selective",
            "write_back",
        ):
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache HiCache: write_policy="
                f"{server_args.hicache_write_policy!r} is not supported yet "
                "(only write_through / write_through_selective / write_back)"
            )

    @staticmethod
    def _device_to_rust_str(device: Any) -> str:
        # Resolve unindexed cuda → "cuda:<current_device>" so per-rank
        # processes (TP > 1) get the right index. Rust's `parse_device`
        # treats bare "cuda" as `Cuda(0)`, which silently puts the rank-N
        # cache on cuda:0 even though incoming KV-index tensors are on
        # cuda:N → `InsertValueWrongDevice` at first insert. Allocator /
        # model_runner pass `device='cuda'` (the global server arg) on
        # all ranks, so the resolution must happen here at the boundary.
        def _resolve_cuda_index() -> str:
            try:
                return f"cuda:{torch.cuda.current_device()}"
            except Exception:
                return "cuda"

        if isinstance(device, torch.device):
            if device.type == "cpu":
                return "cpu"
            if device.type == "cuda":
                return (
                    f"cuda:{device.index}"
                    if device.index is not None
                    else _resolve_cuda_index()
                )
            raise RadixCacheInfraPyError(
                f"RustUnifiedRadixCache: device {device!r} not supported"
            )
        if isinstance(device, str) and device == "cuda":
            return _resolve_cuda_index()
        return str(device)

    # ----- HiCache (host tier) -----

    @staticmethod
    def _py_component_type():
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentType as PyComponentType,
        )

        return PyComponentType

    @staticmethod
    def _pool_name():
        from sglang.srt.mem_cache.hicache_storage import PoolName

        return PoolName

    def _as_host_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Normalize H2D source indices before queuing cache-controller ops."""
        if indices is None or len(indices) == 0:
            return self._empty_host_indices
        return indices.to(device="cpu", dtype=torch.int64)

    def _build_aux_backup_transfers(self, node_idx: int) -> dict[Any, list[Any]]:
        """Build per-component host backup transfers for a Rust node."""
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

        py_ct = self._py_component_type()
        comp_xfers: dict[Any, list[Any]] = {}
        if self.supports_swa():
            swa_value = self._rust_radix.get_swa_device_value(node_idx)
            if swa_value is not None:
                comp_xfers[py_ct.SWA] = [
                    PoolTransfer(name=PoolName.SWA, device_indices=swa_value.to(torch.int64))
                ]
        if self.supports_mamba():
            mamba_value = self._rust_radix.get_mamba_device_value(node_idx)
            if mamba_value is not None:
                comp_xfers[py_ct.MAMBA] = [
                    PoolTransfer(name=PoolName.MAMBA, device_indices=mamba_value)
                ]
        return comp_xfers

    def _build_sidecar_transfers(
        self,
        phase: str,
        kv_xfer: Any,
        comp_xfers: dict[Any, list[Any]],
    ) -> list[Any]:
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            CacheTransferPhase,
        )

        transfers = []
        for spec in self.sidecar_pool_specs:
            if spec.indices_from_pool == PoolName.KV:
                source = kv_xfer
            else:
                source_component = {
                    PoolName.SWA: self._py_component_type().SWA,
                    PoolName.MAMBA: self._py_component_type().MAMBA,
                }.get(spec.indices_from_pool)
                if source_component is None:
                    raise RadixCacheRuntimePyError(
                        f"Unsupported sidecar source pool {spec.indices_from_pool}"
                    )
                matching = comp_xfers.get(source_component, ())
                if not matching:
                    continue
                source = matching[0]

            indices = (
                source.device_indices
                if phase == CacheTransferPhase.BACKUP_HOST
                else source.host_indices
            )
            if indices is None or len(indices) == 0:
                continue
            transfers.append(
                PoolTransfer(
                    name=spec.pool_name,
                    keys=source.keys,
                    hit_policy=spec.hit_policy,
                    indices_from_pool=spec.indices_from_pool,
                )
            )
        return transfers

    def _commit_aux_host_values(self, node_idx: int, comp_xfers: dict[Any, list[Any]]) -> None:
        py_ct = self._py_component_type()
        pool_name = self._pool_name()
        if py_ct.SWA in comp_xfers:
            xfer = comp_xfers[py_ct.SWA][0]
            if xfer.host_indices is not None:
                replaced = self._rust_radix.set_host_swa_values(
                    [node_idx], [xfer.host_indices]
                )
                for old_host_value in replaced:
                    self.host_pool_group.get_pool(pool_name.SWA).free(old_host_value)
        if py_ct.MAMBA in comp_xfers:
            xfer = comp_xfers[py_ct.MAMBA][0]
            if xfer.host_indices is not None:
                replaced = self._rust_radix.set_host_mamba_values(
                    [node_idx], [xfer.host_indices]
                )
                for old_host_value in replaced:
                    self.host_pool_group.get_pool(pool_name.MAMBA).free(old_host_value)

    def _free_evict_result(self, result: Any) -> None:
        full_idx = int(ComponentType.Full)
        swa_idx = int(ComponentType.Swa)
        mamba_idx = int(ComponentType.Mamba)
        if self.token_to_kv_pool_allocator is not None:
            for freed in result.freed[full_idx]:
                self.token_to_kv_pool_allocator.free(freed)
            for freed in result.freed[swa_idx]:
                self.token_to_kv_pool_allocator.free_swa(freed)
        if self.supports_mamba():
            for freed in result.freed[mamba_idx]:
                self.req_to_token_pool.mamba_allocator.free(freed)
        self._process_evict_actions(result.deferred_actions)

    def _write_backup(
        self,
        node_indices: list[int],
        device_values: list[torch.Tensor],
        *,
        lock_device: bool = True,
        track_write_through: bool = True,
    ) -> list[int]:
        """Kick off the backup against `cache_controller` and reflect the
        successful backups in the tree."""
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            CacheTransferPhase,
        )

        if self.supports_mamba():
            # Mamba extra-buffer checkpoints are produced on the forward stream
            # and can be donated to the radix tree immediately afterward. The
            # HiCache write stream must not race that producer when backing the
            # checkpoint up to host, or later host load-back restores stale SSM
            # state while FULL KV is otherwise correct.
            torch.cuda.synchronize()

        backed_nodes: list[int] = []
        host_values: list[torch.Tensor] = []
        aux_by_node: list[dict[Any, list[Any]]] = []
        for node_idx, device_value in zip(node_indices, device_values):
            kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)
            comp_xfers = self._build_aux_backup_transfers(node_idx)
            extra_pools = [x for xfers in comp_xfers.values() for x in xfers]
            extra_pools.extend(
                self._build_sidecar_transfers(
                    CacheTransferPhase.BACKUP_HOST,
                    kv_xfer,
                    comp_xfers,
                )
            )
            host_indices = self.cache_controller.write(
                device_indices=device_value,
                node_id=node_idx,
                extra_pools=extra_pools or None,
            )
            if host_indices is None:
                self.evict_host(len(device_value))
                host_indices = self.cache_controller.write(
                    device_indices=device_value,
                    node_id=node_idx,
                    extra_pools=extra_pools or None,
                )
            if host_indices is None:
                # Stop if any node failed to back up — preserves host-value
                # contiguity (the backed-up set stays a gapless prefix).
                break
            backed_nodes.append(node_idx)
            host_values.append(host_indices)
            aux_by_node.append(comp_xfers)
        if not backed_nodes:
            return []
        if lock_device:
            self._rust_radix.set_host_full_values_and_lock_device(
                backed_nodes, host_values
            )
        else:
            self._rust_radix.set_host_full_values(backed_nodes, host_values)
        if track_write_through:
            for node_idx in backed_nodes:
                self.ongoing_write_through[node_idx] = node_idx
        for node_idx, comp_xfers in zip(backed_nodes, aux_by_node):
            self._commit_aux_host_values(node_idx, comp_xfers)
        return backed_nodes

    def evict_host(self, num_tokens: int, component_type: Any = None) -> int:
        """Best effort to free up at least `num_tokens` host-tier KV in LRU order."""
        if self.cache_controller is None or num_tokens <= 0:
            return 0
        py_ct = self._py_component_type()
        if component_type is None or component_type == py_ct.FULL:
            result = self._rust_radix.evict_host(num_tokens)
            idx = int(ComponentType.Full)
        elif component_type == py_ct.SWA:
            result = self._rust_radix.evict_host_swa(num_tokens)
            idx = int(ComponentType.Swa)
        elif component_type == py_ct.MAMBA:
            result = self._rust_radix.evict_host_mamba(num_tokens)
            idx = int(ComponentType.Mamba)
        else:
            raise RadixCacheRuntimePyError(
                f"RustUnifiedRadixCache: unsupported host eviction component {component_type}"
            )
        self._process_evict_actions(result.deferred_actions)
        return result.evicted[idx]

    def _process_evict_actions(self, deferred_actions: list[tuple]) -> int:
        """Process evict actions generated by the Rust radix tree."""
        rolled_back_tokens = 0
        for action in deferred_actions:
            tag = action[0]
            if tag == "FullDeviceEvictOnBackedUp":
                # Free the device value on an already backed-up node.
                _, node_idx, device_value = action
                logger.info(
                    "evict_to_host: node=%s tokens=%s policy=backed_up",
                    node_idx,
                    len(device_value),
                )
                aux_result = self._rust_radix.demote_aux_device_values(
                    node_idx, device_value
                )
                self._free_evict_result(aux_result)
                if self.token_to_kv_pool_allocator is not None:
                    self.token_to_kv_pool_allocator.free(device_value)
            elif tag == "FullWriteBackOnEvict":
                _, node_idx, device_value = action
                backed_nodes = self._write_backup(
                    [node_idx],
                    [device_value],
                    lock_device=False,
                    track_write_through=False,
                )
                if backed_nodes:
                    self._wait_for_write_back_ack(node_idx)
                    logger.info(
                        "evict_to_host: node=%s tokens=%s policy=write_back",
                        node_idx,
                        len(device_value),
                    )
                    aux_result = self._rust_radix.demote_aux_device_values(
                        node_idx, device_value
                    )
                    self._free_evict_result(aux_result)
                    if self.token_to_kv_pool_allocator is not None:
                        self.token_to_kv_pool_allocator.free(device_value)
                else:
                    self._rust_radix.restore_full_values([node_idx], [device_value])
                    rolled_back_tokens += len(device_value)
            elif tag == "FullHostEvict":
                # Free host value.
                _, _node_idx, host_value = action
                self.token_to_kv_pool_host.free(host_value)
            elif tag == "SwaHostEvict":
                _, _node_idx, host_value = action
                self.host_pool_group.get_pool(self._pool_name().SWA).free(host_value)
            elif tag == "MambaHostEvict":
                _, _node_idx, host_value = action
                self.host_pool_group.get_pool(self._pool_name().MAMBA).free(host_value)
            else:
                raise RadixCacheRuntimePyError(
                    f"_process_evict_actions: unsupported evict action {tag!r}"
                )
        return rolled_back_tokens

    def _wait_for_write_back_ack(self, node_idx: int) -> None:
        """Synchronously drain write acks until the write-back for `node_idx`
        is complete, processing any earlier write-through acks in queue order."""
        while True:
            if not self.cache_controller.ack_write_queue:
                raise RadixCacheRuntimePyError(
                    f"write-back ack for node {node_idx} was not queued"
                )
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            matched = False
            for ack_node_id in ack.node_ids:
                if ack_node_id == node_idx:
                    matched = True
                if ack_node_id in self.ongoing_write_through:
                    self._rust_radix.dec_backup_lock_ref(ack_node_id)
                    self.ongoing_write_through.pop(ack_node_id, None)
            if matched:
                return

    # ----- BasePrefixCache contract: lifecycle -----

    def reset(self) -> None:
        # Mirrors UnifiedRadixCache.reset(): clear runtime tree state and the
        # L2 host allocator bookkeeping together. Keeping old host allocations
        # after dropping Rust's host LRU would make later write-through backups
        # unable to reclaim host memory.
        self._rust_radix.reset()
        self.root_node = self._rust_radix.default_root_idx()
        self.ongoing_write_through.clear()
        self.ongoing_load_back.clear()
        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()

    def supports_fast_match_prefix(self) -> bool:
        return True

    # ----- BasePrefixCache contract: lookup / insert / evict -----

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if params.cow_mamba and not self.supports_mamba():
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: MatchPrefixParams.cow_mamba=True requires "
                "Mamba configuration (HybridReqToTokenPool)"
            )

        if self.disable:
            # Disabled cache: skip Rust entirely. `None` signals "nothing to
            # lock" so inc/dec_lock_ref short-circuit. This is the only path
            # that produces a `None` last-node — every other path passes
            # through Rust's idx (which may be a namespace root for empty
            # keys / no-match cases, where lock-ref ops are documented no-ops).
            return self._empty_match_result()

        token_ids = params.key.raw_token_ids()
        rust_result = self._rust_radix.match_prefix(token_ids, params.key.extra_key)
        if getattr(self, "_debug_trace", False):
            logger.info(
                "rust_urt_debug match key=%s device=%s host=%s swa_host=%s mamba_host=%s "
                "last_device=%s last_host=%s best=%s",
                self._debug_key(token_ids),
                len(rust_result.device_indices),
                rust_result.host_only_length,
                rust_result.swa_host_hit_length,
                rust_result.mamba_host_hit_length,
                rust_result.last_device_node_idx,
                rust_result.last_host_node_idx,
                rust_result.best_match_node_idx,
            )

        # Mamba CoW: copy the SSM state of matched node to callers so
        # they could directly manipulate it freely.
        if params.cow_mamba and rust_result.mamba_value is not None:
            self._copy_on_write_mamba(
                params.req, rust_result.last_device_node_idx, rust_result.mamba_value
            )

        return MatchResult(
            device_indices=rust_result.device_indices,
            last_device_node=rust_result.last_device_node_idx,
            last_host_node=rust_result.last_host_node_idx,
            best_match_node=rust_result.best_match_node_idx,
            host_hit_length=rust_result.host_only_length,
            swa_host_hit_length=rust_result.swa_host_hit_length,
            mamba_host_hit_length=rust_result.mamba_host_hit_length,
            mamba_branching_seqlen=rust_result.mamba_branching_seqlen,
        )

    def _extract_mamba_value(
        self, req: "Req"
    ) -> tuple[Optional[torch.Tensor], Optional[int]]:
        """Build the `mamba_value` tensor to insert into the radix tree."""
        if not self.supports_mamba() or req.mamba_pool_idx is None:
            return None, None
        if not self.enable_mamba_extra_buffer:
            return req.mamba_pool_idx.unsqueeze(-1).clone(), None
        # extra_buffer mode: keep the buffer slot that contains the most
        # recent tracked state. This must match MambaComponent and
        # MambaRadixCache; lazy mode keeps the active state at next_track_idx.
        track_buffer_to_keep = self.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
        mamba_value = (
            req.mamba_ping_pong_track_buffer[track_buffer_to_keep].unsqueeze(-1).clone()
        )
        assert mamba_value.item() != -1, (
            f"Cached mamba slot is -1: keep_idx={track_buffer_to_keep}, "
            f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
            f"next_track_idx={req.mamba_next_track_idx}, "
            f"last_track_seqlen={req.mamba_last_track_seqlen}, "
            f"rid={req.rid}"
        )
        return mamba_value, track_buffer_to_keep

    def _mamba_fork_from(
        self,
        mamba_value: torch.Tensor,
        protect_node_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Fork `mamba_value` in the pool; only try to evict if direct
        allocation failed.

        TODO(Jialin): port this retry-with-alloc wrapper into OSS
        `MambaPool` so callers don't reimplement it.
        """
        mamba_pool = self.req_to_token_pool.mamba_pool
        mamba_allocator = self.req_to_token_pool.mamba_allocator
        dst = mamba_allocator.alloc(1)
        if dst is None:
            if protect_node_idx is not None:
                self.inc_lock_ref(protect_node_idx)
            try:
                self.evict(EvictParams(num_tokens=0, mamba_num=1))
                dst = mamba_allocator.alloc(1)
            finally:
                if protect_node_idx is not None:
                    self.dec_lock_ref(protect_node_idx)
            assert dst is not None, "Can not alloc mamba cache"
        mamba_pool.copy_from(mamba_value, dst)
        return dst

    def _alloc_mamba_slot_for_cow(
        self, protect_node_idx: Optional[int] = None
    ) -> torch.Tensor:
        mamba_allocator = self.req_to_token_pool.mamba_allocator
        dst = mamba_allocator.alloc(1)
        if dst is None:
            if protect_node_idx is not None:
                self.inc_lock_ref(protect_node_idx)
            try:
                self.evict(EvictParams(num_tokens=0, mamba_num=1))
                dst = mamba_allocator.alloc(1)
            finally:
                if protect_node_idx is not None:
                    self.dec_lock_ref(protect_node_idx)
            assert dst is not None, "Can not alloc mamba cache"
        return dst

    def _copy_on_write_mamba(
        self, req: "Req", last_node_idx: int, src_index: torch.Tensor
    ) -> None:
        """Defer copying the matched Mamba SSM state into req-local space."""
        if req.mamba_pool_idx is None:
            req.mamba_pool_idx = self._alloc_mamba_slot_for_cow(
                protect_node_idx=last_node_idx
            )[0]
        req.mamba_cow_src_index = src_index
        req.mamba_needs_clear = False

    def _empty_match_result(self) -> MatchResult:
        # Disabled-cache sentinel. Reuses the cached `_empty_indices` to
        # avoid per-call tensor allocation. `last_device_node=None` makes
        # inc/dec_lock_ref short-circuit so callers don't need to branch.
        return MatchResult(
            device_indices=self._empty_indices,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if params.mamba_value is not None and not self.supports_mamba():
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: InsertParams.mamba_value requires Mamba "
                "configuration (HybridReqToTokenPool)"
            )
        if params.priority != 0:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: InsertParams.priority != 0 not supported (LRU only)"
            )
        # `params.chunked` only affects Python's hit_count, which LRU never
        # reads — silently ignored.

        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        if value is None:
            value = torch.tensor(key.raw_token_ids(), dtype=torch.int64, device=self.device)

        # Normalize the key's `is_bigram` to match `self.is_eagle` at the
        # orchestrator boundary, mirroring OSS `RadixCache.insert` /
        # `SWARadixCache.insert`. Defensive against an external caller
        # passing `RadixKey(is_bigram=False)` to an `is_eagle=True`
        # orchestrator (or vice versa) — without this, downstream
        # `page_aligned(...)` + `len(aligned_key)` math would silently
        # disagree with the Rust wrapper's bigram-pair-count and corrupt
        # `prefix_len` accounting. Idempotent when the caller already
        # set `is_bigram=self.is_eagle` (the `cache_*_req` path).
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)

        # Orchestrator owns page-alignment in atom units. Delegate to
        # `RadixKey.page_aligned`, which is bigram-aware via the key's
        # (now-normalized) `is_bigram` flag.
        # `len(aligned_key)` is the atom count (= N-1 for is_bigram=True,
        # N otherwise); we slice `value` to that length so the cache's
        # value-length invariant holds at the atom granularity. The trim
        # is idempotent: callers that pre-align hit a no-op here.
        #
        # TODO(future PR): make the Rust wrapper / cache layer reject
        # non-aligned keys with a typed error instead of silently
        # trimming. Today `PageAlignedQueryKey::new` does an internal
        # `key.len() / ps * ps` trim — a contract-violation safety net
        # rather than an explicit invariant.
        aligned_key = key.page_aligned(self.page_size)
        atom_count = len(aligned_key)
        token_ids = aligned_key.token_ids
        # Trim value to atom_count. If the caller passed a shorter value
        # the slice returns the original (still-too-short) tensor; the
        # Rust cache layer catches it via `validate_insert_value` →
        # `RadixCacheRuntimePyError::InsertValueTooShort`, so we don't
        # duplicate the check here.
        value = value[:atom_count] if atom_count > 0 else value

        rust_result = self._rust_radix.insert(
            token_ids,
            value,
            aligned_key.extra_key,
            params.prev_prefix_len,
            params.swa_evicted_seqlen,
            params.mamba_value,
        )
        if getattr(self, "_debug_trace", False):
            logger.info(
                "rust_urt_debug insert key=%s atom=%s prefix=%s prev=%s swa_evict=%s "
                "actions=%s mamba_exists=%s",
                self._debug_key(token_ids),
                atom_count,
                rust_result.prefix_len,
                params.prev_prefix_len,
                params.swa_evicted_seqlen,
                [action[0] for action in rust_result.deferred_actions],
                rust_result.mamba_value_exists,
            )
        self._process_insert_actions(rust_result.deferred_actions)
        return InsertResult(
            prefix_len=rust_result.prefix_len,
            mamba_exist=rust_result.mamba_value_exists,
        )

    def _process_insert_actions(self, deferred_actions: list[tuple]) -> None:
        """Apply the insert-path emitted actions in the orchestration layer."""
        if not deferred_actions or self.token_to_kv_pool_allocator is None:
            return

        self._debug_allocator_state("insert_actions:start")
        swa_node_indices: list[int] = []
        swa_values: list[torch.Tensor] = []
        write_through_nodes: list[int] = []
        write_through_values: list[torch.Tensor] = []
        for action in deferred_actions:
            tag = action[0]
            if tag == "FullDupFreed":
                _, freed_indices = action
                self.token_to_kv_pool_allocator.free(freed_indices)
                self._debug_allocator_state(f"insert_actions:after:{tag}:{len(freed_indices)}")
            elif tag == "SwaRecover":
                _, node_idx, freed_full, source_value = action
                self.token_to_kv_pool_allocator.free(freed_full)
                self._debug_allocator_state(f"insert_actions:after:{tag}:free:{len(freed_full)}")
                swa_node_indices.append(node_idx)
                swa_values.append(
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        source_value
                    )
                )
            elif tag == "SwaStamp":
                _, node_idx, source_value = action
                swa_node_indices.append(node_idx)
                swa_values.append(
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        source_value
                    )
                )
            elif tag == "FullWriteThroughBackup":
                _, node_idx, device_value = action
                write_through_nodes.append(node_idx)
                write_through_values.append(device_value)
            else:
                raise RadixCacheRuntimePyError(
                    f"_process_insert_actions: unsupported insert action {tag!r}"
                )

        if swa_node_indices:
            # Single batched apply_swa_writes call — stamps SWA values on
            # all affected nodes, splices into SWA's LRU, credits
            # evictable_size. Mirrors OSS's per-action insert_mru pattern
            # collapsed into one call.
            self._rust_radix.apply_swa_writes(swa_node_indices, swa_values)
            self._debug_allocator_state(
                f"insert_actions:after:apply_swa_writes:{len(swa_node_indices)}"
            )
        if write_through_nodes:
            # Back up FULL values from device to host (write-through). This must
            # run after SWA stamps/recovers so auxiliary host transfers and
            # sidecars derived from SWA see the freshly populated device values.
            self._write_backup(write_through_nodes, write_through_values)
            self._debug_allocator_state(
                f"insert_actions:after:write_backup:{len(write_through_nodes)}"
            )

    def evict(self, params: EvictParams) -> EvictResult:
        if params.mamba_num != 0 and not self.supports_mamba():
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: EvictParams.mamba_num != 0 requires "
                "Mamba configuration (HybridReqToTokenPool)"
            )

        full_budget = max(0, params.num_tokens)
        swa_budget = (
            max(0, params.swa_num_tokens) if self.sliding_window_size is not None else 0
        )
        mamba_budget = max(0, params.mamba_num) if self.supports_mamba() else 0
        if self.disable or (full_budget == 0 and swa_budget == 0 and mamba_budget == 0):
            return EvictResult(num_tokens_evicted=0)

        self._poll_hicache_events_for_eviction()

        # Single Rust call: the dispatcher iterates configured components
        # forward (FULL → SWA → Mamba) with the per-component budget.
        # FULL leaf-evict cross-bumps `result.freed[Swa]` via the
        # `free_swa(full_value)` cookie cascade — iterating both bins
        # below is what releases those cascaded handles back to the SWA
        # allocator. Iterating adapters here would re-trigger the
        # dispatcher per call and (a) double-count budget, (b) drop the
        # cascaded SWA handles from the FULL call's result.
        start_time = time.perf_counter()
        full_idx = int(ComponentType.Full)
        swa_idx = int(ComponentType.Swa)
        mamba_idx = int(ComponentType.Mamba)
        evicted = [0, 0, 0]
        requested = [full_budget, swa_budget, mamba_budget]
        target_full_available = (
            full_budget if full_budget > 0 else None
        )
        attempts = 0
        while any(requested):
            attempts += 1
            result = self._rust_radix.evict(requested)
            if getattr(self, "_debug_trace", False):
                logger.info(
                    "rust_urt_debug evict requested=%s evicted=%s freed=%s actions=%s",
                    requested,
                    list(result.evicted),
                    [len(bin) for bin in result.freed],
                    [action[0] for action in result.deferred_actions],
                )

            # When a component isn't configured, the Rust dispatcher
            # doesn't iterate it, so its `evicted[ct] == 0` and
            # `freed[ct] == []` — the SWA/MAMBA branches below are
            # safe-by-shape and don't need a `sliding_window_size`
            # gate. (FULL leaf-evict CAN cross-bump `freed[Swa]` via the
            # `free_swa(full_value)` cookie cascade even when
            # `swa_budget == 0`, so we always iterate the SWA bin when
            # SWA is configured — empty otherwise.)
            batch_evicted = list(result.evicted)
            if self.token_to_kv_pool_allocator is not None:
                for freed in result.freed[full_idx]:
                    self.token_to_kv_pool_allocator.free(freed)
                for freed in result.freed[swa_idx]:
                    self.token_to_kv_pool_allocator.free_swa(freed)
            if self.supports_mamba():
                for freed in result.freed[mamba_idx]:
                    self.req_to_token_pool.mamba_allocator.free(freed)

            rolled_back_tokens = self._process_evict_actions(result.deferred_actions)
            if rolled_back_tokens:
                batch_evicted[full_idx] = max(
                    0, batch_evicted[full_idx] - rolled_back_tokens
                )
            for i, count in enumerate(batch_evicted):
                evicted[i] += count

            # Full-only serving allocation needs the allocator to actually
            # expose enough free device slots after Rust-side demotion. The
            # Rust tree reports logical eviction size; split/loadback/page
            # rounding can make the allocator-visible release slightly smaller.
            # Retry the visible deficit instead of letting alloc_extend fail.
            if (
                target_full_available is None
                or self.token_to_kv_pool_allocator is None
                or not hasattr(self.token_to_kv_pool_allocator, "available_size")
            ):
                break
            full_available = self.token_to_kv_pool_allocator.available_size()
            if full_available >= target_full_available:
                break
            if self._drain_hicache_events_for_eviction():
                full_available = self.token_to_kv_pool_allocator.available_size()
                if full_available >= target_full_available:
                    break
                requested = [
                    target_full_available - full_available,
                    0,
                    0,
                ]
                continue
            if batch_evicted[full_idx] == 0 or attempts >= 8:
                break
            requested = [
                target_full_available - full_available,
                0,
                0,
            ]

        self.update_eviction_metrics(sum(evicted), start_time)
        return EvictResult(
            num_tokens_evicted=evicted[full_idx],
            swa_num_tokens_evicted=evicted[swa_idx],
            mamba_num_evicted=evicted[mamba_idx],
        )

    # ----- BasePrefixCache contract: lock_ref -----

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        # `None` covers both: empty match (no node to lock) and disabled cache.
        if self.disable or node is None:
            return IncLockRefResult(delta=0)
        # Rust's `inc_lock_ref` is a per-cache dispatcher: forward iter
        # over all configured components (FULL always; SWA when
        # configured). Returns `(delta, swa_uuid_for_lock)` where delta
        # is the signed change to evictable_token_size aggregated across
        # components, and swa_uuid_for_lock is `Some(uuid)` when SWA
        # stamped a window boundary (for symmetric release later).
        delta, swa_uuid_for_lock = self._rust_radix.inc_lock_ref(node)
        return IncLockRefResult(delta=delta, swa_uuid_for_lock=swa_uuid_for_lock)

    def dec_lock_ref(
        self,
        node: Any,
        params: Optional[DecLockRefParams] = None,
    ) -> DecLockRefResult:
        if self.disable or node is None:
            return DecLockRefResult()
        # Rust's `dec_lock_ref` is the symmetric per-cache dispatcher
        # (reverse iter — SWA then FULL). `swa_uuid_for_lock` gates
        # SWA's release walk to stop at the matching boundary; FULL's
        # walk is unconditional. FULL-only configs pass `None` and the
        # param is ignored Rust-side.
        swa_uuid_for_lock = params.swa_uuid_for_lock if params is not None else None
        self._rust_radix.dec_lock_ref(node, swa_uuid_for_lock)
        return DecLockRefResult()

    # ----- BasePrefixCache contract: size accessors -----

    def evictable_size(self) -> int:
        return self._rust_radix.evictable_token_size()

    def protected_size(self) -> int:
        return self._rust_radix.protected_token_size()

    def total_size(self) -> int:
        # Total tokens (evictable + protected) across FULL and SWA components.
        return self._rust_radix.total_token_size()

    # Per-component aliases (mirror OSS `UnifiedRadixCache.full_*` /
    # `swa_*`). Scheduler reads these directly when `is_hybrid_swa`
    # (e.g. `schedule_policy.py rem_total_tokens`); without the
    # overrides the inherited `BasePrefixCache` defaults would silently
    # return 0 and starve the hybrid capacity calculation.

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        if self.sliding_window_size is None:
            return 0
        return self._rust_radix.swa_evictable_token_size()

    def swa_protected_size(self) -> int:
        if self.sliding_window_size is None:
            return 0
        return self._rust_radix.swa_protected_token_size()

    def mamba_evictable_size(self) -> int:
        return (
            self._rust_radix.mamba_evictable_token_size()
            if self.supports_mamba()
            else 0
        )

    def mamba_protected_size(self) -> int:
        return (
            self._rust_radix.mamba_protected_token_size()
            if self.supports_mamba()
            else 0
        )

    def mamba_total_size(self) -> int:
        # Total Mamba slots (evictable + protected); separate from `total_size()` because Mamba's unit is slots, not tokens.
        return self._rust_radix.mamba_total_size() if self.supports_mamba() else 0

    def _poll_hicache_events_for_eviction(self) -> None:
        """Release ready HiCache locks before an allocator-pressure eviction."""
        if self.cache_controller is None:
            return
        self.writing_check()
        self.loading_check()

    def _drain_hicache_events_for_eviction(self) -> bool:
        """Wait for in-flight HiCache copies only when eviction cannot make
        allocator-visible progress. This keeps the normal path asynchronous,
        but avoids treating tokens locked by completed/near-completed copies as
        immediately evictable under allocation pressure."""
        if self.cache_controller is None:
            return False
        before = (
            len(self.ongoing_write_through),
            len(self.ongoing_load_back),
        )
        self.writing_check()
        self.loading_check()
        if before != (len(self.ongoing_write_through), len(self.ongoing_load_back)):
            return True

        waited = False
        if self.ongoing_write_through and self.cache_controller.ack_write_queue:
            self.writing_check(write_back=True)
            waited = True
        if self.ongoing_load_back and self.cache_controller.ack_load_queue:
            self.loading_check(wait=True)
            waited = True
        after = (
            len(self.ongoing_write_through),
            len(self.ongoing_load_back),
        )
        return waited and before != after

    # ----- BasePrefixCache contract: idle invariant check -----

    def sanity_check(self) -> None:
        self._rust_radix.sanity_check()

    # ----- BasePrefixCache contract: SWA capability flag -----

    def supports_swa(self) -> bool:
        # Gates `Scheduler.maybe_evict_swa()` and the schedule-policy
        # paths that preserve `swa_uuid_for_lock` across decode steps.
        # Without `True` here, decode-time SWA evictions never fire and
        # `dec_lock_ref` calls land at the Rust dispatcher with
        # `swa_uuid_for_lock=None`, walking past the SWA boundary and
        # underflowing `swa_lock_ref`.
        return self.sliding_window_size is not None

    def supports_mamba(self) -> bool:
        return self.mamba_cache_chunk_size is not None

    # TODO(Jialin): expose Rust-side iteration; leak-diagnostic only.
    def all_values_flatten(self) -> torch.Tensor:
        return self._empty_indices

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._empty_indices

    # ----- BasePrefixCache contract: features rejected in v1 -----

    def pretty_print(self):
        logger.error(
            "RustUnifiedRadixCache state: active_nodes=%s total=%s "
            "evictable=%s protected=%s ongoing_write=%s ongoing_load=%s",
            self._rust_radix.active_tree_node_count(),
            self.total_size(),
            self.evictable_size(),
            self.protected_size(),
            len(getattr(self, "ongoing_write_through", {})),
            len(getattr(self, "ongoing_load_back", {})),
        )

    def take_events(self):
        # `enable_kv_cache_events=True` is rejected at __init__, so the queue
        # is always empty.
        return []

    # ----- Per-request orchestration -----

    def cache_finished_req(self, req: "Req", is_insert: bool = True, **kwargs) -> None:
        """Mirrors `sglang.srt.mem_cache.radix_cache.RadixCache.cache_finished_req`.

        Cache the prefix of a finished request and free its tail. The disabled
        path frees everything; the inserting path inserts the page-aligned
        prefix and frees only the duplicate slots that the tree already owned.
        """
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            if self.supports_mamba():
                self.req_to_token_pool.free_mamba_cache(req)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Mamba extra_buffer mode: truncate the cache range to Mamba chunk aligned.
        if self.enable_mamba_extra_buffer:
            cache_len = min(req.mamba_last_track_seqlen or 0, len(token_ids))
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        atom_len = len(radix_key)
        values = kv_indices[:atom_len].to(dtype=torch.int64, copy=True)

        mamba_value, mamba_ping_pong_track_buffer_to_keep = self._extract_mamba_value(
            req
        )

        mamba_exist = False
        if is_insert:
            self._debug_allocator_state("cache_finished:before_insert")
            insert_result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    prev_prefix_len=req.cache_protected_len,
                    swa_evicted_seqlen=req.swa_evicted_seqlen,
                    mamba_value=mamba_value,
                )
            )
            mamba_exist = insert_result.mamba_exist
            self._debug_allocator_state("cache_finished:after_insert")
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : atom_len]
            )
            # Skipped insert: caller still owns the Mamba slot.
            mamba_exist = mamba_value is not None

        # Free everything past the aligned atom prefix. For bigram,
        # this includes the trailing boundary token of the last cached
        # pair as well as any unaligned bigram positions.
        self.token_to_kv_pool_allocator.free(kv_indices[atom_len:])
        self._debug_allocator_state("cache_finished:after_free_tail")

        # Mamba slot release.
        #
        #              | tree takes ownership of | req-owned slots to free
        # -------------|-------------------------|--------------------------------
        # extra_buffer | ping_pong[keep_idx]     | primary + ping_pong[other_idx]
        # extra_buffer | nothing (mamba_exist)   | primary + ping_pong[0,1]
        # no_buffer    | primary                 | nothing
        # no_buffer    | nothing (mamba_exist)   | primary
        #
        # extra_buffer: primary is ALWAYS orphaned (the tree took a ping-pong slot,
        # not the primary), so always invoke free_mamba_cache. The
        # `ping_pong_track_buffer_to_keep` arg tells the pool which ping-pong slot
        # to spare; set to None on mamba_exist so all three slots are freed.
        # no_buffer: primary IS the slot handed to the tree, so free only when the
        # tree rejected (mamba_exist=True).
        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None
        free_mamba_cache = self.enable_mamba_extra_buffer or mamba_exist
        if self.supports_mamba() and free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )

        # Release the lock taken when this req was scheduled for prefill.
        # Pass through `swa_uuid_for_lock` so SWA's release walk stops at
        # the right boundary node. FULL-only configs always have it as
        # None (DecLockRefParams default).
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
        )

    def cache_unfinished_req(self, req: "Req", chunked: bool = False, **kwargs) -> None:
        if self.disable:
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Mamba extra_buffer mode: truncate the cache range to Mamba chunk aligned.
        if self.enable_mamba_extra_buffer:
            cache_len = req.mamba_last_track_seqlen
            # No Mamba chunk-aligned boundary reached yet, skip caching.
            if cache_len is None:
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                return
            cache_len = min(cache_len, len(token_ids))
            token_ids = token_ids[:cache_len]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        atom_len = len(radix_key)
        values = kv_indices[:atom_len].to(dtype=torch.int64, copy=True)

        # Mamba: hand the tracked state to the radix cache. In extra_buffer
        # mode this must donate the ping-pong slot and replace it with a fresh
        # request-owned slot, matching MambaComponent. Copying the tracked slot
        # here can race the forward stream that produced it.
        mamba_value_forked = None
        if self.supports_mamba() and req.mamba_pool_idx is not None:
            if self.enable_mamba_extra_buffer:
                new_slot = self._alloc_mamba_slot_for_cow()
                mamba_value_forked = (
                    self.req_to_token_pool.donate_mamba_ping_pong_slot(
                        req, new_slot
                    )
                )
            else:
                mamba_value_src, _ = self._extract_mamba_value(req)
                assert mamba_value_src is not None, (
                    "mamba_value_src must be present when supports_mamba() and "
                    "req.mamba_pool_idx is not None"
                )
                mamba_value_forked = self._mamba_fork_from(mamba_value_src)

        insert_result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                prev_prefix_len=req.cache_protected_len,
                priority=getattr(req, "priority", 0) or 0,
                mamba_value=mamba_value_forked,
            )
        )

        # Mamba: release the forked slot when the cache didn't consume
        # it (target already had a Mamba value).
        if mamba_value_forked is not None and insert_result.mamba_exist:
            self.req_to_token_pool.mamba_allocator.free(mamba_value_forked)

        # Re-match: the tree may have de-duplicated against an existing branch
        # during insert, so the canonical tree-owned indices for this prefix
        # may differ from `values` we just inserted. We need those canonical
        # indices in `req_to_token_pool` so subsequent reads see the survivor.
        #
        # SWA caveat: `match_prefix` can legitimately return FEWER indices
        # than the inserted atom count when the path crosses an SWA
        # tombstone and the contiguous post-tombstone run hasn't refilled
        # `sliding_window_size` yet. So the tight `len(new_indices) ==
        # atom_len` check would crash valid SWA states. Mirrors baseline
        # `swa_radix_cache.py`'s `assert old_prefix_len <= new_prefix_len
        # <= len(keys_np)` (and `unified_radix_cache.py`'s equivalent
        # partial-rematch tolerance). The bookkeeping below already uses
        # `len(new_indices)` everywhere, so a short rematch slots in
        # cleanly: `req.prefix_indices` keeps the unmatched tail,
        # `cache_protected_len` records what the tree actually owns.
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        # Ensure `cache_protected_len` does not extend beyond the
        # last aligned position the tree owns. The `+ page_size - 1`
        # slack tolerates a trailing partial page (page-aligned for
        # FULL/SWA; chunk-aligned for Mamba extra_buffer).
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ) and len(new_indices) <= atom_len, (
            f"cache_unfinished_req post-insert rematch out of bounds: "
            f"{req.cache_protected_len=}, {len(new_indices)=}, "
            f"{atom_len=}, {self.page_size=}"
        )

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # Lock-ref handoff. dec first, inc second so the brief moment between
        # the two doesn't hold a redundant +2 along ancestors that overlap
        # between old and new last_node (this method is synchronous wrt
        # eviction, so the brief drop is safe).
        # Pass `swa_uuid_for_lock` to dec so SWA's release walk stops at
        # the boundary node stamped at acquire time. After the new
        # acquire, store the new uuid back on req for the next call.
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
        )
        inc_result = self.inc_lock_ref(new_last_node)
        req.swa_uuid_for_lock = inc_result.swa_uuid_for_lock

        # Extend back kv indices after the last Mamba chunk or page-aligned boundary.
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node
        req.cache_protected_len = len(new_indices)
        # Clear the chunk-aligned marker so the next call to
        # `cache_unfinished_req` recomputes from the request's current
        # progress instead of reusing a stale boundary.
        if self.supports_mamba():
            req.mamba_last_track_seqlen = None

    # ----- HiCache: OSS-identical bodies -----
    # TODO(Jialin): introduce HiCacheMixin in OSS for consolidation.

    def init_hicache(self, server_args: Any, params: CacheInitParams) -> None:
        """Second-phase setup: build the host pool + `HiCacheController`.

        The factory calls this after construction when
        `enable_hierarchical_cache` is set. Mirrors OSS
        `HiRadixCache.__init__`'s host setup, restricted to the supported
        config: FULL-only, write-through, device<->host only (no
        storage/prefetch).
        """
        self.kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _select_strategy,
        )
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentType as PyComponentType,
        )

        components = {PyComponentType.FULL}
        if self.supports_swa():
            components.add(PyComponentType.SWA)
        if self.supports_mamba():
            components.add(PyComponentType.MAMBA)

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.load_cache_event = threading.Event()

        strategy = _select_strategy(self.kv_cache, components)
        result = strategy.build(
            cache=self,
            kvcache=self.kv_cache,
            params=params,
            server_args=server_args,
            load_cache_event=self.load_cache_event,
            attn_cp_group=self.attn_cp_group,
            attn_tp_group=self.attn_tp_group,
            storage_backend=None,
            storage_backend_extra_config=None,
            prefetch_threshold=256,
            model_name=server_args.served_model_name,
            enable_storage_metrics=False,
        )

        self.host_pool_group = result.host_pool_group
        self.cache_controller = result.cache_controller
        self.sidecar_pool_specs = list(result.sidecars)
        self.token_to_kv_pool_host = result.component_host_pools[PyComponentType.FULL]
        self.full_kv_pool_host = self.token_to_kv_pool_host
        if PyComponentType.SWA in result.component_host_pools:
            self.swa_kv_pool_host = result.component_host_pools[PyComponentType.SWA]
        if PyComponentType.MAMBA in result.component_host_pools:
            self.mamba_pool_host = result.component_host_pools[PyComponentType.MAMBA]

        self.kv_cache.register_layer_transfer_counter(
            self.cache_controller.layer_done_counter
        )
        if result.register_req_to_token_counter:
            self.req_to_token_pool.register_layer_transfer_counter(
                self.cache_controller.layer_done_counter
            )

        # Nodes with an in-flight write-through; the host stamp is deferred to
        # the controller ack (`writing_check`). Keyed by Rust NodeIdx.
        self.ongoing_write_through: dict[int, Any] = {}
        # Nodes with an in-flight load-back; the device lock handed off by
        # `postprocess_load_back`, plus host source locks from
        # `prepare_load_back`, are released on the ack (`loading_check`).
        self.ongoing_load_back: dict[int, tuple[list[int], int]] = {}
        # L3 storage tier not supported yet
        self.enable_storage = False
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

    def writing_check(self, write_back: bool = False) -> None:
        """Release the device lock on nodes whose host copy has completed."""
        if not self.ongoing_write_through:
            return
        if write_back:
            while self.ongoing_write_through:
                if not self.cache_controller.ack_write_queue:
                    raise RadixCacheRuntimePyError(
                        "RustUnifiedRadixCache: pending write-through lock has no ack"
                    )
                ack = self.cache_controller.ack_write_queue.pop(0)
                ack.finish_event.synchronize()
                for node_id in ack.node_ids:
                    if node_id in self.ongoing_write_through:
                        self._rust_radix.dec_backup_lock_ref(node_id)
                        self.ongoing_write_through.pop(node_id, None)
            return

        finish_count = 0
        if self.pp_rank == 0:
            for ack in self.cache_controller.ack_write_queue:
                if not ack.finish_event.query():
                    break
                finish_count += 1
        finish_count = self._hicache_min_ready(finish_count)
        while finish_count > 0:
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for node_id in ack.node_ids:
                if node_id in self.ongoing_write_through:
                    self._rust_radix.dec_backup_lock_ref(node_id)
                    self.ongoing_write_through.pop(node_id, None)
            finish_count -= 1

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def _hicache_min_ready(self, finish_count: int) -> int:
        # All ranks must drain the same number of acks to keep the queue in
        # lockstep; MIN-reduce the locally-ready count over the attn / TP groups.
        if (
            self.tp_world_size <= 1
            and self.attn_cp_group is None
            and self.attn_tp_group is None
        ):
            return finish_count
        tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.MIN, group=group
                )
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        return int(tensor.item())

    def prepare_load_back(self, node_idx: int) -> Any:
        """Compute host-only chain to loadback (starting from node closer to root)."""
        return self._rust_radix.prepare_load_back(node_idx)

    def prepare_aux_load_back(self, node_idx: int) -> Any:
        return self._rust_radix.prepare_aux_load_back(node_idx)

    def postprocess_load_back(
        self,
        chain: list[int],
        ancestor_node_idx: int,
        device_values: Optional[torch.Tensor] = None,
    ) -> None:
        """Commit a load-back: write `device_values` onto `chain` + hand off the
        device lock, or release prepare's locks when `device_values` is None."""
        self._rust_radix.postprocess_load_back(chain, ancestor_node_idx, device_values)

    def finish_load_back(self, chain: list[int], loaded_node_idx: int) -> None:
        """Release locks held until the H2D load-back ack."""
        self._rust_radix.finish_load_back(chain, loaded_node_idx)

    def release_aux_host_locks(
        self, swa_chain: list[int], mamba_chain: list[int]
    ) -> None:
        self._rust_radix.release_aux_host_locks(swa_chain, mamba_chain)

    def init_load_back(self, params: InitLoadBackParams) -> tuple[torch.Tensor, int]:
        """If needed, restore host-backed prefix to device, up to `best_match_node`."""
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            CacheTransferPhase,
        )

        if self.ongoing_write_through:
            # Host values are published to the radix tree as soon as the D2H
            # write is queued. Do not use those host slots as H2D sources until
            # the write stream has actually finished, otherwise a valid Full
            # hit can restore stale auxiliary state.
            self.writing_check(write_back=True)

        node_idx = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        plan = self.prepare_load_back(node_idx)
        old_device_node = (
            req.last_node
            if req is not None and req.last_node is not None
            else plan.ancestor_node_idx
        )
        aux_plan = self.prepare_aux_load_back(node_idx)
        host_indices = self._as_host_indices(plan.host_indices)
        py_ct = self._py_component_type()
        comp_xfers: dict[Any, list[Any]] = {}
        swa_host_indices = self._as_host_indices(aux_plan.swa_host_indices)
        if len(swa_host_indices) > 0:
            comp_xfers[py_ct.SWA] = [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=swa_host_indices,
                    nodes_to_load=list(aux_plan.swa_chain),
                )
            ]
        mamba_host_indices = self._as_host_indices(aux_plan.mamba_host_indices)
        if len(mamba_host_indices) > 0:
            mamba_xfers = [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=mamba_host_indices,
                    nodes_to_load=list(aux_plan.mamba_chain),
                )
            ]
            if req is not None:
                if req.mamba_pool_idx is None:
                    dst = self.req_to_token_pool.mamba_allocator.alloc(1)
                    if dst is None:
                        self.evict(EvictParams(num_tokens=0, mamba_num=1))
                        dst = self.req_to_token_pool.mamba_allocator.alloc(1)
                    if dst is None:
                        self.postprocess_load_back(
                            plan.chain, plan.ancestor_node_idx, None
                        )
                        self.release_aux_host_locks(
                            list(aux_plan.swa_chain), list(aux_plan.mamba_chain)
                        )
                        return self._empty_indices, plan.ancestor_node_idx
                    req.mamba_pool_idx = dst[0]
                req.mamba_cow_src_index = None
                req.mamba_needs_clear = False
                mamba_xfers.append(
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=mamba_host_indices,
                        device_indices=req.mamba_pool_idx.unsqueeze(0),
                    )
                )
            comp_xfers[py_ct.MAMBA] = mamba_xfers
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        extra_pools = [x for xfers in comp_xfers.values() for x in xfers]
        extra_pools.extend(
            self._build_sidecar_transfers(
                CacheTransferPhase.LOAD_BACK,
                kv_xfer,
                comp_xfers,
            )
        )
        logger.info(
            "init_load_back: node=%s chain=%s host_tokens=%s swa_host=%s mamba_host=%s mem_quota=%s",
            node_idx,
            len(plan.chain),
            len(host_indices),
            len(swa_host_indices),
            len(mamba_host_indices),
            mem_quota,
        )
        # Skip tiny loads / those over mem_quota.
        if (len(host_indices) < self.load_back_threshold and not extra_pools) or (
            mem_quota is not None and len(host_indices) > mem_quota
        ):
            self.postprocess_load_back(plan.chain, plan.ancestor_node_idx, None)
            self.release_aux_host_locks(
                list(aux_plan.swa_chain), list(aux_plan.mamba_chain)
            )
            return self._empty_indices, plan.ancestor_node_idx
        # Loadback with retry
        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=node_idx,
            extra_pools=extra_pools or None,
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=node_idx,
                extra_pools=extra_pools or None,
            )
        if device_indices is None:
            self.postprocess_load_back(plan.chain, plan.ancestor_node_idx, None)
            self.release_aux_host_locks(
                list(aux_plan.swa_chain), list(aux_plan.mamba_chain)
            )
            return self._empty_indices, plan.ancestor_node_idx
        swa_device_values = None
        if py_ct.SWA in comp_xfers:
            swa_device_values = comp_xfers[py_ct.SWA][0].device_indices
        mamba_device_values = None
        if py_ct.MAMBA in comp_xfers:
            mamba_device_values = comp_xfers[py_ct.MAMBA][0].device_indices
        self.postprocess_load_back(plan.chain, plan.ancestor_node_idx, device_indices)
        self._rust_radix.postprocess_aux_load_back(
            list(aux_plan.swa_chain),
            list(aux_plan.mamba_chain),
            swa_device_values,
            mamba_device_values,
        )
        if (
            swa_device_values is not None
            and self.token_to_kv_pool_allocator is not None
            and hasattr(self.token_to_kv_pool_allocator, "set_full_to_swa_mapping")
        ):
            mapping = getattr(
                self.token_to_kv_pool_allocator, "full_to_swa_index_mapping", None
            )
            mapping_device = mapping.device if mapping is not None else self.device
            offset = 0
            for node in aux_plan.swa_chain:
                full_value = self._rust_radix.get_full_device_value(node)
                if full_value is None:
                    raise RadixCacheRuntimePyError(
                        "RustUnifiedRadixCache: SWA loadback node has no FULL anchor "
                        f"(node={node})"
                    )
                n_tokens = len(full_value)
                swa_chunk = swa_device_values[offset : offset + n_tokens]
                self.token_to_kv_pool_allocator.set_full_to_swa_mapping(
                    full_value.to(device=mapping_device, dtype=torch.int64),
                    swa_chunk.to(device=mapping_device, dtype=torch.int64),
                )
                offset += n_tokens
            if offset != len(swa_device_values):
                raise RadixCacheRuntimePyError(
                    "RustUnifiedRadixCache: SWA loadback mapping length mismatch "
                    f"(mapped={offset}, total={len(swa_device_values)})"
                )
        prefix_delta = self._rust_radix.collect_full_device_values_between(
            node_idx, old_device_node
        )
        handoff_locked = bool(plan.chain)
        logger.info(
            "load_back: queued node=%s chain=%s tokens=%s swa=%s mamba=%s",
            node_idx,
            len(plan.chain),
            len(device_indices),
            0 if swa_device_values is None else len(swa_device_values),
            0 if mamba_device_values is None else len(mamba_device_values),
        )
        # Save the ongoing loadback for lock ref reverts.
        self.ongoing_load_back[node_idx] = (
            list(plan.chain),
            node_idx,
            list(aux_plan.swa_chain),
            list(aux_plan.mamba_chain),
            handoff_locked,
        )
        return prefix_delta, node_idx

    def loading_check(self, wait: bool = False) -> None:
        """Release the device lock handed off to each loaded prefix once its
        host->device copy has completed."""
        if not self.ongoing_load_back:
            return
        finish_count = 0
        if wait:
            finish_count = len(self.cache_controller.ack_load_queue)
        elif self.pp_rank == 0:
            for ack in self.cache_controller.ack_load_queue:
                if not ack.finish_event.query():
                    break
                finish_count += 1
        finish_count = self._hicache_min_ready(finish_count)
        while finish_count > 0:
            ack = self.cache_controller.ack_load_queue.pop(0)
            ack.finish_event.synchronize()
            for node_id in ack.node_ids:
                entry = self.ongoing_load_back.pop(node_id, None)
                if entry is not None:
                    chain, end_node, swa_chain, mamba_chain, handoff_locked = entry
                    self.release_aux_host_locks(swa_chain, mamba_chain)
                    if handoff_locked:
                        self.finish_load_back(chain, end_node)
            finish_count -= 1

    def ready_to_load_host_cache(self) -> int:
        """Kick off the queued host->device loads; return the consumer index the
        scheduler tracks (-1 when the load queue is empty)."""
        consumer_index = self.cache_controller.start_loading()
        if consumer_index != -1 and self.supports_mamba():
            # Mamba COW is collected/executed before the layer-wise pool wait in
            # the forward path. Keep Rust's published device Mamba state from
            # becoming a COW source until the H2D load stream has materialized
            # it.
            self.loading_check(wait=True)
        return consumer_index

    def check_hicache_events(self):
        """Revert locks for already finished loadback and backup."""
        if self.cache_controller is None:
            return
        self.writing_check()
        self.loading_check()


def install_rust_radix_cache() -> None:
    """Register Rust Unified Radix Cache.

    The native extension is loaded by the factory, not during registration, so
    importing the default registry path remains safe before the extension is
    built.
    """
    from sglang.srt.mem_cache.registry import (
        get_radix_cache_factory,
        register_radix_cache_backend,
    )

    if get_radix_cache_factory("rust_unified") is not None:
        return

    def factory(ctx):
        _load_native_symbols()
        if ctx.server_args.enable_lmcache:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: LMCache is not supported"
            )
        return RustUnifiedRadixCache(ctx.params)

    register_radix_cache_backend("rust_unified", factory)
