from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState, PrefetchState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
        )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        self.ongoing_prefetch = {}
        logger.info("Enable offload kv cache for decode side")

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
            self.offloaded_state[req.rid] = state
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Early free prefill-offloaded GPU memory
        if state.prefill_len > 0 and state.inc_len == 0:
            self.token_to_kv_pool_allocator.free(token_indices[: state.prefill_len])

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
        )
        state.inc_len += incremental_aligned_len
        return True

    def check_offload_progress(self):
        """Check offload (D→H→S) and prefetch (S→H→D) progress.

        Prefetch phase transitions are TP-synced via a single merged all_reduce
        to prevent transfer_queue length divergence across ranks.

        Returns:
            List of request rids whose prefetch completed this tick.
        """
        cc = self.cache_controller

        # --- Collect local prefetch phase status ---
        prefetch_rids = list(self.ongoing_prefetch.keys())
        n_pf = len(prefetch_rids)
        s2h_local = []
        h2d_local = []
        for rid in prefetch_rids:
            state = self.ongoing_prefetch[rid]
            if state.phase == "s2h":
                op = state.prefetch_op
                s2h_local.append(
                    1
                    if (op.is_terminated() or op.completed_tokens >= state.cached_tokens)
                    else 0
                )
                h2d_local.append(0)
            elif state.phase == "h2d":
                s2h_local.append(0)
                h2d_local.append(
                    1
                    if (
                        state.load_ack is not None
                        and state.load_ack.finish_event.query()
                    )
                    else 0
                )
            else:
                s2h_local.append(0)
                h2d_local.append(0)

        # --- Single all_reduce: [n_write, n_backup, s2h_0..s2h_n, h2d_0..h2d_n] ---
        sync_data = torch.tensor(
            [len(cc.ack_write_queue), cc.ack_backup_queue.qsize()]
            + s2h_local
            + h2d_local,
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                sync_data, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write = int(sync_data[0])
        n_backup = int(sync_data[1])
        s2h_synced = sync_data[2 : 2 + n_pf]
        h2d_synced = sync_data[2 + n_pf : 2 + 2 * n_pf]

        # --- Process offload / backup ---
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

        # --- Advance prefetch phases (TP-synced) ---
        completed = []
        for idx, rid in enumerate(prefetch_rids):
            state = self.ongoing_prefetch[rid]
            if state.phase == "s2h" and s2h_synced[idx].item() == 1:
                # All ranks agree S→H is done – start H→D
                self.cache_controller.load_to_indices(
                    host_indices=state.host_indices,
                    device_indices=state.device_indices,
                )
                self.cache_controller.start_loading()
                state.load_ack = self.cache_controller.ack_load_queue[-1]
                state.phase = "h2d"
                logger.info(
                    f"[HiCache-Inc] D-side prefetch s2h->h2d: req={rid}, "
                    f"completed_tokens={state.prefetch_op.completed_tokens}/{state.cached_tokens}"
                )
            elif state.phase == "h2d" and h2d_synced[idx].item() == 1:
                # All ranks agree H→D is done – mark complete
                state.phase = "done"
                completed.append(rid)
                self.decode_host_mem_pool.free(state.host_indices)
                del self.ongoing_prefetch[rid]
                logger.info(
                    f"[HiCache-Inc] D-side prefetch DONE: req={rid}, "
                    f"cached_tokens={state.cached_tokens}"
                )

        return completed

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                ) = self.ongoing_offload.pop(ack_id)

                if req.finished():
                    self._release_finished_req(req, start)
                else:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, start:end
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)

                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                last_hash = self._trigger_backup(
                    req, host_indices, incremental_tokens, start_time, prior_hash
                )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        kv_committed_len = req.pop_committed_kv_cache()
        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (NSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = req.pop_overallocated_kv_cache()
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        self.req_to_token_pool.free(req)
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time = self.ongoing_backup.pop(ack_id)

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self, req, host_indices, incremental_tokens, start_time, prior_hash
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def query_storage_cached_tokens(self, tokens) -> int:
        """Query storage backend to find how many consecutive prefix tokens are cached (page-aligned)."""
        if not self.cache_controller.enable_storage:
            logger.info("[HiCache-Inc] D-side query: storage NOT enabled, skip")
            return 0
        page_aligned_len = (len(tokens) // self.page_size) * self.page_size
        if page_aligned_len == 0:
            return 0

        # Diagnostic: compute first page hash manually and check file existence
        from sglang.srt.mem_cache.hicache_storage import get_hash_str

        first_page_tokens = tokens[: self.page_size]
        first_hash = get_hash_str(list(first_page_tokens), prior_hash=None)
        sb = self.cache_controller.storage_backend
        suffix = getattr(sb, "config_suffix", "N/A")
        storage_dir = getattr(sb, "file_path", "N/A")
        import os

        expected_file = os.path.join(storage_dir, f"{first_hash}{suffix}.bin")
        file_exists = os.path.exists(expected_file)
        logger.info(
            f"[HiCache-Inc] DIAG: first_page_tokens[:5]={list(first_page_tokens[:5])}, "
            f"first_hash={first_hash[:16]}..., suffix={suffix}, "
            f"file_exists={file_exists}, path={expected_file}"
        )
        # Also list a few actual files for comparison
        try:
            actual_files = os.listdir(storage_dir)[:3]
            logger.info(
                f"[HiCache-Inc] DIAG: sample files in storage: {actual_files}"
            )
        except Exception as e:
            logger.info(f"[HiCache-Inc] DIAG: cannot list storage dir: {e}")

        page_hashes, hit_count = self.cache_controller._storage_hit_query(
            type(
                "_Dummy",
                (),
                {
                    "last_hash": None,
                    "token_ids": tokens[:page_aligned_len],
                    "prefix_keys": None,
                },
            )()
        )
        logger.info(
            f"[HiCache-Inc] D-side storage query: {hit_count}/{page_aligned_len} tokens cached "
            f"({hit_count // self.page_size} pages), total_pages={page_aligned_len // self.page_size}"
        )
        return hit_count

    def start_prefetch(
        self, req, cached_tokens: int, device_indices: torch.Tensor
    ) -> bool:
        """Start async prefetch for [0:cached_tokens] from storage to device."""
        host_indices = self.decode_host_mem_pool.alloc(cached_tokens)
        if host_indices is None:
            logger.warning(
                f"Not enough host memory for prefetch of request {req.rid}"
            )
            return False
        tokens = req.origin_input_ids[:cached_tokens]
        prefetch_op = self.cache_controller.prefetch(
            request_id=req.rid,
            host_indices=host_indices,
            new_input_tokens=tokens,
            last_hash=None,
        )
        self.ongoing_prefetch[req.rid] = PrefetchState(
            prefetch_op=prefetch_op,
            host_indices=host_indices,
            device_indices=device_indices[:cached_tokens].clone().to(torch.int64),
            cached_tokens=cached_tokens,
        )
        logger.info(
            f"[HiCache-Inc] D-side prefetch started: req={req.rid}, "
            f"cached_tokens={cached_tokens}, host_indices={len(host_indices)}"
        )
        return True

    def is_prefetch_done(self, rid: str) -> bool:
        """Check if prefetch for a given request is complete."""
        return rid not in self.ongoing_prefetch

    def cleanup_prefetch(self, rid: str):
        """Clean up prefetch state for a failed/aborted request. Releases host memory."""
        if rid in self.ongoing_prefetch:
            state = self.ongoing_prefetch.pop(rid)
            self.decode_host_mem_pool.free(state.host_indices)
            logger.debug(f"Cleaned up prefetch state for request {rid}")

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        if req.req_pool_idx == -1:
            return
        # Clean up any pending prefetch state
        if req.rid in self.ongoing_prefetch:
            state = self.ongoing_prefetch.pop(req.rid)
            self.decode_host_mem_pool.free(state.host_indices)
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # If no incremental offload ever happened, the prefill-aligned part was never freed.
        # Free the prefill portion on request finish to avoid leaks.
        if prefill_len > 0 and inc_len == 0:
            token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            self.token_to_kv_pool_allocator.free(token_indices[:prefill_len])
            logger.info(
                f"Finalize release: freed prefill-aligned KV for req {req.rid}, len:{prefill_len}"
            )
        start_offset = prefill_len + inc_len
        self._release_finished_req(req, start_offset)
