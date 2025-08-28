import logging
import threading

import torch

from sglang import ServerArgs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args

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
        self.decode_cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=server_args.hicache_storage_backend_extra_config,
        )

        self.ongoing_backup = {}
        logger.info("Enable offload kv cache for decode side")

    def offload_kv_cache(self, req) -> None:
        """Offload a finished request's KV cache to storage."""
        if self.decode_cache_controller is None or self.decode_host_mem_pool is None:
            return

        if req.req_pool_idx == -1:
            return

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            logger.debug(
                f"Request {req.rid} has invalid token_indices: {token_indices}"
            )
            return

        # Allocate host indices and copy from device to host
        tokens = req.origin_input_ids + req.output_ids
        aligned_len = (len(tokens) // self.page_size) * self.page_size
        if aligned_len == 0:
            return

        token_indices = token_indices[:aligned_len]
        tokens = tokens[:aligned_len]

        host_indices = self.decode_host_mem_pool.alloc(aligned_len)
        if host_indices is None or len(host_indices) != aligned_len:
            if host_indices is not None:
                self.decode_host_mem_pool.free(host_indices)
            logger.error(f"Not enough host memory for request {req.rid}")
            return

        logger.info(f"Transferring KV cache for req {req.rid}: {aligned_len} tokens")

        device_pool = self.token_to_kv_pool_allocator.get_kvcache()
        new_host_indices, device_indices = self.decode_cache_controller.move_indices(
            host_indices, token_indices.long()
        )
        self.decode_host_mem_pool.backup_from_device_all_layer(
            device_pool, new_host_indices, device_indices, "direct"
        )
        torch.cuda.synchronize()

        # Generate page hashes and write to storage
        page_hashes = []
        last_hash = ""
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.decode_cache_controller.get_hash_str(
                page_tokens, last_hash
            )
            page_hashes.append(last_hash)

        ack_id = self.decode_cache_controller.write_storage(
            host_indices,
            tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = host_indices

    def check_backup_progress(self):
        queue_size = torch.tensor(
            self.decode_cache_controller.ack_backup_queue.qsize(), dtype=torch.int
        )
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to mem_pool cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        for _ in range(queue_size.item()):
            ack_id, _ = self.decode_cache_controller.ack_backup_queue.get()
            host_indices = self.ongoing_backup[ack_id]

            self.decode_host_mem_pool.free(host_indices)
            logger.info(
                f"Free host memory for request {ack_id}, len:{len(host_indices)}"
            )
            del self.ongoing_backup[ack_id]
