import concurrent.futures
import json
import logging
import os
from typing import List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.storage.hf3fs.client_hf3fs import Hf3fsClient
from sglang.srt.mem_cache.storage.hf3fs.metadata_client import MetadataClient

logger = logging.getLogger(__name__)

class AtomicCounter:
    def __init__(self, n: int):
        assert n > 0
        self.n = n
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            current = self._value
            self._value = (current + 1) % self.n
            return current

class HiCacheHF3FS(HiCacheStorage):
    default_env_var: str = "SGLANG_HICACHE_HF3FS_CONFIG_PATH"
    default_metadata_server_url: str = "http://localhost:8000"

    def __init__(
        self,
        rank: int,
        file_path: str,
        file_size: int,
        numjobs: int,
        bytes_per_page: int,
        entries: int,
        dtype: torch.dtype,
        metadata_server_url: str,
    ):
        self.rank = rank
        self.file_path = file_path
        self.file_size = file_size
        self.numjobs = numjobs
        self.bytes_per_page = bytes_per_page
        self.entries = entries
        self.dtype = dtype
        self.metadata_client = MetadataClient(metadata_server_url)

        self.numel = self.bytes_per_page // self.dtype.itemsize
        self.num_pages = self.file_size // self.bytes_per_page

        logger.info(
            f"[Rank {self.rank}] HiCacheHF3FS Client Initializing: "
            f"file_path={self.file_path}, "
            f"file_size={self.file_size/(2**30):.2f} GB, "
            f"num_pages={self.num_pages}"
        )

        self.ac = AtomicCounter(self.numjobs)
        self.clients = [
            Hf3fsClient(
                self.file_path, self.file_size, self.bytes_per_page, self.entries
            )
            for _ in range(numjobs)
        ]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.numjobs, thread_name_prefix=f"HiCacheHF3FS-Rank{self.rank}"
        )

        self.metadata_client.initialize(self.rank, self.num_pages)

    @staticmethod
    def from_env_config(
        rank: int, bytes_per_page: int, dtype: torch.dtype
    ) -> "HiCacheHF3FS":
        config_path = os.getenv(HiCacheHF3FS.default_env_var)
        if not config_path:
            return HiCacheHF3FS(
                rank=rank,
                file_path=f"/data/hicache.{rank}.bin",
                file_size=1 << 40,
                numjobs=16,
                bytes_per_page=bytes_per_page,
                entries=8,
                dtype=dtype,
                metadata_server_url=HiCacheHF3FS.default_metadata_server_url,
            )

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")

        required_keys = {
            "file_path_prefix",
            "file_size",
            "numjobs",
            "entries",
        }
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in config: {missing_keys}")

        return HiCacheHF3FS(
            rank=rank,
            file_path=f"{config['file_path_prefix']}.{rank}.bin",
            file_size=int(config["file_size"]),
            numjobs=int(config["numjobs"]),
            bytes_per_page=bytes_per_page,
            entries=int(config["entries"]),
            dtype=dtype,
            metadata_server_url=config.get("metadata_server_url", HiCacheHF3FS.default_metadata_server_url),
        )

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        return self.batch_get([key], [target_location] if target_location else None)[0]

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        page_indices = self.metadata_client.get_page_indices(self.rank, keys)
        
        batch_indices, file_offsets = [], []
        results = [None] * len(keys)

        for i, page_index in enumerate(page_indices):
            if page_index is not None:
                batch_indices.append(i)
                file_offsets.append(page_index * self.bytes_per_page)

        if not batch_indices:
            return results

        file_results = [
            torch.empty(self.numel, dtype=self.dtype) for _ in range(len(batch_indices))
        ]

        futures = [
            self.executor.submit(
                self.clients[self.ac.next()].batch_read,
                file_offsets[i : i + self.entries],
                file_results[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        read_results = [result for future in futures for result in future.result()]

        for batch_index, file_result, read_result in zip(
            batch_indices, file_results, read_results
        ):
            if read_result == self.bytes_per_page:
                results[batch_index] = file_result
            else:
                logging.error(f"[Rank {self.rank}] HiCacheHF3FS get {keys[batch_index]} failed")

        return results

    def set(self, key: str, value: torch.Tensor) -> bool:
        return self.batch_set([key], [value])[0]

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> List[bool]:
        indices = self.metadata_client.reserve_and_get_indices(self.rank, keys)
        
        batch_indices, file_offsets, file_values = [], [], []
        pages_to_release = []
        
        for i, (value, (is_written, index)) in enumerate(zip(values, indices)):
            if is_written:
                continue
            if index == -1:
                logging.warning(f"[Rank {self.rank}] No space for key {keys[i]}, skipping set.")
                return [False] * len(keys)        
            
            batch_indices.append(i)
            file_offsets.append(index * self.bytes_per_page)
            file_values.append(value.contiguous())

        futures = [
            self.executor.submit(
                self.clients[self.ac.next()].batch_write,
                file_offsets[i : i + self.entries],
                file_values[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        write_results = [
            result == self.bytes_per_page
            for future in futures
            for result in future.result()
        ]

        written_keys_to_confirm = []
        results = [item[0] for item in indices]
        for i, write_result in enumerate(write_results):
            original_index = batch_indices[i]
            key = keys[original_index]
            index = indices[original_index][1]
            if write_result:
                written_keys_to_confirm.append((key, index))
                results[original_index] = True
            else:
                logging.error(f"[Rank {self.rank}] HiCacheHF3FS set {key} failed")
                pages_to_release.append(index)
                results[original_index] = False
        
        if written_keys_to_confirm:
            self.metadata_client.confirm_write(self.rank, written_keys_to_confirm)
        if pages_to_release:
            self.metadata_client.release_pages(self.rank, pages_to_release)
            
        return results

    def delete(self, key: str) -> None:
        self.metadata_client.delete_keys(self.rank, [key])

    def exists(self, key: str) -> bool:
        return self.metadata_client.exists(self.rank, key)

    def clear(self) -> None:
        self.metadata_client.clear(self.rank)

    def close(self) -> None:
        for c in self.clients:
            c.close()
        self.executor.shutdown(wait=True)
