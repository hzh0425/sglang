import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Set

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

FILE_V2_MAGIC = b"HCF2"


def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram mode: hash both elements to uniquely identify the bigram
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # Regular mode: single integer token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert SHA256 hex string to signed 64-bit integer for events.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    """
    # Take first 16 hex chars to get 64-bit value
    uint64_val = int(hash_str[:16], 16)
    # Convert to signed int64 range [-2^63, 2^63-1]
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    is_mla_model: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = (None,)
    extra_info: Optional[dict] = None


@dataclass
class StorageTransfer:
    pool_name: str
    keys: List[str]
    indices: torch.Tensor


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if not hasattr(self, "registered_pools") or self.registered_pools is None:
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool

    def batch_exists_v2(
        self,
        keys: List[str],
        auxiliary_constraints: List[dict[str, Any]],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        raise NotImplementedError(
            "batch_exists_v2 must be implemented by each storage backend."
        )

    def batch_get_v2(
        self,
        transfers: List[StorageTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[StorageTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        raise NotImplementedError()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"
        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        if component_name is None or component_name in ("__default__", "kv"):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    def _get_component_path(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        return os.path.join(
            self.file_path, f"{self._get_component_key(key, component_name)}.bin"
        )

    def _get_aux_component_names(self) -> list[str]:
        registered_pools = getattr(self, "registered_pools", {})
        return [name for name in registered_pools if name != "kv"]

    @staticmethod
    def _get_primary_component_name(components: dict[str, torch.Tensor]) -> str:
        if "__default__" in components:
            return "__default__"
        if "kv" in components:
            return "kv"
        return next(iter(components))

    @staticmethod
    def _encode_v2_payload(components: dict[str, torch.Tensor]) -> bytes:
        encoded_components = []
        payload_parts = []
        for name, tensor in components.items():
            flat_bytes = tensor.contiguous().view(torch.uint8).reshape(-1).cpu()
            encoded_components.append(
                {"name": name, "num_bytes": int(flat_bytes.numel())}
            )
            payload_parts.append(flat_bytes.numpy().tobytes())
        header = json.dumps({"components": encoded_components}, sort_keys=True).encode(
            "utf-8"
        )
        return (
            FILE_V2_MAGIC
            + len(header).to_bytes(8, byteorder="little", signed=False)
            + header
            + b"".join(payload_parts)
        )

    @staticmethod
    def _decode_v2_payload(
        file_data: bytes,
    ) -> tuple[dict[str, bytes], list[str]] | None:
        if not file_data.startswith(FILE_V2_MAGIC) or len(file_data) < 12:
            return None
        header_len = int.from_bytes(file_data[4:12], byteorder="little", signed=False)
        header_start = 12
        header_end = header_start + header_len
        if len(file_data) < header_end:
            raise ValueError("Corrupted HiCacheFile v2 header.")
        header = json.loads(file_data[header_start:header_end].decode("utf-8"))
        cursor = header_end
        components = {}
        names = []
        for entry in header.get("components", []):
            name = entry["name"]
            num_bytes = int(entry["num_bytes"])
            components[name] = file_data[cursor : cursor + num_bytes]
            names.append(name)
            cursor += num_bytes
        if cursor != len(file_data):
            raise ValueError("Corrupted HiCacheFile v2 payload size.")
        return components, names

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        auxiliary_constraints: Optional[List[dict[str, Any]]] = None,
    ) -> Set[str]:
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}
        for constraint in auxiliary_constraints or []:
            component_name = constraint["name"]
            for key in keys:
                target_files.add(f"{self._get_component_key(key, component_name)}.bin")

        existing_files = set()
        with os.scandir(self.file_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name in target_files:
                    existing_files.add(entry.name)
        return existing_files

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        existing_files = self._collect_existing_component_keys(keys)
        for i, key in enumerate(keys):
            if f"{self._get_component_key(key)}.bin" not in existing_files:
                return i
        return len(keys)

    def batch_exists_v2(
        self,
        keys: List[str],
        auxiliary_constraints: List[dict[str, Any]],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        existing_files = self._collect_existing_component_keys(
            keys, auxiliary_constraints
        )

        def has_component(page_idx: int, component_name: str) -> bool:
            return (
                f"{self._get_component_key(keys[page_idx], component_name)}.bin"
                in existing_files
            )

        full_hit_page_num = 0
        while full_hit_page_num < len(keys):
            if (
                f"{self._get_component_key(keys[full_hit_page_num])}.bin"
                not in existing_files
            ):
                break
            full_hit_page_num += 1

        final_hit_page_num = full_hit_page_num
        auxiliary_boundaries: dict[str, int] = {}
        for constraint in auxiliary_constraints:
            name = constraint["name"]
            policy = constraint.get("policy", "all_pages")
            boundary = 0

            if policy == "all_pages":
                while boundary < full_hit_page_num and has_component(boundary, name):
                    boundary += 1
            elif policy == "trailing_pages":
                trailing_pages = max(1, int(constraint.get("trailing_pages", 1)))
                for prefix_len in range(full_hit_page_num, 0, -1):
                    start = max(0, prefix_len - trailing_pages)
                    if all(
                        has_component(page_idx, name)
                        for page_idx in range(start, prefix_len)
                    ):
                        boundary = prefix_len
                        break
            else:
                raise ValueError(f"Unsupported auxiliary hit policy: {policy}")

            auxiliary_boundaries[name] = boundary
            final_hit_page_num = min(final_hit_page_num, boundary)
            if final_hit_page_num == 0:
                break

        if auxiliary_constraints:
            logger.info(
                "HiCacheFile batch_exists_v2: full_hit_pages=%s final_hit_pages=%s aux_boundaries=%s first_key=%s last_key=%s",
                full_hit_page_num,
                final_hit_page_num,
                auxiliary_boundaries,
                keys[0] if keys else None,
                (
                    keys[min(final_hit_page_num, len(keys)) - 1]
                    if final_hit_page_num > 0
                    else None
                ),
            )

        return final_hit_page_num

    def batch_get_v2(
        self,
        transfers: List[StorageTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        results: dict[str, List[bool]] = {}
        if extra_info is not None:
            extra_info.extra_info = extra_info.extra_info or {}
            loaded_names = extra_info.extra_info.setdefault("loaded_names", set())
            hit_page_count_by_name = extra_info.extra_info.setdefault(
                "hit_page_count_by_name", {}
            )
        else:
            loaded_names = None
            hit_page_count_by_name = None

        for transfer in transfers:
            host_pool = self.registered_pools[transfer.pool_name]
            page_size = getattr(host_pool, "page_size", 1) or 1
            if transfer.indices.numel() != len(transfer.keys) * page_size:
                logger.error(
                    "batch_get_v2 indices length mismatch for %s: expected %s, got %s",
                    transfer.pool_name,
                    len(transfer.keys) * page_size,
                    transfer.indices.numel(),
                )
                pool_results = [False] * len(transfer.keys)
            else:
                pool_results = []
                for i, key in enumerate(transfer.keys):
                    tensor_path = self._get_component_path(
                        key, None if transfer.pool_name == "kv" else transfer.pool_name
                    )
                    if not os.path.exists(tensor_path):
                        pool_results.append(False)
                        continue
                    try:
                        with open(tensor_path, "rb", buffering=0) as f:
                            file_data = f.read()
                        decoded = self._decode_v2_payload(file_data)
                        dummy_page = host_pool.get_dummy_flat_data_page()
                        if decoded is None:
                            expected_bytes = (
                                dummy_page.numel() * dummy_page.element_size()
                            )
                            if len(file_data) != expected_bytes:
                                logger.warning(
                                    "File size mismatch for %s: expected %s, got %s",
                                    (
                                        key
                                        if transfer.pool_name == "kv"
                                        else f"{key}.{transfer.pool_name}"
                                    ),
                                    expected_bytes,
                                    len(file_data),
                                )
                                pool_results.append(False)
                                continue
                            data_page = (
                                torch.frombuffer(
                                    file_data,
                                    dtype=dummy_page.dtype,
                                    count=dummy_page.numel(),
                                )
                                .clone()
                                .reshape(dummy_page.shape)
                            )
                        else:
                            if transfer.pool_name == "kv":
                                component_name = self._get_primary_component_name(
                                    decoded[0]
                                )
                                component_tensor = torch.frombuffer(
                                    decoded[0][component_name],
                                    dtype=dummy_page.dtype,
                                    count=dummy_page.numel(),
                                ).clone()
                                data_page = component_tensor.reshape(dummy_page.shape)
                            else:
                                component_name = transfer.pool_name
                                data_page = torch.frombuffer(
                                    decoded[0][component_name], dtype=torch.uint8
                                ).clone()
                        page_start_offset = transfer.indices[i * page_size].item()
                        host_pool.set_from_flat_data_page(page_start_offset, data_page)
                        pool_results.append(True)
                    except Exception as e:
                        logger.error(
                            "Failed to read key %s%s: %s",
                            key,
                            (
                                ""
                                if transfer.pool_name == "kv"
                                else f".{transfer.pool_name}"
                            ),
                            e,
                        )
                        pool_results.append(False)
            if loaded_names is not None and hit_page_count_by_name is not None:
                contiguous_hits = 0
                for ok in pool_results:
                    if not ok:
                        break
                    contiguous_hits += 1
                if contiguous_hits > 0:
                    loaded_names.add(transfer.pool_name)
                    hit_page_count_by_name[transfer.pool_name] = contiguous_hits
            results[transfer.pool_name] = pool_results
        return results

    def batch_set_v2(
        self,
        transfers: List[StorageTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        del extra_info
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.pool_name]
            page_size = getattr(host_pool, "page_size", 1) or 1
            if transfer.indices.numel() != len(transfer.keys) * page_size:
                logger.error(
                    "batch_set_v2 indices length mismatch for %s: expected %s, got %s",
                    transfer.pool_name,
                    len(transfer.keys) * page_size,
                    transfer.indices.numel(),
                )
                pool_results = [False] * len(transfer.keys)
            else:
                pool_results = []
                for i, key in enumerate(transfer.keys):
                    try:
                        page_start_offset = transfer.indices[i * page_size].item()
                        data_page = host_pool.get_data_page(
                            page_start_offset, flat=True
                        )
                        encoded_payload = self._encode_v2_payload(
                            {
                                (
                                    "__default__"
                                    if transfer.pool_name == "kv"
                                    else transfer.pool_name
                                ): data_page
                            }
                        )
                        component_path = self._get_component_path(
                            key,
                            None if transfer.pool_name == "kv" else transfer.pool_name,
                        )
                        if os.path.exists(component_path):
                            with open(component_path, "rb", buffering=0) as f:
                                existing_data = f.read()
                            if existing_data == encoded_payload:
                                pool_results.append(True)
                                continue
                            if len(existing_data) != len(encoded_payload):
                                logger.warning(
                                    "Overwriting stale storage entry for %s: expected %s bytes, got %s",
                                    (
                                        key
                                        if transfer.pool_name == "kv"
                                        else f"{key}.{transfer.pool_name}"
                                    ),
                                    len(encoded_payload),
                                    len(existing_data),
                                )
                        with open(component_path, "wb", buffering=0) as f:
                            f.write(encoded_payload)
                        if transfer.pool_name != "kv":
                            logger.info(
                                "HiCacheFile wrote auxiliary component %s for key %s",
                                transfer.pool_name,
                                key,
                            )
                        pool_results.append(True)
                    except Exception as e:
                        logger.error(
                            "Failed to store key %s%s: %s",
                            key,
                            (
                                ""
                                if transfer.pool_name == "kv"
                                else f".{transfer.pool_name}"
                            ),
                            e,
                        )
                        pool_results.append(False)
            results[transfer.pool_name] = pool_results
        return results

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
