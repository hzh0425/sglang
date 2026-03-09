import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Set

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.host_pool_base import HostPoolBase
    from sglang.srt.mem_cache.transfer_view import TransferView

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


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    def _set_registered_host_pool(self, host_pool: "HostPoolBase"):
        self.mem_pool_host = host_pool
        on_registered = getattr(self, "_on_host_pool_registered", None)
        if callable(on_registered):
            on_registered(host_pool)

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self._set_registered_host_pool(mem_pool_host)

    def register_host_pool(self, host_pool: "HostPoolBase"):
        """Register a host pool (can be any HostPoolBase implementation, not just HostKVCache)."""
        self._set_registered_host_pool(host_pool)

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

    # ==================== V2 Interface (TransferView-based) ====================

    def supports_transfer_view(self) -> bool:
        """
        Check if this storage backend supports TransferView-based operations.
        """
        return False

    def batch_get_v2(
        self,
        keys: List[str],
        transfer_view: "TransferView",
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys into TransferView-specified locations.

        This is the new interface that works with any HostPoolBase implementation.
        The TransferView describes where to write the data in host memory.

        Args:
            keys: List of hash keys to retrieve.
            transfer_view: Describes the target memory layout for retrieved data.
            extra_info: Optional extra information for the operation.

        Returns:
            List of booleans indicating success for each key.

        Note:
            Default implementation falls back to v1 interface if the pool is HostKVCache.
            Override this method for backends that want to support arbitrary pool types.
        """
        # Default: fall back to v1 if available
        if hasattr(self, "mem_pool_host") and transfer_view.offsets is not None:
            return self.batch_get_v1(keys, transfer_view.offsets, extra_info)
        raise NotImplementedError(
            "batch_get_v2 is not implemented. "
            "Override supports_transfer_view() to return True and implement this method."
        )

    def batch_set_v2(
        self,
        keys: List[str],
        transfer_view: "TransferView",
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        """
        Store values from TransferView-specified locations for multiple keys.

        This is the new interface that works with any HostPoolBase implementation.
        The TransferView describes where to read the data from host memory.

        Args:
            keys: List of hash keys to store.
            transfer_view: Describes the source memory layout for data to store.
            extra_info: Optional extra information for the operation.

        Returns:
            List of booleans indicating success for each key.

        Note:
            Default implementation falls back to v1 interface if the pool is HostKVCache.
            Override this method for backends that want to support arbitrary pool types.
        """
        # Default: fall back to v1 if available
        if hasattr(self, "mem_pool_host") and transfer_view.offsets is not None:
            return self.batch_set_v1(keys, transfer_view.offsets, extra_info)
        raise NotImplementedError(
            "batch_set_v2 is not implemented. "
            "Override supports_transfer_view() to return True and implement this method."
        )

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

    def batch_exists_v2(
        self,
        keys: List[str],
        auxiliary_constraints: List[dict[str, Any]],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        raise NotImplementedError(
            "batch_exists_v2 must be implemented by each storage backend."
        )

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
        self.transfer_suffix = ""

        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def _on_host_pool_registered(self, host_pool: "HostPoolBase"):
        dummy_page = host_pool.get_dummy_flat_data_page()
        page_num_bytes = 0
        dummy_dtype = None
        if dummy_page is not None:
            page_num_bytes = int(dummy_page.contiguous().view(torch.uint8).numel())
            dummy_dtype = str(dummy_page.dtype)

        pool_descriptor = {
            "host_pool_type": type(host_pool).__name__,
            "page_num_bytes": page_num_bytes,
            "page_size": getattr(host_pool, "page_size", None),
            "layout": getattr(host_pool, "layout", None),
            "dtype": dummy_dtype,
        }
        if hasattr(host_pool, "entries"):
            pool_descriptor["entries"] = [
                {
                    "name": entry.name,
                    "host_pool_type": type(entry.host_pool).__name__,
                    "page_num_bytes": int(
                        entry.host_pool.get_dummy_flat_data_page()
                        .contiguous()
                        .view(torch.uint8)
                        .numel()
                    )
                    if entry.host_pool.get_dummy_flat_data_page() is not None
                    else 0,
                }
                for entry in host_pool.entries
            ]

        signature_json = json.dumps(pool_descriptor, sort_keys=True)
        signature_hash = hashlib.sha1(signature_json.encode("utf-8")).hexdigest()[:12]
        self.transfer_suffix = f"_tv2_{signature_hash}"

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix + self.transfer_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        if component_name is None or component_name in ("__default__", "kv"):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    def _get_component_path(self, key: str, component_name: Optional[str] = None) -> str:
        return os.path.join(
            self.file_path, f"{self._get_component_key(key, component_name)}.bin"
        )

    def _get_aux_component_names(self) -> list[str]:
        host_pool = getattr(self, "mem_pool_host", None)
        if host_pool is None or not hasattr(host_pool, "entries"):
            return []
        return [
            entry.name
            for entry in host_pool.entries
            if not getattr(entry, "is_primary_index_anchor", False)
        ]

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
            encoded_components.append({"name": name, "num_bytes": int(flat_bytes.numel())})
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
    def _decode_v2_payload(file_data: bytes) -> tuple[dict[str, bytes], list[str]] | None:
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
        self, keys: List[str], auxiliary_constraints: Optional[List[dict[str, Any]]] = None
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
        existing_files = self._collect_existing_component_keys(keys, auxiliary_constraints)

        def has_component(page_idx: int, component_name: str) -> bool:
            return (
                f"{self._get_component_key(keys[page_idx], component_name)}.bin"
                in existing_files
            )

        full_hit_page_num = 0
        while full_hit_page_num < len(keys):
            if f"{self._get_component_key(keys[full_hit_page_num])}.bin" not in existing_files:
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
                    if all(has_component(page_idx, name) for page_idx in range(start, prefix_len)):
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
                keys[min(final_hit_page_num, len(keys)) - 1] if final_hit_page_num > 0 else None,
            )

        return final_hit_page_num

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

    # ==================== V2 Interface (TransferView-based) ====================

    def supports_transfer_view(self) -> bool:
        """HiCacheFile supports TransferView-based operations."""
        return True

    def _validate_transfer_view(
        self, keys: List[str], transfer_view: "TransferView"
    ) -> Optional[str]:
        """Validate TransferView parameters. Returns error message if invalid, None if valid."""
        if not hasattr(self, "mem_pool_host"):
            return "No host pool registered"
        if transfer_view.offsets is None:
            return "No offsets specified in TransferView"
        page_size = transfer_view.page_size or 1
        if len(transfer_view.offsets) != len(keys) * page_size:
            return f"Offsets length mismatch: expected {len(keys) * page_size}, got {len(transfer_view.offsets)}"
        return None

    def batch_get_v2(
        self,
        keys: List[str],
        transfer_view: "TransferView",
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        """Retrieve values for multiple keys into host pool via TransferView."""
        error = self._validate_transfer_view(keys, transfer_view)
        if error:
            logger.error(f"batch_get_v2 validation failed: {error}")
            return [False] * len(keys)

        offsets = transfer_view.offsets
        page_size = transfer_view.page_size or 1
        dummy_page = self.mem_pool_host.get_dummy_flat_data_page()
        expected_bytes = dummy_page.numel() * dummy_page.element_size()
        page_entry_names = None
        if extra_info is not None:
            if extra_info.extra_info is None:
                extra_info.extra_info = {}
            page_entry_names = extra_info.extra_info.setdefault("page_entry_names", [])

        results = []
        for i, key in enumerate(keys):
            tensor_path = self._get_component_path(key)

            if not os.path.exists(tensor_path):
                results.append(False)
                continue

            try:
                page_start_offset = offsets[i * page_size].item()

                with open(tensor_path, "rb", buffering=0) as f:
                    file_data = f.read()

                decoded = self._decode_v2_payload(file_data)
                merged_components: dict[str, torch.Tensor] = {}
                component_names: list[str] = []
                if decoded is None:
                    if len(file_data) != expected_bytes:
                        logger.warning(
                            f"File size mismatch for {key}: expected {expected_bytes}, got {len(file_data)}"
                        )
                        results.append(False)
                        continue
                    merged_components["__default__"] = torch.frombuffer(
                        file_data, dtype=dummy_page.dtype, count=dummy_page.numel()
                    ).reshape(dummy_page.shape)
                    if page_entry_names is not None:
                        component_names.append("__default__")
                else:
                    components_bytes, component_names = decoded
                    merged_components = {
                        name: torch.frombuffer(component_bytes, dtype=torch.uint8).clone()
                        for name, component_bytes in components_bytes.items()
                    }
                for component_name in self._get_aux_component_names():
                    aux_path = self._get_component_path(key, component_name)
                    if not os.path.exists(aux_path):
                        continue
                    with open(aux_path, "rb", buffering=0) as f:
                        aux_data = f.read()
                    aux_decoded = self._decode_v2_payload(aux_data)
                    if aux_decoded is None:
                        raise ValueError(
                            f"Auxiliary payload for {key}.{component_name} is not v2 encoded."
                        )
                    aux_components_bytes, aux_component_names = aux_decoded
                    for name, component_bytes in aux_components_bytes.items():
                        merged_components[name] = torch.frombuffer(
                            component_bytes, dtype=torch.uint8
                        ).clone()
                    component_names.extend(aux_component_names)
                if "mamba" in component_names:
                    logger.info(
                        "HiCache mamba prefetch restored for key %s: components=%s",
                        key,
                        ",".join(component_names),
                    )

                if hasattr(self.mem_pool_host, "set_from_page_components"):
                    self.mem_pool_host.set_from_page_components(
                        page_start_offset, merged_components
                    )
                else:
                    flat_page = torch.cat(list(merged_components.values()))
                    self.mem_pool_host.set_from_flat_data_page(
                        page_start_offset, flat_page
                    )
                if page_entry_names is not None:
                    page_entry_names.append(component_names or ["__default__"])
                results.append(True)

            except Exception as e:
                logger.error(f"Failed to read key {key}: {e}")
                results.append(False)

        return results

    def batch_set_v2(
        self,
        keys: List[str],
        transfer_view: "TransferView",
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        """Store values from host pool to storage via TransferView."""
        error = self._validate_transfer_view(keys, transfer_view)
        if error:
            logger.error(f"batch_set_v2 validation failed: {error}")
            return [False] * len(keys)

        offsets = transfer_view.offsets
        page_size = transfer_view.page_size or 1

        results = []
        for i, key in enumerate(keys):
            try:
                page_start_offset = offsets[i * page_size].item()
                components = None
                if hasattr(self.mem_pool_host, "get_page_components"):
                    components = self.mem_pool_host.get_page_components(page_start_offset)
                if components is None:
                    data_page = self.mem_pool_host.get_data_page(
                        page_start_offset, flat=True
                    )
                    components = {"__default__": data_page}
                primary_name = self._get_primary_component_name(components)
                payloads = {
                    primary_name: self._encode_v2_payload(
                        {primary_name: components[primary_name]}
                    )
                }
                for component_name, component_value in components.items():
                    if component_name == primary_name:
                        continue
                    payloads[component_name] = self._encode_v2_payload(
                        {component_name: component_value}
                    )

                for component_name, encoded_payload in payloads.items():
                    component_path = self._get_component_path(
                        key, None if component_name == primary_name else component_name
                    )
                    if os.path.exists(component_path):
                        with open(component_path, "rb", buffering=0) as f:
                            existing_data = f.read()
                        if existing_data == encoded_payload:
                            continue
                        if len(existing_data) == len(encoded_payload):
                            logger.debug(
                                "Reusing existing storage entry for %s without rewrite.",
                                key
                                if component_name == primary_name
                                else f"{key}.{component_name}",
                            )
                            continue
                        logger.warning(
                            f"Overwriting stale storage entry for "
                            f"{key if component_name == primary_name else f'{key}.{component_name}'}: "
                            f"expected {len(encoded_payload)} bytes, got {len(existing_data)}"
                        )
                    with open(component_path, "wb", buffering=0) as f:
                        f.write(encoded_payload)
                    if component_name != primary_name:
                        logger.info(
                            "HiCacheFile wrote auxiliary component %s for key %s",
                            component_name,
                            key,
                        )

                for component_name in self._get_aux_component_names():
                    if component_name in payloads:
                        continue
                    component_path = self._get_component_path(key, component_name)
                    if os.path.exists(component_path):
                        os.remove(component_path)
                results.append(True)

            except Exception as e:
                logger.error(f"Failed to store key {key}: {e}")
                results.append(False)

        return results
