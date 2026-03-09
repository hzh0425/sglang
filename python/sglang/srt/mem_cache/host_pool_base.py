# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
HostPoolBase: Abstract base class for host memory pools.

This module provides the base class for any memory pool that can be offloaded
to host memory, enabling generic pool support beyond KV cache pools.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.transfer_view import TransferView


class HostPoolBase(ABC):
    """
    Abstract base class for host memory pools that support offloading.

    This class defines the interface for any memory pool that can:
    - Allocate and free host memory
    - Transfer data to/from device memory
    - Generate TransferView for storage backend operations

    Implementations include:
    - HostKVCache: For KV cache pools (MHATokenToKVPoolHost, MLATokenToKVPoolHost)
    - Future: HostEmbeddingPool, HostActivationPool, etc.

    The abstraction enables:
    1. Storage backends to work with arbitrary pool types via TransferView
    2. Cache controller to manage multiple pool types uniformly
    3. Zero-copy transfers for RDMA-capable storage backends
    """

    @abstractmethod
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """
        Allocate host memory slots.

        Args:
            need_size: Number of slots to allocate.

        Returns:
            Tensor of allocated indices, or None if allocation failed.
        """
        pass

    @abstractmethod
    def free(self, indices: torch.Tensor) -> int:
        """
        Free host memory slots.

        Args:
            indices: Tensor of indices to free.

        Returns:
            Number of slots freed.
        """
        pass

    @abstractmethod
    def get_transfer_view(
        self, indices: torch.Tensor, transfer_ctx: Optional[dict] = None
    ) -> "TransferView":
        """
        Generate a TransferView for the given indices.

        The TransferView describes the memory layout of the specified slots,
        enabling storage backends to perform zero-copy operations.

        Args:
            indices: Tensor of indices to create a view for.

        Returns:
            TransferView describing the memory layout.
        """
        del transfer_ctx
        pass

    @abstractmethod
    def load_from_device(
        self,
        device_pool: Any,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        **kwargs,
    ) -> None:
        """
        Load data from device pool to host pool.

        Args:
            device_pool: The device memory pool to load from.
            host_indices: Indices in the host pool to load into.
            device_indices: Indices in the device pool to load from.
            **kwargs: Additional backend-specific parameters (e.g., io_backend, layer_id).
        """
        pass

    @abstractmethod
    def backup_to_device(
        self,
        device_pool: Any,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        **kwargs,
    ) -> None:
        """
        Backup data from host pool to device pool.

        Args:
            device_pool: The device memory pool to backup to.
            host_indices: Indices in the host pool to backup from.
            device_indices: Indices in the device pool to backup into.
            **kwargs: Additional backend-specific parameters (e.g., io_backend).
        """
        pass

    @abstractmethod
    def load_to_device_per_layer(
        self,
        device_pool: Any,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        layer_id: int,
        io_backend: str,
        transfer_ctx: Optional[dict] = None,
    ) -> None:
        """Load one logical model layer from host to device."""
        del transfer_ctx
        pass

    @abstractmethod
    def backup_from_device_all_layer(
        self,
        device_pool: Any,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
        transfer_ctx: Optional[dict] = None,
    ) -> None:
        """Back up all logical model layers from device to host."""
        del transfer_ctx
        pass

    def get_data_page(
        self,
        index: int,
        flat: bool = True,
        transfer_ctx: Optional[dict] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get a single page of data from the pool.

        This is a legacy compatibility method. New code should use get_transfer_view().

        Args:
            index: Starting index of the page.
            flat: If True, return flattened data.

        Returns:
            Tensor containing the page data, or None if not supported.
        """
        del transfer_ctx
        return None

    def set_from_flat_data_page(
        self,
        index: int,
        data_page: torch.Tensor,
        transfer_ctx: Optional[dict] = None,
    ) -> None:
        """
        Set a page of data from a flat tensor.

        This is a legacy compatibility method. New code should use get_transfer_view().

        Args:
            index: Starting index to write to.
            data_page: Flat tensor containing the page data.
        """
        del transfer_ctx
        raise NotImplementedError(
            "set_from_flat_data_page is not implemented. "
            "Use load_from_device with TransferView instead."
        )

    def get_dummy_flat_data_page(
        self, transfer_ctx: Optional[dict] = None
    ) -> Optional[torch.Tensor]:
        """
        Get a dummy page for prefetching initialization.

        This is a legacy compatibility method.

        Returns:
            Dummy tensor for prefetching, or None if not supported.
        """
        del transfer_ctx
        return None

    def get_page_components(
        self, index: int, transfer_ctx: Optional[dict] = None
    ) -> Optional[dict[str, torch.Tensor]]:
        """
        Get named page components for storage backends that support variable layouts.

        Returns:
            Mapping from component name to flat uint8 tensor, or None if unsupported.
        """
        del transfer_ctx
        return None

    def set_from_page_components(
        self,
        index: int,
        components: dict[str, torch.Tensor],
        transfer_ctx: Optional[dict] = None,
    ) -> None:
        """
        Restore named page components written by a storage backend.
        """
        default_page = components.get("__default__")
        if default_page is None:
            raise NotImplementedError(
                "set_from_page_components is not implemented for this host pool."
            )
        self.set_from_flat_data_page(index, default_page, transfer_ctx=transfer_ctx)

    def clear(self) -> None:
        """
        Clear the pool, freeing all allocations.

        Default implementation does nothing. Override if needed.
        """
        pass

    def get_size_per_token(self) -> int:
        """
        Get the size (in bytes) per token.

        Default implementation returns element_size_bytes.
        Override for pools with different per-token sizing.

        Returns:
            Size in bytes per token.
        """
        return self.dtype.itemsize if self.dtype else 0


class HostPoolRegistry:
    """
    Registry for host pool types.

    Enables dynamic discovery and creation of host pools based on pool type names.
    """

    _registry: dict = {}

    @classmethod
    def register(cls, name: str, pool_class: type) -> None:
        """Register a host pool class."""
        cls._registry[name] = pool_class

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a registered host pool class by name."""
        return cls._registry.get(name)

    @classmethod
    def list_registered(cls) -> list:
        """List all registered pool names."""
        return list(cls._registry.keys())
