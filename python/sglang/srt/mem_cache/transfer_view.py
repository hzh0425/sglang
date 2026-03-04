# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
TransferView: Abstract view for data transfer operations.

This module provides a generic abstraction for describing data transfer operations
between different memory pools (device/host/storage), enabling support for arbitrary
pool types beyond KV cache pools.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class TransferView:
    """
    Abstract transfer view describing all necessary information for a data transfer.

    A TransferView can represent:
    - A contiguous region of a single buffer (ptr + offset + size)
    - Multiple buffers at the same indices (e.g., K and V buffers in KV cache)
    - Sparse indexed access (via indices tensor)
    - Direct tensor references (for non-pointer-based transfers)

    This abstraction enables storage backends to work with any pool type
    without knowing the specific memory layout or pool implementation.

    Attributes:
        pool_ptrs: List of memory pointers (as integers) for the underlying buffers.
                   Used for zero-copy DMA operations by RDMA-capable backends.
        offsets: Tensor of indices specifying which elements to transfer.
                 Can be host indices or device indices depending on context.
        element_size_bytes: Size of each element in bytes.
        tensors: Optional direct tensor references. Used when pointers are not
                 available or when backend prefers tensor-based access.
        element_shape: Shape of a single element (e.g., (num_heads, head_dim)).
        dtype: Data type of the elements.
        device: Device where the data resides ("cpu", "cuda", etc.).
        layout: Memory layout hint for optimization ("layer_first", "page_first", etc.).
        page_size: Number of elements per page (for page-based transfers).
        layer_id: Optional layer ID for multi-layer transfers (e.g., KV cache layers).
        num_layers: Total number of layers (for multi-layer pool like KV cache).
    """

    # Core transfer information
    pool_ptrs: List[int] = field(default_factory=list)
    offsets: Optional[torch.Tensor] = None
    element_size_bytes: int = 0

    # Alternative: direct tensor references
    tensors: Optional[List[torch.Tensor]] = None

    # Metadata for interpretation
    element_shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[str] = None
    layout: Optional[str] = None

    # Optional parameters for specific use cases
    page_size: Optional[int] = None
    layer_id: Optional[int] = None
    num_layers: Optional[int] = None

    def __post_init__(self):
        """Validate TransferView after initialization."""
        if not self.pool_ptrs and not self.tensors:
            # Allow empty TransferView for special cases (e.g., dummy operations)
            pass

        if self.offsets is not None and not isinstance(self.offsets, torch.Tensor):
            raise TypeError(f"offsets must be a torch.Tensor, got {type(self.offsets)}")

    @property
    def num_elements(self) -> int:
        """Number of elements to transfer."""
        if self.offsets is not None:
            return self.offsets.numel()
        elif self.tensors is not None:
            # Sum of first dimension across all tensors
            return sum(t.shape[0] for t in self.tensors)
        return 0

    @property
    def total_bytes(self) -> int:
        """Total bytes to transfer."""
        return self.num_elements * self.element_size_bytes

    def get_flat_tensor(self, tensor_idx: int = 0) -> Optional[torch.Tensor]:
        """
        Get a flattened view of the tensor at the given index.

        Args:
            tensor_idx: Index into the tensors list (default 0).

        Returns:
            Flattened tensor view, or None if tensors not available.
        """
        if self.tensors is None or tensor_idx >= len(self.tensors):
            return None
        return self.tensors[tensor_idx].view(-1)

    def get_element_at(self, index: int, tensor_idx: int = 0) -> Optional[torch.Tensor]:
        """
        Get a single element at the given index.

        Args:
            index: Element index within the transfer view.
            tensor_idx: Index into the tensors list.

        Returns:
            Tensor element, or None if not available.
        """
        if self.offsets is None or self.tensors is None:
            return None
        if tensor_idx >= len(self.tensors):
            return None

        actual_index = self.offsets[index].item()
        return self.tensors[tensor_idx][actual_index]

    def slice_by_page(self, page_idx: int, num_pages: int = 1) -> "TransferView":
        """
        Create a new TransferView for a subset of pages.

        Args:
            page_idx: Starting page index.
            num_pages: Number of pages to include.

        Returns:
            New TransferView for the specified page range.
        """
        if self.page_size is None or self.offsets is None:
            raise ValueError("page_size and offsets must be set for page slicing")

        start = page_idx * self.page_size
        end = min((page_idx + num_pages) * self.page_size, self.offsets.numel())

        new_offsets = self.offsets[start:end]

        return TransferView(
            pool_ptrs=self.pool_ptrs,
            offsets=new_offsets,
            element_size_bytes=self.element_size_bytes,
            tensors=self.tensors,
            element_shape=self.element_shape,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            page_size=self.page_size,
            layer_id=self.layer_id,
            num_layers=self.num_layers,
        )

    def is_compatible_with(self, other: "TransferView") -> bool:
        """
        Check if this TransferView is compatible with another for paired operations.

        Two TransferViews are compatible if they have the same:
        - Number of elements
        - Element size
        - Data type
        - Device

        Args:
            other: Another TransferView to compare with.

        Returns:
            True if compatible, False otherwise.
        """
        return (
            self.num_elements == other.num_elements
            and self.element_size_bytes == other.element_size_bytes
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def to_dict(self) -> dict:
        """Serialize TransferView to a dictionary for debugging/logging."""
        return {
            "num_elements": self.num_elements,
            "total_bytes": self.total_bytes,
            "element_size_bytes": self.element_size_bytes,
            "element_shape": self.element_shape,
            "dtype": str(self.dtype) if self.dtype else None,
            "device": self.device,
            "layout": self.layout,
            "page_size": self.page_size,
            "layer_id": self.layer_id,
            "num_layers": self.num_layers,
            "num_pool_ptrs": len(self.pool_ptrs),
            "num_tensors": len(self.tensors) if self.tensors else 0,
        }

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        element_size_bytes: Optional[int] = None,
    ) -> "TransferView":
        """
        Create a TransferView from a single tensor.

        This is a convenience factory method for simple single-tensor cases.

        Args:
            tensor: The tensor to create a view for.
            offsets: Optional indices tensor. If None, creates indices for all elements.
            element_size_bytes: Size per element. If None, computed from tensor.

        Returns:
            TransferView for the given tensor.
        """
        if element_size_bytes is None:
            # Compute element size from tensor shape (excluding first dimension)
            if tensor.dim() > 1:
                element_size = tensor[0].numel() * tensor.element_size()
            else:
                element_size = tensor.element_size()
        else:
            element_size = element_size_bytes

        ptr = tensor.data_ptr() if tensor.numel() > 0 else 0

        # If offsets not provided, create default indices for all elements in first dimension
        if offsets is None:
            offsets = torch.arange(tensor.shape[0], dtype=torch.long)

        return cls(
            pool_ptrs=[ptr],
            offsets=offsets,
            element_size_bytes=element_size,
            tensors=[tensor],
            element_shape=tensor.shape[1:] if tensor.dim() > 1 else (),
            dtype=tensor.dtype,
            device=str(tensor.device),
        )

    @classmethod
    def from_tensors(
        cls,
        tensors: List[torch.Tensor],
        offsets: Optional[torch.Tensor] = None,
        element_size_bytes: Optional[int] = None,
    ) -> "TransferView":
        """
        Create a TransferView from multiple tensors.

        This is useful for multi-buffer pools like KV cache (K and V tensors).

        Args:
            tensors: List of tensors to create a view for.
            offsets: Optional indices tensor. If None, creates indices for all elements.
            element_size_bytes: Size per element per tensor. If None, computed.

        Returns:
            TransferView for the given tensors.
        """
        if not tensors:
            raise ValueError("tensors list cannot be empty")

        if element_size_bytes is None:
            t = tensors[0]
            if t.dim() > 1:
                element_size = t[0].numel() * t.element_size()
            else:
                element_size = t.element_size()
        else:
            element_size = element_size_bytes

        ptrs = [t.data_ptr() for t in tensors]
        dtype = tensors[0].dtype
        device = str(tensors[0].device)
        element_shape = tensors[0].shape[1:] if tensors[0].dim() > 1 else ()

        # If offsets not provided, create default indices for all elements in first dimension
        if offsets is None:
            offsets = torch.arange(tensors[0].shape[0], dtype=torch.long)

        return cls(
            pool_ptrs=ptrs,
            offsets=offsets,
            element_size_bytes=element_size,
            tensors=tensors,
            element_shape=element_shape,
            dtype=dtype,
            device=device,
        )