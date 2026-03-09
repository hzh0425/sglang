# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
TransferView: Generic abstraction for memory transfers between pools.

Enables storage backends to work with arbitrary pool types (KV cache, embeddings, etc.)
without knowing specific memory layouts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TransferView:
    """
    Describes data transfer between memory pools (device/host/storage).

    Supports both pointer-based (zero-copy) and tensor-based transfers.
    """

    pool_ptrs: List[int] = field(default_factory=list)
    offsets: Optional[torch.Tensor] = None
    element_size_bytes: int = 0
    tensors: Optional[List[torch.Tensor]] = None
    element_shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[str] = None
    layout: Optional[str] = None
    page_size: Optional[int] = None
    layer_id: Optional[int] = None
    num_layers: Optional[int] = None
    subviews: Optional[Dict[str, "TransferView"]] = None

    def __post_init__(self):
        if self.offsets is not None and not isinstance(self.offsets, torch.Tensor):
            raise TypeError(f"offsets must be a torch.Tensor, got {type(self.offsets)}")

    @property
    def num_elements(self) -> int:
        if self.offsets is not None:
            return self.offsets.numel()
        elif self.tensors is not None:
            return sum(t.shape[0] for t in self.tensors)
        return 0

    @property
    def total_bytes(self) -> int:
        return self.num_elements * self.element_size_bytes

    def get_flat_tensor(self, tensor_idx: int = 0) -> Optional[torch.Tensor]:
        """Get flattened view of tensor at given index."""
        if self.tensors is None or tensor_idx >= len(self.tensors):
            return None
        return self.tensors[tensor_idx].view(-1)

    def get_element_at(self, index: int, tensor_idx: int = 0) -> Optional[torch.Tensor]:
        """Get element at given offset index."""
        if (
            self.offsets is None
            or self.tensors is None
            or tensor_idx >= len(self.tensors)
        ):
            return None
        return self.tensors[tensor_idx][self.offsets[index].item()]

    def slice_by_page(self, page_idx: int, num_pages: int = 1) -> "TransferView":
        """Create a new TransferView for a subset of pages."""
        if self.page_size is None or self.offsets is None:
            raise ValueError("page_size and offsets must be set for page slicing")

        start = page_idx * self.page_size
        end = min((page_idx + num_pages) * self.page_size, self.offsets.numel())

        return TransferView(
            pool_ptrs=self.pool_ptrs,
            offsets=self.offsets[start:end],
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
        """Check if compatible with another TransferView for paired operations."""
        return (
            self.num_elements == other.num_elements
            and self.element_size_bytes == other.element_size_bytes
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for debugging."""
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
            "num_subviews": len(self.subviews) if self.subviews else 0,
        }

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        element_size_bytes: Optional[int] = None,
    ) -> "TransferView":
        """Create TransferView from a single tensor."""
        if element_size_bytes is None:
            element_size = (
                tensor[0].numel() * tensor.element_size()
                if tensor.dim() > 1
                else tensor.element_size()
            )
        else:
            element_size = element_size_bytes

        ptr = tensor.data_ptr() if tensor.numel() > 0 else 0
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
        """Create TransferView from multiple tensors (e.g., K and V buffers)."""
        if not tensors:
            raise ValueError("tensors list cannot be empty")

        t = tensors[0]
        if element_size_bytes is None:
            element_size = (
                t[0].numel() * t.element_size() if t.dim() > 1 else t.element_size()
            )
        else:
            element_size = element_size_bytes

        if offsets is None:
            offsets = torch.arange(t.shape[0], dtype=torch.long)

        return cls(
            pool_ptrs=[t.data_ptr() for t in tensors],
            offsets=offsets,
            element_size_bytes=element_size,
            tensors=tensors,
            element_shape=t.shape[1:] if t.dim() > 1 else (),
            dtype=t.dtype,
            device=str(t.device),
        )
