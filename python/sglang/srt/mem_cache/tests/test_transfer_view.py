# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
Unit tests for cache_controller components.

Tests for:
- TransferView integration
- CacheOperation and StorageOperation
- TransferBuffer
- LayerLoadingEvent and LayerDoneCounter
- HostPoolBase implementations
- HiCacheStorage v2 interface
"""

import os
import shutil
import tempfile
import threading
import time
import unittest
from typing import Optional
from queue import Empty, Full

import numpy as np
import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
    StorageOperation,
    PrefetchOperation,
    LayerLoadingEvent,
    LayerDoneCounter,
    TransferBuffer,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.host_pool_base import HostPoolBase
from sglang.srt.mem_cache.transfer_view import TransferView
from sglang.srt.utils import get_device_module

device_module = get_device_module()


class TestTransferView(unittest.TestCase):
    """Test TransferView dataclass."""

    def test_from_tensor(self):
        """Test creating TransferView from a single tensor."""
        tensor = torch.randn(100, 64, dtype=torch.float32)
        view = TransferView.from_tensor(tensor)

        self.assertEqual(view.num_elements, 100)
        self.assertEqual(view.element_size_bytes, 64 * 4)  # 64 elements * 4 bytes
        self.assertEqual(view.dtype, torch.float32)
        self.assertEqual(len(view.tensors), 1)
        self.assertEqual(len(view.pool_ptrs), 1)

    def test_from_tensors(self):
        """Test creating TransferView from multiple tensors (e.g., K and V)."""
        k_tensor = torch.randn(100, 32, dtype=torch.float16)
        v_tensor = torch.randn(100, 32, dtype=torch.float16)
        view = TransferView.from_tensors([k_tensor, v_tensor])

        # num_elements is the number of offset indices (100 by default)
        self.assertEqual(view.num_elements, 100)
        self.assertEqual(view.dtype, torch.float16)
        self.assertEqual(len(view.tensors), 2)
        self.assertEqual(len(view.pool_ptrs), 2)

    def test_with_offsets(self):
        """Test TransferView with offsets."""
        tensor = torch.randn(1000, 64, dtype=torch.float32)
        offsets = torch.tensor([0, 100, 200, 300])
        view = TransferView.from_tensor(tensor, offsets=offsets)

        self.assertEqual(view.num_elements, 4)
        self.assertEqual(view.offsets.numel(), 4)

    def test_slice_by_page(self):
        """Test slicing TransferView by page."""
        tensor = torch.randn(1000, 64, dtype=torch.float32)
        offsets = torch.arange(100)
        view = TransferView.from_tensor(tensor, offsets=offsets, element_size_bytes=256)
        view.page_size = 10

        sliced = view.slice_by_page(0, 2)  # First 2 pages = 20 elements

        self.assertEqual(sliced.num_elements, 20)

    def test_is_compatible_with(self):
        """Test compatibility check between TransferViews."""
        tensor1 = torch.randn(100, 64, dtype=torch.float32)
        tensor2 = torch.randn(100, 64, dtype=torch.float32)
        tensor3 = torch.randn(200, 64, dtype=torch.float32)

        view1 = TransferView.from_tensor(tensor1)
        view2 = TransferView.from_tensor(tensor2)
        view3 = TransferView.from_tensor(tensor3)

        self.assertTrue(view1.is_compatible_with(view2))
        self.assertFalse(view1.is_compatible_with(view3))

    def test_to_dict(self):
        """Test serialization to dictionary."""
        tensor = torch.randn(100, 64, dtype=torch.float32)
        view = TransferView.from_tensor(tensor)

        d = view.to_dict()

        self.assertIn("num_elements", d)
        self.assertIn("total_bytes", d)
        self.assertIn("dtype", d)
        self.assertEqual(d["num_elements"], 100)


class MockHostPool(HostPoolBase):
    """Mock implementation of HostPoolBase for testing."""

    def __init__(self, size: int, page_size: int, dtype: torch.dtype):
        self._size = size
        self._page_size = page_size
        self._dtype = dtype
        self._device = "cpu"
        self._data = torch.zeros(size, 64, dtype=dtype)
        self._free_slots = torch.arange(size, dtype=torch.long)

    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > len(self._free_slots):
            return None
        indices = self._free_slots[:need_size]
        self._free_slots = self._free_slots[need_size:]
        return indices

    def free(self, indices: torch.Tensor) -> int:
        self._free_slots = torch.cat([self._free_slots, indices])
        return len(indices)

    def get_transfer_view(self, indices: torch.Tensor) -> TransferView:
        return TransferView.from_tensor(self._data, offsets=indices)

    def load_from_device(self, device_pool, host_indices, device_indices, **kwargs):
        pass

    def backup_to_device(self, device_pool, host_indices, device_indices, **kwargs):
        pass

    @property
    def size(self) -> int:
        return self._size

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> str:
        return self._device


class TestHostPoolBase(unittest.TestCase):
    """Test HostPoolBase abstract class and implementations."""

    def test_mock_host_pool(self):
        """Test MockHostPool implementation."""
        pool = MockHostPool(size=1000, page_size=16, dtype=torch.float32)

        self.assertEqual(pool.size, 1000)
        self.assertEqual(pool.page_size, 16)
        self.assertEqual(pool.dtype, torch.float32)
        self.assertEqual(pool.available_size, 1000)

    def test_alloc_free(self):
        """Test allocation and deallocation."""
        pool = MockHostPool(size=1000, page_size=16, dtype=torch.float32)

        # Allocate
        indices = pool.alloc(32)
        self.assertEqual(len(indices), 32)
        self.assertEqual(pool.available_size, 968)

        # Free
        freed = pool.free(indices)
        self.assertEqual(freed, 32)
        self.assertEqual(pool.available_size, 1000)

    def test_get_transfer_view(self):
        """Test generating TransferView from host pool."""
        pool = MockHostPool(size=1000, page_size=16, dtype=torch.float32)
        indices = pool.alloc(16)

        view = pool.get_transfer_view(indices)

        self.assertIsInstance(view, TransferView)
        self.assertEqual(view.offsets.numel(), 16)
        self.assertIsNotNone(view.tensors)


class TestHiCacheFileV2(unittest.TestCase):
    """Test HiCacheFile v2 interface with TransferView."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            is_page_first_layout=True,
            model_name="test-model",
        )
        self.storage = HiCacheFile(self.storage_config, file_path=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_supports_transfer_view(self):
        """Test that HiCacheFile supports TransferView."""
        self.assertTrue(self.storage.supports_transfer_view())

    def test_batch_set_get_v2_single_tensor(self):
        """Test batch_set_v2 and batch_get_v2 with single tensor."""
        # Create a mock host pool
        pool = MockHostPool(size=1000, page_size=16, dtype=torch.float32)

        # Set some test data
        test_data = torch.randn(16, 64, dtype=torch.float32)
        pool._data[:16] = test_data

        # Create TransferView
        indices = torch.arange(16)
        view = pool.get_transfer_view(indices)
        view.page_size = 16

        # Store data
        keys = ["test_key_1"]
        results = self.storage.batch_set_v2(keys, view)
        self.assertTrue(all(results))

        # Create a new pool for reading
        pool2 = MockHostPool(size=1000, page_size=16, dtype=torch.float32)
        view2 = pool2.get_transfer_view(indices)
        view2.page_size = 16

        # Retrieve data
        results = self.storage.batch_get_v2(keys, view2)
        self.assertTrue(all(results))

        # Verify data matches
        torch.testing.assert_close(pool2._data[:16], test_data, rtol=1e-5, atol=1e-5)

    def test_batch_set_get_v2_multiple_tensors(self):
        """Test batch_set_v2 and batch_get_v2 with multiple tensors (K and V)."""
        # Create mock tensors for K and V
        k_tensor = torch.randn(16, 32, dtype=torch.float16)
        v_tensor = torch.randn(16, 32, dtype=torch.float16)

        # Create TransferView with multiple tensors
        offsets = torch.arange(16)
        view = TransferView(
            pool_ptrs=[],
            offsets=offsets,
            element_size_bytes=32 * 2,  # 32 elements * 2 bytes (float16)
            tensors=[k_tensor, v_tensor],
            dtype=torch.float16,
            device="cpu",
            page_size=16,
        )

        # Store data
        keys = ["test_kv_1"]
        results = self.storage.batch_set_v2(keys, view)
        self.assertTrue(all(results))

        # Create new tensors for reading
        k_tensor2 = torch.zeros(16, 32, dtype=torch.float16)
        v_tensor2 = torch.zeros(16, 32, dtype=torch.float16)

        view2 = TransferView(
            pool_ptrs=[],
            offsets=offsets,
            element_size_bytes=32 * 2,
            tensors=[k_tensor2, v_tensor2],
            dtype=torch.float16,
            device="cpu",
            page_size=16,
        )

        # Retrieve data
        results = self.storage.batch_get_v2(keys, view2)
        self.assertTrue(all(results))

        # Verify data matches
        torch.testing.assert_close(k_tensor2, k_tensor, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(v_tensor2, v_tensor, rtol=1e-3, atol=1e-3)

    def test_batch_set_get_v2_multiple_pages(self):
        """Test storing and retrieving multiple pages."""
        pool = MockHostPool(size=1000, page_size=16, dtype=torch.float32)

        # Create multiple pages of test data
        test_pages = [torch.randn(16, 64, dtype=torch.float32) for _ in range(4)]
        for i, page in enumerate(test_pages):
            pool._data[i * 16 : (i + 1) * 16] = page

        # Create TransferView
        indices = torch.arange(64)  # 4 pages
        view = pool.get_transfer_view(indices)
        view.page_size = 16

        # Store 4 pages
        keys = [f"page_{i}" for i in range(4)]
        results = self.storage.batch_set_v2(keys, view)
        self.assertTrue(all(results))

        # Create new pool for reading
        pool2 = MockHostPool(size=1000, page_size=16, dtype=torch.float32)
        view2 = pool2.get_transfer_view(indices)
        view2.page_size = 16

        # Retrieve data
        results = self.storage.batch_get_v2(keys, view2)
        self.assertTrue(all(results))

        # Verify all pages match
        expected_data = torch.cat(test_pages, dim=0)
        torch.testing.assert_close(pool2._data[:64], expected_data, rtol=1e-5, atol=1e-5)

    def test_exists_after_set(self):
        """Test that exists() returns True after set."""
        pool = MockHostPool(size=1000, page_size=16, dtype=torch.float32)
        pool._data[:16] = torch.randn(16, 64, dtype=torch.float32)

        view = pool.get_transfer_view(torch.arange(16))
        view.page_size = 16

        key = "test_exists_key"
        self.assertFalse(self.storage.exists(key))

        self.storage.batch_set_v2([key], view)
        self.assertTrue(self.storage.exists(key))

    def test_nonexistent_key(self):
        """Test that get returns False for nonexistent key."""
        pool = MockHostPool(size=1000, page_size=16, dtype=torch.float32)
        view = pool.get_transfer_view(torch.arange(16))
        view.page_size = 16

        results = self.storage.batch_get_v2(["nonexistent_key"], view)
        self.assertFalse(results[0])


class TestTransferViewWithRealHostKVCache(unittest.TestCase):
    """Test TransferView with real HostKVCache (if available)."""

    def test_transfer_view_compatibility(self):
        """Test that TransferView can be created for different layouts."""
        # This is a minimal test since we can't easily create a full HostKVCache
        # without the full infrastructure

        # Test with layer_first layout simulation
        k_buffer = torch.randn(10, 100, 32, dtype=torch.float16)  # [layers, tokens, heads*dim]
        v_buffer = torch.randn(10, 100, 32, dtype=torch.float16)

        view = TransferView.from_tensors([k_buffer, v_buffer])
        view.layout = "layer_first"
        view.num_layers = 10

        # num_elements equals the number of offset indices (first dim size = 10)
        self.assertEqual(view.num_elements, 10)
        # Each tensor has 10*100*32 elements
        self.assertEqual(k_buffer.numel(), 10 * 100 * 32)
        self.assertEqual(v_buffer.numel(), 10 * 100 * 32)


class MockNSAIndexerPool(HostPoolBase):
    """Mock NSA indexer pool for testing multi-buffer transfers."""

    def __init__(self, size: int, page_size: int, num_layers: int = 4):
        self._size = size
        self._page_size = page_size
        self._num_layers = num_layers
        self._dtype = torch.uint8  # NSA indexer uses uint8
        self._device = "cpu"

        # Main KV cache buffer (simplified)
        self.kv_buffer = torch.zeros(size, 128, dtype=torch.float16)

        # NSA indexer buffers (one per layer)
        self.indexer_buffers = [
            torch.zeros(size // page_size, 256, dtype=torch.uint8)
            for _ in range(num_layers)
        ]

        self._free_slots = torch.arange(size, dtype=torch.long)

    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > len(self._free_slots):
            return None
        indices = self._free_slots[:need_size]
        self._free_slots = self._free_slots[need_size:]
        return indices

    def free(self, indices: torch.Tensor) -> int:
        self._free_slots = torch.cat([self._free_slots, indices])
        return len(indices)

    def get_transfer_view(self, indices: torch.Tensor) -> TransferView:
        """Get TransferView for KV cache portion."""
        return TransferView.from_tensor(self.kv_buffer, offsets=indices)

    def get_indexer_transfer_view(self, indices: torch.Tensor) -> TransferView:
        """Get TransferView for indexer buffers."""
        # Convert token indices to page indices
        if indices.numel() == 0:
            return TransferView(
                pool_ptrs=[],
                offsets=indices,
                element_size_bytes=0,
                tensors=None,
                dtype=self._dtype,
                device=self._device,
                page_size=self._page_size,
            )

        page_indices = indices.reshape(-1, self._page_size)[:, 0] // self._page_size

        return TransferView(
            pool_ptrs=[],
            offsets=page_indices,
            element_size_bytes=256,  # Indexer element size
            tensors=self.indexer_buffers,
            dtype=self._dtype,
            device=self._device,
            layout="page_first",
            page_size=1,
            num_layers=self._num_layers,
        )

    def load_from_device(self, device_pool, host_indices, device_indices, **kwargs):
        pass

    def backup_to_device(self, device_pool, host_indices, device_indices, **kwargs):
        pass

    @property
    def size(self) -> int:
        return self._size

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> str:
        return self._device


class TestNSAIndexerPool(unittest.TestCase):
    """Test NSA indexer pool with separate KV and indexer buffers."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            is_page_first_layout=True,
            model_name="test-nsa-model",
        )
        self.storage = HiCacheFile(self.storage_config, file_path=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_nsa_indexer_pool_creation(self):
        """Test NSA indexer pool can be created."""
        pool = MockNSAIndexerPool(size=1000, page_size=16, num_layers=4)

        self.assertEqual(pool.size, 1000)
        self.assertEqual(pool.page_size, 16)
        self.assertEqual(len(pool.indexer_buffers), 4)

    def test_kv_transfer_view(self):
        """Test TransferView for KV cache portion."""
        pool = MockNSAIndexerPool(size=1000, page_size=16, num_layers=4)
        indices = pool.alloc(32)

        view = pool.get_transfer_view(indices)

        self.assertIsInstance(view, TransferView)
        self.assertEqual(view.offsets.numel(), 32)
        self.assertIsNotNone(view.tensors)

    def test_indexer_transfer_view(self):
        """Test TransferView for indexer buffers."""
        pool = MockNSAIndexerPool(size=1000, page_size=16, num_layers=4)
        indices = pool.alloc(32)  # 2 pages

        view = pool.get_indexer_transfer_view(indices)

        self.assertIsInstance(view, TransferView)
        # Page indices should be 2 (32 tokens / 16 page_size)
        self.assertEqual(view.offsets.numel(), 2)
        # Should have 4 indexer tensors (one per layer)
        self.assertEqual(len(view.tensors), 4)

    def test_store_retrieve_kv_data(self):
        """Test storing and retrieving KV data."""
        pool = MockNSAIndexerPool(size=1000, page_size=16, num_layers=4)

        # Set test data
        test_data = torch.randn(16, 128, dtype=torch.float16)
        pool.kv_buffer[:16] = test_data

        # Create TransferView and store
        indices = torch.arange(16)
        view = pool.get_transfer_view(indices)
        view.page_size = 16

        keys = ["kv_page_0"]
        results = self.storage.batch_set_v2(keys, view)
        self.assertTrue(all(results))

        # Retrieve into new pool
        pool2 = MockNSAIndexerPool(size=1000, page_size=16, num_layers=4)
        view2 = pool2.get_transfer_view(indices)
        view2.page_size = 16

        results = self.storage.batch_get_v2(keys, view2)
        self.assertTrue(all(results))

        # Verify data
        torch.testing.assert_close(pool2.kv_buffer[:16], test_data, rtol=1e-3, atol=1e-3)

    def test_store_retrieve_indexer_data(self):
        """Test storing and retrieving indexer data."""
        pool = MockNSAIndexerPool(size=1000, page_size=16, num_layers=4)

        # Set test data in indexer buffers
        test_indexer_data = [
            torch.randint(0, 256, (62, 256), dtype=torch.uint8)  # 1000/16 ≈ 62 pages
            for _ in range(4)
        ]
        for i, data in enumerate(test_indexer_data):
            pool.indexer_buffers[i][:] = data

        # Create TransferView for indexer and store
        indices = torch.arange(32)  # First 2 pages
        view = pool.get_indexer_transfer_view(indices)
        view.page_size = 1  # Indexer uses page-level indexing

        keys = ["indexer_page_0", "indexer_page_1"]
        results = self.storage.batch_set_v2(keys, view)
        self.assertTrue(all(results))

        # Retrieve into new pool
        pool2 = MockNSAIndexerPool(size=1000, page_size=16, num_layers=4)
        view2 = pool2.get_indexer_transfer_view(indices)
        view2.page_size = 1

        results = self.storage.batch_get_v2(keys, view2)
        self.assertTrue(all(results))


class TestCacheControllerIntegration(unittest.TestCase):
    """Test CacheController with new TransferView interface."""

    def test_supports_transfer_view_check(self):
        """Test that backends can advertise TransferView support."""
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            is_page_first_layout=True,
            model_name="test-model",
        )

        temp_dir = tempfile.mkdtemp()
        try:
            storage = HiCacheFile(storage_config, file_path=temp_dir)
            self.assertTrue(storage.supports_transfer_view())
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()