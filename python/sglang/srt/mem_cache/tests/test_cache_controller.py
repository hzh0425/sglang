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
from queue import Empty, Full
from typing import Optional

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


class TestCacheOperation(unittest.TestCase):
    """Test CacheOperation class."""

    def test_cache_operation_creation(self):
        """Test creating a CacheOperation."""
        host_indices = torch.tensor([0, 1, 2, 3])
        device_indices = torch.tensor([10, 11, 12, 13])

        op = CacheOperation(host_indices, device_indices, node_id=1)

        self.assertEqual(op.node_ids, [1])
        self.assertIsNotNone(op.id)
        self.assertEqual(op.priority, op.id)

    def test_cache_operation_with_priority(self):
        """Test creating a CacheOperation with explicit priority."""
        host_indices = torch.tensor([0, 1, 2])
        device_indices = torch.tensor([10, 11, 12])

        op = CacheOperation(host_indices, device_indices, node_id=1, priority=5)

        self.assertEqual(op.priority, 5)

    def test_merge_single_operation(self):
        """Test merging a single operation returns itself."""
        host_indices = torch.tensor([0, 1, 2])
        device_indices = torch.tensor([10, 11, 12])
        op = CacheOperation(host_indices, device_indices, node_id=1)

        merged = CacheOperation.merge_ops([op])

        self.assertIs(merged, op)

    def test_merge_multiple_operations(self):
        """Test merging multiple operations."""
        host_indices1 = torch.tensor([0, 1])
        device_indices1 = torch.tensor([10, 11])
        op1 = CacheOperation(host_indices1, device_indices1, node_id=1, priority=1)

        host_indices2 = torch.tensor([2, 3])
        device_indices2 = torch.tensor([12, 13])
        op2 = CacheOperation(host_indices2, device_indices2, node_id=2, priority=2)

        merged = CacheOperation.merge_ops([op1, op2])

        self.assertEqual(len(merged.host_indices), 4)
        self.assertEqual(len(merged.device_indices), 4)
        self.assertEqual(merged.node_ids, [1, 2])
        self.assertEqual(merged.priority, 1)  # min priority

    def test_comparison(self):
        """Test CacheOperation comparison."""
        op1 = CacheOperation(torch.tensor([0]), torch.tensor([1]), node_id=1, priority=1)
        op2 = CacheOperation(torch.tensor([0]), torch.tensor([1]), node_id=2, priority=2)

        self.assertTrue(op1 < op2)


class TestTransferBuffer(unittest.TestCase):
    """Test TransferBuffer class."""

    def test_transfer_buffer_creation(self):
        """Test creating a TransferBuffer."""
        stop_event = threading.Event()
        buffer = TransferBuffer(stop_event, buffer_count=3, max_buffer_size=10)

        self.assertTrue(buffer.empty())
        self.assertFalse(buffer.full())

    def test_transfer_buffer_put_get(self):
        """Test put and get operations."""
        stop_event = threading.Event()
        buffer = TransferBuffer(stop_event, buffer_count=3, max_buffer_size=10)

        item = {"test": "data"}
        buffer.put(item, block=False)

        self.assertFalse(buffer.empty())

        retrieved = buffer.get(block=False)
        self.assertEqual(retrieved, item)

    def test_transfer_buffer_full(self):
        """Test buffer full condition."""
        stop_event = threading.Event()
        buffer = TransferBuffer(stop_event, buffer_count=2, max_buffer_size=10)

        buffer.put({"item": 1}, block=False)
        buffer.put({"item": 2}, block=False)

        self.assertTrue(buffer.full())

    def test_transfer_buffer_clear(self):
        """Test buffer clear."""
        stop_event = threading.Event()
        buffer = TransferBuffer(stop_event, buffer_count=3, max_buffer_size=10)

        buffer.put({"item": 1}, block=False)
        buffer.put({"item": 2}, block=False)

        buffer.clear()

        self.assertTrue(buffer.empty())

    def test_transfer_buffer_stop_event(self):
        """Test that stop_event stops put operation."""
        stop_event = threading.Event()
        buffer = TransferBuffer(stop_event, buffer_count=1, max_buffer_size=10)

        # Fill the buffer
        buffer.put({"item": 1}, block=False)
        self.assertTrue(buffer.full())

        # Set stop event and try to put - should not block forever
        stop_event.set()

        # This should return quickly without blocking
        start = time.time()
        buffer.put({"item": 2}, block=True, timeout=0.1)
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.5)


class TestStorageOperation(unittest.TestCase):
    """Test StorageOperation class."""

    def test_storage_operation_creation(self):
        """Test creating a StorageOperation."""
        host_indices = torch.tensor([0, 1, 2, 3])
        token_ids = [1, 2, 3, 4]

        op = StorageOperation(host_indices, token_ids)

        self.assertEqual(op.token_ids, token_ids)
        self.assertEqual(op.completed_tokens, 0)
        self.assertIsNotNone(op.id)

    def test_storage_operation_with_hash(self):
        """Test StorageOperation with hash values."""
        host_indices = torch.tensor([0, 1])
        token_ids = [1, 2]
        hash_value = ["hash1", "hash2"]

        op = StorageOperation(host_indices, token_ids, hash_value=hash_value)

        self.assertEqual(op.hash_value, hash_value)


class TestPrefetchOperation(unittest.TestCase):
    """Test PrefetchOperation class."""

    def test_prefetch_operation_creation(self):
        """Test creating a PrefetchOperation."""
        host_indices = torch.tensor([0, 1, 2, 3])
        token_ids = [1, 2, 3, 4]
        request_id = "test_req_1"

        op = PrefetchOperation(request_id, host_indices, token_ids)

        self.assertEqual(op.request_id, request_id)
        self.assertEqual(op.completed_tokens, 0)
        self.assertFalse(op.is_terminated())

    def test_increment(self):
        """Test increment operation."""
        host_indices = torch.tensor([0, 1, 2, 3])
        token_ids = [1, 2, 3, 4]
        request_id = "test_req_1"

        op = PrefetchOperation(request_id, host_indices, token_ids)

        result = op.increment(10)
        self.assertTrue(result)
        self.assertEqual(op.completed_tokens, 10)

    def test_terminate(self):
        """Test terminate operation."""
        host_indices = torch.tensor([0, 1, 2, 3])
        token_ids = [1, 2, 3, 4]
        request_id = "test_req_1"

        op = PrefetchOperation(request_id, host_indices, token_ids)
        op.mark_terminate()

        self.assertTrue(op.is_terminated())

        # Increment should return False after termination
        result = op.increment(10)
        self.assertFalse(result)
        self.assertEqual(op.completed_tokens, 0)


class TestLayerLoadingEvent(unittest.TestCase):
    """Test LayerLoadingEvent class."""

    def test_layer_loading_event_creation(self):
        """Test creating a LayerLoadingEvent."""
        event = LayerLoadingEvent(num_layers=4)

        self.assertEqual(len(event.load_events), 4)
        self.assertIsNotNone(event.start_event)

    def test_complete_and_wait(self):
        """Test complete and wait operations."""
        event = LayerLoadingEvent(num_layers=2)

        # Complete first layer
        event.complete(0)
        # This should not raise an error
        event.wait(0)

    def test_finish_event(self):
        """Test finish_event property."""
        event = LayerLoadingEvent(num_layers=2)

        self.assertIs(event.finish_event, event.load_events[-1])


class TestLayerDoneCounter(unittest.TestCase):
    """Test LayerDoneCounter class."""

    def test_layer_done_counter_creation(self):
        """Test creating a LayerDoneCounter."""
        counter = LayerDoneCounter(num_layers=4)

        self.assertEqual(counter.num_layers, 4)
        self.assertEqual(counter.num_counters, 3)
        self.assertEqual(len(counter.events), 3)
        self.assertEqual(counter.producer_index, -1)
        self.assertEqual(counter.consumer_index, -1)

    def test_update_producer(self):
        """Test update_producer operation."""
        counter = LayerDoneCounter(num_layers=2)

        idx = counter.update_producer()
        self.assertEqual(idx, 0)

        idx = counter.update_producer()
        self.assertEqual(idx, 1)

    def test_set_consumer(self):
        """Test set_consumer operation."""
        counter = LayerDoneCounter(num_layers=2)

        counter.set_consumer(0)
        self.assertEqual(counter.consumer_index, 0)

    def test_reset(self):
        """Test reset operation."""
        counter = LayerDoneCounter(num_layers=2)

        counter.update_producer()
        counter.set_consumer(0)
        counter.reset()

        self.assertEqual(counter.producer_index, -1)
        self.assertEqual(counter.consumer_index, -1)


class TestHostPoolBaseImplementations(unittest.TestCase):
    """Test HostPoolBase implementations."""

    def test_mock_host_pool_full_implementation(self):
        """Test a fully implemented mock HostPoolBase."""
        pool = MockHostPoolFull(size=1000, page_size=16, dtype=torch.float32)

        # Test properties
        self.assertEqual(pool.size, 1000)
        self.assertEqual(pool.page_size, 16)
        self.assertEqual(pool.dtype, torch.float32)
        self.assertEqual(pool.device, "cpu")

        # Test allocation
        indices = pool.alloc(32)
        self.assertEqual(len(indices), 32)

        # Test TransferView
        view = pool.get_transfer_view(indices)
        self.assertIsInstance(view, TransferView)

        # Test free
        freed = pool.free(indices)
        self.assertEqual(freed, 32)


class MockHostPoolFull(HostPoolBase):
    """Full mock implementation of HostPoolBase for testing."""

    def __init__(self, size: int, page_size: int, dtype: torch.dtype):
        self._size = size
        self._page_size = page_size
        self._dtype = dtype
        self._device = "cpu"
        self._data = torch.zeros(size, 64, dtype=dtype)
        self._free_slots = torch.arange(size, dtype=torch.long)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
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


class TestHiCacheStorageV2Full(unittest.TestCase):
    """Full tests for HiCacheStorage v2 interface."""

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

    def test_v2_interface_availability(self):
        """Test that v2 interface is available."""
        self.assertTrue(hasattr(self.storage, "batch_get_v2"))
        self.assertTrue(hasattr(self.storage, "batch_set_v2"))
        self.assertTrue(self.storage.supports_transfer_view())

    def test_batch_set_get_with_different_dtypes(self):
        """Test v2 interface with different data types."""
        for dtype in [torch.float32, torch.float16]:
            tensor = torch.randn(16, 32, dtype=dtype)
            view = TransferView.from_tensor(tensor)
            view.page_size = 16

            keys = [f"dtype_test_{dtype}_0"]
            results = self.storage.batch_set_v2(keys, view)
            self.assertTrue(all(results), f"Failed for dtype {dtype}")

            tensor2 = torch.zeros(16, 32, dtype=dtype)
            view2 = TransferView.from_tensor(tensor2)
            view2.page_size = 16

            results = self.storage.batch_get_v2(keys, view2)
            self.assertTrue(all(results), f"Failed to retrieve for dtype {dtype}")

    def test_batch_set_get_with_various_sizes(self):
        """Test v2 interface with various sizes."""
        for size in [1, 16, 64, 128]:
            tensor = torch.randn(size, 32, dtype=torch.float32)
            view = TransferView.from_tensor(tensor)
            view.page_size = size

            keys = [f"size_test_{size}_0"]
            results = self.storage.batch_set_v2(keys, view)
            self.assertTrue(all(results), f"Failed for size {size}")

            tensor2 = torch.zeros(size, 32, dtype=torch.float32)
            view2 = TransferView.from_tensor(tensor2)
            view2.page_size = size

            results = self.storage.batch_get_v2(keys, view2)
            self.assertTrue(all(results), f"Failed to retrieve for size {size}")

    def test_batch_operations_with_multiple_keys(self):
        """Test batch operations with multiple keys."""
        num_keys = 10
        tensors = [torch.randn(16, 32, dtype=torch.float32) for _ in range(num_keys)]

        # Store all tensors
        keys = [f"multi_key_{i}" for i in range(num_keys)]
        for i, tensor in enumerate(tensors):
            view = TransferView.from_tensor(tensor)
            view.page_size = 16
            results = self.storage.batch_set_v2([keys[i]], view)
            self.assertTrue(all(results))

        # Verify all can be retrieved
        for i, tensor in enumerate(tensors):
            tensor2 = torch.zeros(16, 32, dtype=torch.float32)
            view = TransferView.from_tensor(tensor2)
            view.page_size = 16
            results = self.storage.batch_get_v2([keys[i]], view)
            self.assertTrue(all(results))
            torch.testing.assert_close(tensor2, tensor, rtol=1e-5, atol=1e-5)

    def test_idempotent_set(self):
        """Test that setting an existing key succeeds."""
        tensor = torch.randn(16, 32, dtype=torch.float32)
        view = TransferView.from_tensor(tensor)
        view.page_size = 16

        key = "idempotent_key"
        results = self.storage.batch_set_v2([key], view)
        self.assertTrue(all(results))

        # Set again with different data - should succeed (idempotent)
        tensor2 = torch.randn(16, 32, dtype=torch.float32)
        view2 = TransferView.from_tensor(tensor2)
        view2.page_size = 16

        results = self.storage.batch_set_v2([key], view2)
        self.assertTrue(all(results))


class TestTransferViewEdgeCases(unittest.TestCase):
    """Test TransferView edge cases."""

    def test_empty_tensor(self):
        """Test TransferView with empty tensor."""
        tensor = torch.tensor([], dtype=torch.float32)
        view = TransferView.from_tensor(tensor)

        self.assertEqual(view.num_elements, 0)
        self.assertEqual(view.total_bytes, 0)

    def test_single_element(self):
        """Test TransferView with single element."""
        tensor = torch.tensor([42.0], dtype=torch.float32)
        view = TransferView.from_tensor(tensor)

        self.assertEqual(view.num_elements, 1)
        self.assertEqual(view.element_size_bytes, 4)

    def test_very_large_tensor(self):
        """Test TransferView with large tensor."""
        tensor = torch.randn(10000, 128, dtype=torch.float32)
        view = TransferView.from_tensor(tensor)

        self.assertEqual(view.num_elements, 10000)
        expected_bytes = 10000 * 128 * 4
        self.assertEqual(view.total_bytes, expected_bytes)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with v1 interface."""

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

    def test_v1_interface_still_works(self):
        """Test that v1 interface still works."""
        tensor = torch.randn(16, 32, dtype=torch.float32)

        # Use v1 set
        self.assertTrue(self.storage.set("v1_key", tensor))

        # Use v1 get
        target = torch.zeros(16, 32, dtype=torch.float32)
        result = self.storage.get("v1_key", target)

        self.assertIsNotNone(result)
        torch.testing.assert_close(result, tensor, rtol=1e-5, atol=1e-5)

    def test_exists_still_works(self):
        """Test that exists() still works."""
        tensor = torch.randn(16, 32, dtype=torch.float32)

        self.assertFalse(self.storage.exists("new_key"))

        self.storage.set("new_key", tensor)

        self.assertTrue(self.storage.exists("new_key"))


if __name__ == "__main__":
    unittest.main()