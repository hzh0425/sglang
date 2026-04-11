"""Unit tests for HiCache integration in UnifiedRadixCache.

Tests the HiCache-specific features: evicted/backuped properties,
host-aware traversal, write_backup/load_back, and scheduler entry points.
"""

import unittest
from collections import namedtuple
from unittest.mock import MagicMock

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-test-small-1-gpu")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_PAGE_SIZE = 1
_HEAD_NUM = 2
_HEAD_DIM = 128
_NUM_LAYERS = 24
_GLOBAL_INTERVAL = 4
_DTYPE = torch.bfloat16


def _full_attention_layer_ids():
    return [i for i in range(_GLOBAL_INTERVAL - 1, _NUM_LAYERS, _GLOBAL_INTERVAL)]


def _mamba_layer_ids():
    full_set = set(_full_attention_layer_ids())
    return [i for i in range(_NUM_LAYERS) if i not in full_set]


# ---------------------------------------------------------------------------
# Mock CUDA event + HiCacheAck for testing writing_check / loading_check
# ---------------------------------------------------------------------------
MockHiCacheAck = namedtuple("MockHiCacheAck", ["start_event", "finish_event", "node_ids"])


class MockEvent:
    """Simulates a CUDA event for unit tests."""

    def __init__(self, done: bool = True):
        self._done = done

    def query(self) -> bool:
        return self._done

    def synchronize(self) -> None:
        self._done = True


# ---------------------------------------------------------------------------
# Mock cache controller
# ---------------------------------------------------------------------------
class MockCacheController:
    """Minimal mock for HybridCacheController used in HiCache tests.

    Provides ack_write_queue / ack_load_queue (list[HiCacheAck])
    matching the real CacheController interface.
    """

    write_policy = "write_through"

    def __init__(self, device):
        self._device = device
        self._next_host_idx = 0
        # Real CacheController queues (list[HiCacheAck])
        self.ack_write_queue: list = []
        self.ack_load_queue: list = []

    def write(self, device_value, priority=None, node_id=-1, extra_pools=None):
        n = len(device_value)
        host_indices = torch.arange(
            self._next_host_idx, self._next_host_idx + n, dtype=torch.int64
        )
        self._next_host_idx += n
        # Fill auto-allocated host_indices in extra_pools
        if extra_pools:
            for pt in extra_pools:
                if pt.host_indices is None:
                    pt.host_indices = torch.arange(
                        self._next_host_idx,
                        self._next_host_idx + len(pt.device_indices),
                        dtype=torch.int64,
                    )
                    self._next_host_idx += len(pt.device_indices)
        # Simulate async completion: immediately add ack
        self.ack_write_queue.append(
            MockHiCacheAck(
                start_event=MockEvent(done=True),
                finish_event=MockEvent(done=True),
                node_ids=[node_id] if node_id >= 0 else [],
            )
        )
        return host_indices

    def load(self, host_indices, priority=None, node_id=-1, extra_pools=None):
        if len(host_indices) == 0:
            return torch.empty((0,), dtype=torch.int64, device=self._device)
        n = len(host_indices)
        # Return fresh device indices
        device_indices = torch.arange(1000, 1000 + n, dtype=torch.int64, device=self._device)
        # Fill auto-allocated device_indices in extra_pools
        if extra_pools:
            for pt in extra_pools:
                if pt.device_indices is None and pt.host_indices is not None:
                    pt.device_indices = torch.arange(
                        2000,
                        2000 + len(pt.host_indices),
                        dtype=torch.int64,
                        device=self._device,
                    )
        # Simulate async completion: immediately add ack
        self.ack_load_queue.append(
            MockHiCacheAck(
                start_event=MockEvent(done=True),
                finish_event=MockEvent(done=True),
                node_ids=[node_id] if node_id >= 0 else [],
            )
        )
        return device_indices

    def evict_host(self, host_value):
        pass

    def start_loading(self):
        return 0


# ===================================================================
# Tests: HiCache integration with Full + Mamba
# ===================================================================
class TestUnifiedRadixCacheHiCache(unittest.TestCase):
    """HiCache-specific tests for UnifiedRadixCache."""

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=_PAGE_SIZE)
        )

    def _build_tree(
        self,
        kv_size: int = 128,
        max_num_reqs: int = 10,
        mamba_cache_size: int = 20,
        max_context_len: int = 128,
    ):
        device = get_device()
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(
                shape=shape, layers=_mamba_layer_ids()
            )

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=kv_size,
            dtype=_DTYPE,
            page_size=_PAGE_SIZE,
            head_num=_HEAD_NUM,
            head_dim=_HEAD_DIM,
            full_attention_layer_ids=_full_attention_layer_ids(),
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=kv_size,
            dtype=_DTYPE,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        tree = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=_PAGE_SIZE,
                disable=False,
                tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            ),
        )

        # Enable HiCache mode
        tree.cache_controller = MockCacheController(device)
        full_comp = tree.components[ComponentType.FULL]
        full_comp._hicache_enabled = True
        if ComponentType.MAMBA in tree.components:
            mamba_comp = tree.components[ComponentType.MAMBA]
            mamba_comp._hicache_enabled = True

        def make_req():
            sp = SamplingParams(temperature=0, max_new_tokens=1)
            req = Req(
                rid=0,
                origin_input_text="",
                origin_input_ids=[],
                sampling_params=sp,
            )
            req_to_token_pool.alloc([req])
            return req

        return tree, allocator, req_to_token_pool, make_req

    def _insert(self, tree, alloc, make_req, token_ids):
        """Helper: insert a sequence and return the leaf node."""
        req = make_req()
        v = alloc.alloc(len(token_ids))
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=v,
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        return m.last_device_node

    # ------- evicted / backuped properties -------

    def test_node_evicted_backuped_properties(self):
        """Verify evicted/backuped correctly reflect node state."""
        tree, alloc, _, make_req = self._build_tree()

        # Insert and get leaf node
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3])
        self.assertFalse(leaf.evicted)
        self.assertFalse(leaf.backuped)

        # Set host_value → backuped=True, evicted still False
        leaf.host_value = torch.tensor([100], dtype=torch.int64)
        self.assertTrue(leaf.backuped)
        self.assertFalse(leaf.evicted)

        # Manually evict device value → evicted=True
        old_value = leaf.component_data[BASE_COMPONENT_TYPE].value
        leaf.component_data[BASE_COMPONENT_TYPE].value = None
        self.assertTrue(leaf.evicted)
        self.assertTrue(leaf.backuped)

        # Remove host_value → dead node (evicted + !backuped)
        leaf.host_value = None
        self.assertTrue(leaf.evicted)
        self.assertFalse(leaf.backuped)

        # Root is never evicted
        self.assertFalse(tree.root_node.evicted)

    # ------- match prefix with evicted nodes -------

    def test_match_prefix_stops_at_dead_node(self):
        """Evicted + !backuped nodes stop traversal."""
        tree, alloc, _, make_req = self._build_tree()

        # Insert [1,2,3] and [1,2,3,4,5]
        self._insert(tree, alloc, make_req, [1, 2, 3])
        self._insert(tree, alloc, make_req, [1, 2, 3, 4, 5])

        # Find the [1,2,3] internal node
        child_key = tree.get_child_key_fn(RadixKey([1, 2, 3]))
        internal = tree.root_node.children[child_key]
        self.assertEqual(len(internal.key), 3)

        # Make [1,2,3] a dead node (evicted + no backup)
        internal.component_data[BASE_COMPONENT_TYPE].value = None
        # No host_value → not backuped

        # Match should stop at root (can't traverse dead node)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        self.assertEqual(len(m.device_indices), 0)

    def test_match_prefix_traverses_evicted_backuped(self):
        """Evicted + backuped nodes are traversed but value not collected."""
        tree, alloc, _, make_req = self._build_tree()

        # Insert [1,2,3,4,5]
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3, 4, 5])

        # Split to create [1,2,3] internal + [4,5] leaf
        self._insert(tree, alloc, make_req, [1, 2, 3, 6, 7])

        child_key = tree.get_child_key_fn(RadixKey([1, 2, 3]))
        internal = tree.root_node.children[child_key]

        # Make internal evicted + backuped
        internal.host_value = torch.tensor([200, 201, 202], dtype=torch.int64)
        internal.component_data[BASE_COMPONENT_TYPE].value = None

        # Match [1,2,3,4,5]: should traverse evicted internal but not collect its value
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        # device_indices should only have the [4,5] leaf's value (2 tokens)
        self.assertEqual(len(m.device_indices), 2)
        # host_hit_length should capture the evicted internal's host_value length
        self.assertEqual(m.host_hit_length, 3)

    # ------- write_backup -------

    def test_write_backup_sets_host_value(self):
        """write_backup stores host indices on the node."""
        tree, alloc, _, make_req = self._build_tree()
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3])

        self.assertIsNone(leaf.host_value)
        written = tree.write_backup(leaf)
        self.assertEqual(written, 3)
        self.assertIsNotNone(leaf.host_value)
        self.assertEqual(len(leaf.host_value), 3)
        self.assertTrue(leaf.backuped)

    # ------- evict to host -------

    def test_evict_to_host(self):
        """_evict_to_host frees device, node stays in tree with host backup."""
        tree, alloc, _, make_req = self._build_tree()
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3])

        # Backup first
        tree.write_backup(leaf)
        self.assertTrue(leaf.backuped)
        self.assertFalse(leaf.evicted)

        # Complete write-through to unlock
        tree.writing_check()

        # Now evict to host
        tree._evict_to_host(leaf)
        self.assertTrue(leaf.evicted)
        self.assertTrue(leaf.backuped)
        # Device value should be None
        self.assertIsNone(leaf.component_data[BASE_COMPONENT_TYPE].value)
        # Node should still be in the tree
        child_key = tree.get_child_key_fn(RadixKey([1, 2, 3]))
        self.assertIn(child_key, tree.root_node.children)

    # ------- load_back -------

    def test_load_back_restores_device_value(self):
        """load_back restores device value from host."""
        tree, alloc, _, make_req = self._build_tree()
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3])

        # Backup → evict to host
        tree.write_backup(leaf)
        tree.writing_check()
        tree._evict_to_host(leaf)
        self.assertTrue(leaf.evicted)

        # Load back
        result = tree.load_back(leaf)
        self.assertIsNotNone(result)
        # Device value should be restored
        self.assertFalse(leaf.evicted)
        self.assertIsNotNone(leaf.component_data[BASE_COMPONENT_TYPE].value)

    # ------- init_load_back (scheduler entry) -------

    def test_init_load_back_returns_indices(self):
        """init_load_back returns device indices for evicted node."""
        tree, alloc, _, make_req = self._build_tree()
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3])

        # Backup → evict
        tree.write_backup(leaf)
        tree.writing_check()
        tree._evict_to_host(leaf)

        params = InitLoadBackParams(
            last_host_node=leaf,
            host_hit_length=3,
        )
        indices, node = tree.init_load_back(params)
        self.assertEqual(len(indices), 3)
        self.assertEqual(node, leaf)

    def test_init_load_back_non_evicted(self):
        """init_load_back returns empty for non-evicted node."""
        tree, alloc, _, make_req = self._build_tree()
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3])

        params = InitLoadBackParams(
            last_host_node=leaf,
            host_hit_length=0,
        )
        indices, node = tree.init_load_back(params)
        self.assertEqual(len(indices), 0)

    # ------- check_hicache_events -------

    def test_check_hicache_events_no_crash(self):
        """check_hicache_events runs without error when nothing pending."""
        tree, _, _, _ = self._build_tree()
        tree.check_hicache_events()  # should not raise

    # ------- _evict_device_leaf strategy -------

    def test_evict_device_leaf_backuped_demotes(self):
        """Backuped leaf is demoted to host, not deleted from tree."""
        tree, alloc, _, make_req = self._build_tree()
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3])

        tree.write_backup(leaf)
        tree.writing_check()
        self.assertTrue(leaf.backuped)

        freed = tree._evict_device_leaf(leaf)
        self.assertEqual(freed, 3)
        # Node stays in tree as evicted
        self.assertTrue(leaf.evicted)
        child_key = tree.get_child_key_fn(RadixKey([1, 2, 3]))
        self.assertIn(child_key, tree.root_node.children)

    def test_evict_device_leaf_not_backuped_cascades(self):
        """Non-backuped leaf is cascade-evicted (deleted from tree)."""
        tree, alloc, _, make_req = self._build_tree()
        leaf = self._insert(tree, alloc, make_req, [1, 2, 3])

        self.assertFalse(leaf.backuped)
        freed = tree._evict_device_leaf(leaf)
        self.assertGreater(freed, 0)
        # Node removed from tree
        child_key = tree.get_child_key_fn(RadixKey([1, 2, 3]))
        self.assertNotIn(child_key, tree.root_node.children)


if __name__ == "__main__":
    unittest.main()
