"""CPU tests for UnifiedRadixCache external Bundle transfers."""

from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.external_cache_connector import (
    ExternalCacheConnector,
    ExternalCacheHit,
    ExternalStoreCompletion,
)
from sglang.srt.mem_cache.external_cache_pool import (
    ExternalPoolStack,
    LogicalDevicePool,
    PagedDevicePool,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
    MooncakeConnector,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Allocator:
    device = "cpu"

    def __init__(self):
        self.next_slot = 100
        self.freed = []

    def available_size(self):
        return 1024

    def alloc(self, count):
        slots = torch.arange(self.next_slot, self.next_slot + count)
        self.next_slot += count
        return slots

    def free(self, slots):
        self.freed.append(torch.as_tensor(slots).clone())


class _Connector(ExternalCacheConnector):
    def __init__(self):
        self.hit_pages = 2
        self.load_ok = True
        self.query_calls = []
        self.load_calls = []
        self.store_calls = []
        self.completed = []

    def query(self, key, local_tokens, transfers):
        self.query_calls.append((key, local_tokens, transfers))
        keys = [f"p{i}" for i in range(self.hit_pages)]
        return ExternalCacheHit(page_keys=keys, hit_tokens=2 * self.hit_pages)

    def load(self, hit, transfers):
        self.load_calls.append((hit, transfers))
        return self.load_ok

    def store_async(self, key, transfers):
        self.store_calls.append((key, transfers))
        return len(self.store_calls)

    def poll_completed(self):
        completed, self.completed = self.completed, []
        return completed

    def wait_for_all_stores(self):
        return self.poll_completed()

    def reset(self):
        pass

    def close(self):
        pass


def _make_cache():
    allocator = _Allocator()
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=allocator,
            page_size=2,
            tree_components=(ComponentType.FULL,),
        )
    )
    connector = _Connector()
    cache.install_external_cache(
        connector,
        sidecars=(
            SidecarPoolSpec(
                PoolName.DEEPSEEK_V4_C4,
                PoolName.KV,
                PoolHitPolicy.ALL_PAGES,
            ),
            SidecarPoolSpec(
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.KV,
                PoolHitPolicy.ALL_PAGES,
            ),
        ),
    )
    return cache, allocator, connector


class TestExternalDevicePools(CustomTestCase):
    def test_logical_pool_has_no_physical_buffer(self):
        pool = LogicalDevicePool(page_size=2)
        self.assertIsNone(pool.kv_buffer)
        self.assertEqual(pool.get_hybrid_pool_buffer(), [])

    def test_paged_pool_resolves_page_major_layer_pointers(self):
        buffers = [
            torch.zeros((8, 4), dtype=torch.uint8),
            torch.zeros((8, 4), dtype=torch.uint8),
        ]
        pool = PagedDevicePool(buffers, slot_page_size=2)

        ptrs, sizes = pool.get_page_buffer_meta(torch.tensor([2, 3, 6, 7]))

        self.assertEqual(
            ptrs,
            [
                buffers[0][1].data_ptr(),
                buffers[1][1].data_ptr(),
                buffers[0][3].data_ptr(),
                buffers[1][3].data_ptr(),
            ],
        )
        self.assertEqual(sizes, [4, 4, 4, 4])


class _MooncakeStorage:
    def __init__(self):
        self.get_calls = []
        self.set_calls = []
        self.clear_calls = 0

    def batch_exists_v2(self, keys, transfers):
        return PoolTransferResult(
            len(keys), {transfer.name: len(keys) for transfer in transfers}
        )

    def batch_get_v2(self, transfers):
        self.get_calls.append(transfers)
        return {transfer.name: [True] * len(transfer.keys) for transfer in transfers}

    def batch_set_v2(self, transfers):
        self.set_calls.append(transfers)
        return {transfer.name: [True] * len(transfer.keys) for transfer in transfers}

    def clear(self):
        self.clear_calls += 1

    def close(self):
        pass


def _make_mooncake_connector():
    storage = _MooncakeStorage()
    stack = ExternalPoolStack(
        anchor=LogicalDevicePool(page_size=2),
        pools={
            PoolName.DEEPSEEK_V4_C4: PagedDevicePool(
                [torch.zeros((8, 4), dtype=torch.uint8)],
                slot_page_size=2,
            )
        },
        sidecars=(
            SidecarPoolSpec(
                PoolName.DEEPSEEK_V4_C4,
                PoolName.KV,
                PoolHitPolicy.ALL_PAGES,
            ),
        ),
    )
    connector = MooncakeConnector(
        pool_stack=stack,
        model_config=SimpleNamespace(hf_config=None),
        server_args=SimpleNamespace(
            model_path="model",
            revision=None,
            kv_cache_dtype="bf16",
            mooncake_store_workers=1,
        ),
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        _storage=storage,
    )
    return connector, storage


class TestMooncakeConnectorBundle(CustomTestCase):
    def test_logical_full_is_not_sent_to_v2_io(self):
        connector, storage = _make_mooncake_connector()
        key = RadixKey(array("q", [1, 2, 3, 4]))
        transfers = [
            PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([2, 3, 6, 7])),
            PoolTransfer(
                name=PoolName.DEEPSEEK_V4_C4,
                device_indices=torch.tensor([2, 3, 6, 7]),
                indices_from_pool=PoolName.KV,
            ),
        ]

        hit = connector.query(key, 0, transfers)
        self.assertEqual(hit.hit_tokens, 4)
        self.assertTrue(connector.load(hit, transfers))

        loaded = storage.get_calls[0]
        self.assertEqual(
            [transfer.name for transfer in loaded], [PoolName.DEEPSEEK_V4_C4]
        )
        self.assertEqual(loaded[0].host_indices.tolist(), [2, 3, 6, 7])
        connector.close()

    def test_store_completion_and_reset_do_not_clear_remote(self):
        connector, storage = _make_mooncake_connector()
        key = RadixKey(array("q", [1, 2]))
        transfers = [
            PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([2, 3])),
            PoolTransfer(
                name=PoolName.DEEPSEEK_V4_C4,
                device_indices=torch.tensor([2, 3]),
                indices_from_pool=PoolName.KV,
            ),
        ]

        operation_id = connector.store_async(key, transfers)
        completions = connector.wait_for_all_stores()
        connector.reset()

        self.assertEqual(completions, [ExternalStoreCompletion(operation_id, True)])
        self.assertEqual(len(storage.set_calls), 1)
        self.assertEqual(storage.clear_calls, 0)
        connector.close()


class TestUnifiedExternalLoad(CustomTestCase):
    def test_match_and_load_commit_the_whole_bundle(self):
        cache, _, connector = _make_cache()
        req = SimpleNamespace(
            rid="r",
            last_node=cache.root_node,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
        )
        key = RadixKey(array("q", [1, 2, 3, 4]))

        match = cache.match_prefix(MatchPrefixParams(key=key, req=req))
        self.assertEqual(match.host_hit_length, 4)
        slots, node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )

        self.assertEqual(slots.tolist(), [100, 101, 102, 103])
        self.assertIsNot(node, cache.root_node)
        self.assertEqual(len(connector.load_calls), 1)
        transfers = connector.load_calls[0][1]
        self.assertEqual(
            [transfer.name for transfer in transfers],
            [
                PoolName.KV,
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
            ],
        )
        for transfer in transfers:
            self.assertEqual(transfer.device_indices.tolist(), slots.tolist())
        local = cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(local.device_indices.tolist(), slots.tolist())

    def test_load_failure_releases_all_slots_without_tree_commit(self):
        cache, allocator, connector = _make_cache()
        connector.load_ok = False
        req = SimpleNamespace(
            rid="r",
            last_node=cache.root_node,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
        )
        key = RadixKey(array("q", [1, 2, 3, 4]))
        match = cache.match_prefix(MatchPrefixParams(key=key, req=req))

        slots, node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )

        self.assertEqual(slots.numel(), 0)
        self.assertIs(node, cache.root_node)
        self.assertEqual(allocator.freed[0].tolist(), [100, 101, 102, 103])
        self.assertEqual(len(cache.root_node.children), 0)

    def test_tree_clamps_remote_hit_with_rank_minimum(self):
        cache, _, _ = _make_cache()
        cache._sync_external_min = lambda value: 1
        req = SimpleNamespace(rid="r")

        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                req=req,
            )
        )

        self.assertEqual(match.host_hit_length, 2)

    def test_rank_allocation_failure_aborts_before_io(self):
        cache, allocator, connector = _make_cache()
        agreed = iter([2, 0])
        cache._sync_external_min = lambda value: next(agreed)
        req = SimpleNamespace(
            rid="r",
            last_node=cache.root_node,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
        )
        key = RadixKey(array("q", [1, 2, 3, 4]))
        match = cache.match_prefix(MatchPrefixParams(key=key, req=req))

        slots, node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )

        self.assertEqual(slots.numel(), 0)
        self.assertIs(node, cache.root_node)
        self.assertEqual(allocator.freed[0].tolist(), [100, 101, 102, 103])
        self.assertEqual(connector.load_calls, [])


class TestUnifiedExternalStoreLifecycle(CustomTestCase):
    def test_completion_releases_store_lock_once(self):
        cache, _, connector = _make_cache()
        key = RadixKey(array("q", [1, 2]))
        slots = torch.tensor([10, 11])
        cache.insert(InsertParams(key=key, value=slots))
        node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node

        op_id = cache._submit_external_store(key, node, slots)
        self.assertEqual(node.component_data[ComponentType.FULL].lock_ref, 1)
        connector.completed = [ExternalStoreCompletion(op_id, True)]

        cache.check_hicache_events()
        self.assertEqual(node.component_data[ComponentType.FULL].lock_ref, 0)
        cache.check_hicache_events()
        self.assertEqual(node.component_data[ComponentType.FULL].lock_ref, 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
