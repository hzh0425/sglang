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
    build_deepseek_v4_external_pool_stack,
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

    def test_dsv4_split_layout_exposes_component_and_state_pools(self):
        def paged_pool(layers=1):
            return SimpleNamespace(
                kv_buffer=[
                    torch.zeros((8, 4), dtype=torch.uint8) for _ in range(layers)
                ]
            )

        def state_pool():
            return SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(
                    kv_score=torch.zeros((16, 3), dtype=torch.float32)
                ),
            )

        c4_state = state_pool()
        c4_indexer_state = state_pool()
        kvcache = SimpleNamespace(
            _unified_kv=False,
            swa_page_size=2,
            swa_kv_pool=paged_pool(2),
            c4_kv_pool=paged_pool(),
            c128_kv_pool=paged_pool(),
            c4_indexer_kv_pool=SimpleNamespace(
                index_k_with_scale_buffer=paged_pool().kv_buffer
            ),
            start_layer=0,
            end_layer=2,
            layer_mapping=[
                SimpleNamespace(compress_ratio=4),
                SimpleNamespace(compress_ratio=128),
            ],
            compress_state_pools=[c4_state, state_pool()],
            indexer_compress_state_pools=[c4_indexer_state, None],
        )

        stack = build_deepseek_v4_external_pool_stack(kvcache, page_size=2)

        self.assertEqual(stack.component_pools, {ComponentType.SWA: PoolName.SWA})
        self.assertIn(PoolName.DEEPSEEK_V4_C4_STATE, stack.pools)
        self.assertIn(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, stack.pools)
        state_specs = {
            spec.pool_name: spec for spec in stack.sidecars if "state" in spec.pool_name
        }
        self.assertTrue(state_specs)
        self.assertTrue(
            all(
                spec.indices_from_pool == PoolName.SWA
                and spec.hit_policy == PoolHitPolicy.TRAILING_PAGES
                for spec in state_specs.values()
            )
        )


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


def _make_mooncake_connector(*, tp_rank=0, tp_size=1):
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
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        _storage=storage,
    )
    return connector, storage


class TestMooncakeConnectorBundle(CustomTestCase):
    def test_each_tp_rank_uses_its_own_namespace_and_stores(self):
        connector, storage = _make_mooncake_connector(tp_rank=1, tp_size=2)
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

        self.assertIn(
            "_tp1of2_cp0of1", storage.storage_config.extra_config["extra_backend_tag"]
        )
        self.assertEqual(completions, [ExternalStoreCompletion(operation_id, True)])
        self.assertEqual(len(storage.set_calls), 1)
        connector.close()

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

    def test_trailing_component_and_sidecar_use_the_same_tail(self):
        connector, storage = _make_mooncake_connector()
        connector.pool_stack.pools.update(
            {
                PoolName.SWA: PagedDevicePool(
                    [torch.zeros((8, 4), dtype=torch.uint8)], slot_page_size=2
                ),
                PoolName.DEEPSEEK_V4_C4_STATE: PagedDevicePool(
                    [torch.zeros((8, 4), dtype=torch.uint8)], slot_page_size=2
                ),
            }
        )
        key = RadixKey(array("q", [1, 2, 3, 4]))
        transfers = [
            PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([2, 3, 6, 7])),
            PoolTransfer(
                name=PoolName.DEEPSEEK_V4_C4,
                device_indices=torch.tensor([2, 3, 6, 7]),
                indices_from_pool=PoolName.KV,
            ),
            PoolTransfer(
                name=PoolName.SWA,
                device_indices=torch.tensor([10, 11]),
                keys=["__placeholder__"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
            PoolTransfer(
                name=PoolName.DEEPSEEK_V4_C4_STATE,
                device_indices=torch.tensor([10, 11]),
                keys=["__placeholder__"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
                indices_from_pool=PoolName.SWA,
            ),
        ]

        hit = connector.query(key, 0, transfers)
        self.assertTrue(connector.load(hit, transfers))

        loaded = {transfer.name: transfer for transfer in storage.get_calls[0]}
        self.assertEqual(len(loaded[PoolName.DEEPSEEK_V4_C4].keys), 2)
        self.assertEqual(len(loaded[PoolName.SWA].keys), 1)
        self.assertEqual(
            loaded[PoolName.DEEPSEEK_V4_C4_STATE].keys,
            loaded[PoolName.SWA].keys,
        )
        self.assertEqual(
            loaded[PoolName.DEEPSEEK_V4_C4_STATE].host_indices.tolist(),
            [10, 11],
        )
        connector.close()


class TestUnifiedExternalLoad(CustomTestCase):
    def test_component_tail_materializes_its_physical_page(self):
        indices = torch.tensor([0, 0, 6, 7])

        physical = UnifiedRadixCache._materialize_external_pages(indices, 4)

        self.assertEqual(physical.tolist(), [4, 5, 6, 7])

    def test_match_and_load_commit_the_whole_bundle(self):
        cache, _, connector = _make_cache()
        req = SimpleNamespace(
            rid="r",
            session=None,
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
            session=None,
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
        req = SimpleNamespace(rid="r", session=None)

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
            session=None,
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
