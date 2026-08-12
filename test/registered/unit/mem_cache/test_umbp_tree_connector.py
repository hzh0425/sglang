import json
import os
import threading
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_mappings import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.umbp.umbp_tree_connector import (
    RANGES_PER_CALL,
    LayerWiseLoadCounter,
    UMBPTreeConnector,
    _object_sizes_per_page,
    _ordered_layers,
    _PoolRangePlan,
    _resolve_umbp_pool_group,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_cache_connector_mixin import (
    UnifiedCacheConnectorMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DeploymentMode(Enum):
    Local = 0
    StandaloneProcess = 1
    Distributed = 2


def _assert_page_pointers_match_views(test, entry, indices):
    """Verify precomputed row pointers against tensor-view data pointers."""
    pointers, _ = entry.get_page_buffer_meta(indices)
    test.assertEqual(
        pointers,
        [
            buffer[row].data_ptr()
            for row in entry.prepare_locations(indices)
            for component in entry.components
            for buffer in component
        ],
    )


class TestUMBPTreeConnector(unittest.TestCase):
    page_size = 2
    num_layers = 3

    def setUp(self):
        self.kv_buffers = [
            torch.zeros((32, 1, 4), dtype=torch.uint8) for _ in range(self.num_layers)
        ]
        self.indexer_buffers = [
            torch.zeros((16, 6), dtype=torch.uint8) for _ in range(self.num_layers)
        ]
        identity = {layer: layer for layer in range(self.num_layers)}
        self.pool_group = DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=PoolName.KV,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[self.kv_buffers],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=False,
                ),
                DevicePoolEntry(
                    name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[self.indexer_buffers],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=True,
                ),
            ],
            self.num_layers,
            self.page_size,
        )
        self.pools = self.pool_group.entry_map

        self.client = MagicMock()
        self.client.is_distributed.return_value = True
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess
        self.client.get_backend_mode.return_value = _DeploymentMode.Local
        self.client.supports_ranged_io.return_value = True
        self.client.register_memory.return_value = True
        self.client.batch_exists.side_effect = lambda keys: [True] * len(keys)
        self.client.batch_put_ranges_from_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        self.client.batch_get_ranges_into_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        self.client.report_external_kv_blocks.return_value = True
        self.client.revoke_external_kv_blocks.return_value = True
        self.client.revoke_all_external_kv_blocks_at_tier.return_value = True

        self.storage = MagicMock()
        self.storage.client = self.client
        self.storage._disable_zero_copy_register = False
        self.storage._get_hybrid_page_component_keys.side_effect = (
            lambda keys, transfer, rank_suffix=None: (
                [f"{key}_{rank_suffix or 'rank'}_{transfer.name}" for key in keys],
                1,
            )
        )

        self.freeze_gc_patcher = patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.freeze_gc"
        )
        self.freeze_gc_mock = self.freeze_gc_patcher.start()
        self.addCleanup(self.freeze_gc_patcher.stop)
        self.event_patcher = patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.device_module.Event",
            side_effect=lambda: SimpleNamespace(
                record=lambda: None, synchronize=lambda: None
            ),
        )
        self.event_patcher.start()
        self.addCleanup(self.event_patcher.stop)

        self.server_args = SimpleNamespace(
            hicache_storage_backend_extra_config=None,
            tp_size=1,
            model_path="test-model",
            unified_tree_connector_load_strategy="layer_wise",
        )
        self.params = SimpleNamespace(
            page_size=self.page_size,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tp_cache_group=None,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
        )
        self.connectors = []

    def tearDown(self):
        for connector in self.connectors:
            connector.close()

    def make_connector(
        self, extra_config=None, pool_group=None, load_strategy="layer_wise"
    ):
        pool_group = pool_group or self.pool_group
        self.server_args.hicache_storage_backend_extra_config = (
            json.dumps(extra_config) if extra_config is not None else None
        )
        self.server_args.unified_tree_connector_load_strategy = load_strategy
        with (
            patch(
                "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector._resolve_umbp_pool_group",
                return_value=pool_group,
            ),
            patch(
                "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector._parse_storage_extra_config",
                return_value=dict(extra_config or {}),
            ),
        ):
            connector = UMBPTreeConnector(
                self.server_args, self.params, _storage=self.storage
            )
        self.connectors.append(connector)
        return connector

    def transfer(self, pages=2):
        starts = torch.arange(pages, dtype=torch.int64) * self.page_size
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        indices = (starts[:, None] + offsets).flatten()
        return PoolTransfer(
            name=PoolName.KV,
            device_indices=indices,
            keys=[f"page-{index}" for index in range(pages)],
        )

    @staticmethod
    def wait_for_offloads(connector):
        connector._offload_queue.join()

    def test_object_key_and_pointer_order_are_page_major(self):
        connector = self.make_connector()
        transfer = connector.pool_group.resolve_transfers([self.transfer(pages=2)])[0]

        keys = connector._object_keys(transfer)
        entry = connector.pools[transfer.name]
        locations = entry.prepare_locations(transfer.host_indices)

        # One object per page now; the layer lives inside it as a byte range.
        self.assertEqual(keys, [f"page-{page}_rank_kv" for page in range(2)])

        # `batch_*_ranges` pairs keys with range entries positionally, so the
        # two must stay the same length for every layer. A mismatch here is the
        # shape that silently shifts every range by one object.
        for layer in range(self.num_layers):
            ptrs, sizes, offsets = entry.get_prepared_layer_range_meta(locations, layer)
            self.assertEqual(len(ptrs), len(keys))
            self.assertEqual(len(sizes), len(keys))
            self.assertEqual(len(offsets), len(keys))

        # Offsets must tile the object exactly, in layer order, and the object
        # size the connector declares must match that tiling -- deriving it
        # from the emitted ranges instead would hide a dropped trailing layer.
        per_page = _object_sizes_per_page(entry)
        self.assertEqual(len(per_page), 1)
        cursor = 0
        for layer in range(self.num_layers):
            _, sizes, offsets = entry.get_prepared_layer_range_meta(locations, layer)
            self.assertEqual(offsets[0], [cursor])
            cursor += sizes[0][0]
        self.assertEqual(cursor, per_page[0])

    def test_dsa_transfer_resolution_matches_legacy_expansion(self):
        connector = self.make_connector()
        source = self.transfer(pages=2)

        resolved = connector.pool_group.resolve_transfers([source])

        self.assertEqual([transfer.name for transfer in resolved], list(self.pools))
        for transfer in resolved:
            self.assertEqual(transfer.keys, source.keys)
            self.assertTrue(torch.equal(transfer.host_indices, source.device_indices))
            self.assertIsNone(transfer.indices_from_pool)
            _assert_page_pointers_match_views(
                self, connector.pools[transfer.name], transfer.host_indices
            )

    def test_resolver_accepts_deepseek_v4_pool(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def state_pool():
            return SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(
                    kv_score=torch.zeros((8, 3), dtype=torch.uint8)
                ),
            )

        kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        kvcache._unified_kv = False
        kvcache.start_layer = 0
        kvcache.end_layer = 3
        kvcache.swa_page_size = self.page_size
        kvcache.swa_kv_pool = SimpleNamespace(
            kv_buffer=[
                torch.zeros((8, 3), dtype=torch.uint8) for _ in range(kvcache.end_layer)
            ]
        )
        kvcache.c4_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)]
        )
        kvcache.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[
                torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
            ]
        )
        kvcache.c128_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)]
        )
        kvcache.layer_mapping = [
            DeepSeekV4LayerItem(4, 1),
            DeepSeekV4LayerItem(128, 0),
            DeepSeekV4LayerItem(4, 0),
        ]
        kvcache.compress_state_pools = [state_pool(), None, state_pool()]
        kvcache.indexer_compress_state_pools = [state_pool(), None, state_pool()]

        group = _resolve_umbp_pool_group(kvcache, self.page_size, None)

        self.assertEqual(group.num_layers, 3)
        self.assertEqual(len(group.entry_map), 6)
        self.assertEqual(
            _ordered_layers(group.entry_map[PoolName.DEEPSEEK_V4_C4]), [2, 0]
        )
        probe = torch.arange(self.page_size, dtype=torch.int64)
        for entry in group.entry_map.values():
            _assert_page_pointers_match_views(self, entry, probe)

    def test_hybrid_linear_component_keys_match_qwen_and_kimi_layouts(self):
        from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

        qwen_group, _ = self._hybrid_linear_pool_group(use_mla=False)
        kimi_group, _ = self._hybrid_linear_pool_group(use_mla=True)

        def keys_for(group, pool_name):
            store = UMBPStore.__new__(UMBPStore)
            store.registered_pools = group.entry_map
            store.mla_suffix = "tp0_cp0_pp0"
            store.mha_suffix = "tp0_cp0_pp0"
            store.config_prefix = None
            return store._get_hybrid_page_component_keys(
                ["page"], PoolTransfer(name=pool_name)
            )

        qwen_kv, qwen_multiplier = keys_for(qwen_group, PoolName.KV)
        kimi_kv, kimi_multiplier = keys_for(kimi_group, PoolName.KV)
        mamba, mamba_multiplier = keys_for(qwen_group, PoolName.MAMBA)

        # Qwen's KV pool has separate k and v buffers, but the entry is packed,
        # so a page is one object and gets one key -- same layout Mooncake
        # reaches by giving PoolName.KV a single suffix unconditionally. MAMBA
        # is the one entry built with packed=False, so it keeps one key per
        # component.
        self.assertEqual(qwen_multiplier, 1)
        self.assertEqual(qwen_kv, ["page_tp0_cp0_pp0_kv"])
        self.assertEqual(kimi_multiplier, 1)
        self.assertEqual(kimi_kv, ["page_tp0_cp0_pp0_kv"])
        self.assertEqual(mamba_multiplier, 2)
        self.assertEqual(
            mamba,
            ["page_tp0_cp0_pp0_temporal", "page_tp0_cp0_pp0_conv_0"],
        )

    def test_range_budget_counts_ranges_not_layers(self):
        """The per-RPC budget is in ranges, so it must be read off the ranges.

        A packed pool with two components puts two ranges on an object per
        layer, so deriving the budget from the layer count would overshoot it by
        the component count and could push a long-context request past gRPC's
        message limit. Qwen's KV pool is exactly that shape: separate k and v
        buffers, one packed object.
        """
        group, _ = self._hybrid_linear_pool_group(use_mla=False)
        entry = group.entry_map[PoolName.KV]
        self.assertTrue(entry.packed)
        self.assertEqual(len(entry.components), 2)

        layer = next(iter(entry.layer_mapping))
        _, sizes, _ = entry.get_prepared_layer_range_meta([0, 1], layer=layer)
        # One layer, one packed object: two ranges, not one.
        self.assertEqual([len(entry_sizes) for entry_sizes in sizes], [2, 2])

        budget = UMBPTreeConnector._entries_per_call(sizes)
        self.assertEqual(budget, RANGES_PER_CALL // 2)
        self.assertLessEqual(budget * len(sizes[0]), RANGES_PER_CALL)

    def test_hybrid_linear_page_pointers_match_tensor_views(self):
        probe = torch.arange(self.page_size, dtype=torch.int64)
        for use_mla in (False, True):
            group, _ = self._hybrid_linear_pool_group(use_mla=use_mla)
            for entry in group.entry_map.values():
                _assert_page_pointers_match_views(self, entry, probe)

    def test_connector_registers_mamba_layer_counter(self):
        pool_group, _ = self._hybrid_linear_pool_group(use_mla=False)

        connector = self.make_connector(pool_group=pool_group)

        self.params.req_to_token_pool.register_layer_transfer_counter.assert_called_once_with(
            connector.layer_done_counter
        )

    def test_hybrid_linear_object_keys_align_with_flattened_pointers(self):
        from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

        pool_group, _ = self._hybrid_linear_pool_group(use_mla=False)
        store = UMBPStore.__new__(UMBPStore)
        store.registered_pools = pool_group.entry_map
        store.mla_suffix = "tp0_cp0_pp0"
        store.mha_suffix = "tp0_cp0_pp0"
        store.config_prefix = None
        self.storage._get_hybrid_page_component_keys.side_effect = (
            store._get_hybrid_page_component_keys
        )
        connector = self.make_connector(pool_group=pool_group)
        transfers = pool_group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    device_indices=torch.tensor([0, 1]),
                    keys=["page"],
                ),
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=torch.tensor([0]),
                    keys=["page"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
            ]
        )

        by_name = {transfer.name: transfer for transfer in transfers}
        kv_keys = connector._object_keys(by_name[PoolName.KV])
        kv_ptrs, kv_sizes = connector.pools[PoolName.KV].get_page_buffer_meta(
            by_name[PoolName.KV].host_indices
        )
        mamba_keys = connector._object_keys(by_name[PoolName.MAMBA])
        mamba_ptrs, mamba_sizes = connector.pools[PoolName.MAMBA].get_page_buffer_meta(
            by_name[PoolName.MAMBA].host_indices
        )

        self.assertEqual(
            kv_keys,
            ["page_tp0_cp0_pp0_kv"],
        )
        del kv_ptrs, kv_sizes
        self.assertEqual(
            mamba_keys,
            [
                "page_tp0_cp0_pp0_temporal",
                "page_tp0_cp0_pp0_conv_0",
            ],
        )
        del mamba_ptrs, mamba_sizes
        # MAMBA is the one pool built with packed=False, so it keeps one object
        # per component. Both pools must still pair one key to one range entry
        # per layer -- that equality is what `batch_*_ranges` relies on.
        for name in (PoolName.KV, PoolName.MAMBA):
            entry = connector.pools[name]
            locations = entry.prepare_locations(by_name[name].host_indices)
            keys = connector._object_keys(by_name[name])
            self.assertEqual(
                len(keys),
                len(locations) * (1 if entry.packed else len(entry.components)),
            )
            for layer in connector.pool_layers[name]:
                meta = entry.get_prepared_layer_range_meta(locations, layer)
                if meta is None:
                    continue
                self.assertEqual(len(meta[0]), len(keys))

    def _hybrid_linear_pool_group(self, *, use_mla):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        kvcache.use_mla = use_mla
        kvcache.full_attention_layer_id_mapping = {3: 0}
        if use_mla:
            kvcache.full_kv_pool = SimpleNamespace(
                kv_buffer=[torch.zeros((6, 9), dtype=torch.uint8)]
            )
        else:
            kvcache.full_kv_pool = SimpleNamespace(
                size=4,
                k_scale_buffer=None,
                k_buffer=[torch.zeros((6, 3), dtype=torch.uint8)],
                v_buffer=[torch.zeros((6, 5), dtype=torch.uint8)],
            )
        req_pool = SimpleNamespace(
            mamba_ckpt_pool=None,
            mamba_map={0: 0, 1: 1, 2: 2},
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(
                    temporal=torch.zeros((3, 2, 2, 2), dtype=torch.uint8),
                    conv=[torch.zeros((3, 2, 4), dtype=torch.uint8)],
                )
            ),
            translate_mamba_indices=lambda indices: indices,
        )
        return _resolve_umbp_pool_group(kvcache, self.page_size, req_pool), req_pool

    def test_pool_layers_follow_buffer_indices_not_logical_layer_order(self):
        buffers = [
            torch.zeros((4, 3), dtype=torch.uint8),
            torch.zeros((4, 5), dtype=torch.uint8),
        ]
        entry = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[buffers],
            layer_mapping={0: 1, 2: 0},
            page_size=2,
            rows_are_pages=False,
        )

        layers = _ordered_layers(entry)
        ptrs, _ = entry.get_page_buffer_meta(torch.tensor([0, 1]))

        self.assertEqual(layers, [2, 0])
        self.assertEqual(ptrs, [buffers[0][0].data_ptr(), buffers[1][0].data_ptr()])

    def test_load_plans_allow_different_page_counts_per_pool(self):
        connector = self.make_connector()
        kv_transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([0, 1, 2, 3]),
            keys=["kv-0", "kv-1"],
        )
        indexer_transfer = PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=torch.tensor([0, 1]),
            keys=["indexer-0"],
        )

        plans = connector._build_load_plans([[kv_transfer, indexer_transfer]])

        # One object per page per pool, and the rows the ranges are built from
        # must stay in step with the keys.
        self.assertEqual(
            {plan.name: len(plan.keys) for plan in plans},
            {PoolName.KV: 2, PoolName.INDEXER: 1},
        )
        for plan in plans:
            self.assertEqual(plan.entries_per_page, 1)
            self.assertEqual(len(plan.locations), len(plan.keys))

    def test_layerwise_load_completes_logical_layers_without_objects(self):
        connector = UMBPTreeConnector.__new__(UMBPTreeConnector)
        connector.num_layers = 3
        connector.layer_group = 1
        connector.pool_layers = {PoolName.KV: [0, 2]}
        connector.storage = self.storage
        connector._trace_perf = False
        connector._traced = 0
        connector._trace_budget = 0
        connector._lookup_traced = 0
        connector._start_traced = 0
        connector._exists_build_ms = 0.0
        connector._exists_rpc_ms = 0.0
        connector._exists_keys = 0
        connector.layer_done_counter = LayerWiseLoadCounter(connector.num_layers)
        connector.pools = {PoolName.KV: self.pools[PoolName.KV]}
        plan = _PoolRangePlan(
            name=PoolName.KV,
            keys=["page"],
            locations=[0],
            entries_per_page=1,
        )
        counter = connector.layer_done_counter.update_producer()
        connector.layer_done_counter.set_consumer(counter)

        connector._run_layer_wise_batch(counter, [plan])

        connector.layer_done_counter.wait_until(2)
        # Layers 0 and 2 belong to the pool; layer 1 has no objects and must
        # still complete so the forward thread is released.
        self.assertEqual(self.client.batch_get_ranges_into_ptr.call_count, 2)

    def test_lookup_stops_at_first_partial_page_across_chunks(self):
        connector = self.make_connector()
        # One object per page now, so a chunk holds far more pages than before
        # and all four fit in one probe per pool: pages 1-3 present, page 4 not.
        self.client.batch_exists.side_effect = [
            [True, True, True, False],
            [True, True, True, False],
        ]
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.CHUNK_PAGES",
            2,
        ):
            hit = connector.lookup("rid", [self.transfer(pages=4)])

        self.assertEqual(hit, [1, 2, 3])
        self.assertEqual(self.client.batch_exists.call_count, 2)

    def test_trailing_hit_policy_returns_sparse_valid_prefixes(self):
        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=["tail"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        valid = UMBPTreeConnector._apply_hit_policy(
            [1, 2, 3, 4], [True, True, False, True], transfer
        )

        self.assertEqual(valid, [1, 2, 4])

    def test_lookup_uses_full_kv_key_domain_for_trailing_pool(self):
        pool_group = self._hybrid_pool_group()
        connector = self.make_connector(pool_group=pool_group)
        kv = PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(8),
            keys=["p0", "p1", "p2", "p3"],
        )
        swa = PoolTransfer(
            name=PoolName.SWA,
            device_indices=torch.tensor([6, 7]),
            keys=["p3"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        valid = connector.lookup("rid", [kv, swa])

        self.assertEqual(valid, [1, 2, 3, 4])
        queried = [
            key
            for call in self.client.batch_exists.call_args_list
            for key in call.args[0]
        ]
        self.assertIn("p0_rank_swa", queried)
        self.assertIn("p3_rank_swa", queried)

    def test_offload_allows_partial_hybrid_sources(self):
        connector = self.make_connector(pool_group=self._hybrid_pool_group())
        kv = PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(8),
            keys=["p0", "p1", "p2", "p3"],
        )

        self.assertTrue(connector.offload([kv]))
        self.wait_for_offloads(connector)

        self.assertTrue(connector.pop_completed_offload())
        sent_keys = [
            key
            for call in self.client.batch_put_ranges_from_ptr.call_args_list
            for key in call.args[0]
        ]
        self.assertTrue(sent_keys)
        self.assertTrue(all("deepseek_v4_c4" in key for key in sent_keys))
        self.assertTrue(all("_swa_" not in key for key in sent_keys))

    def _hybrid_pool_group(self):
        identity = {0: 0}
        return DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=PoolName.DEEPSEEK_V4_C4,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[[torch.zeros((4, 3), dtype=torch.uint8)]],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=True,
                ),
                DevicePoolEntry(
                    name=PoolName.SWA,
                    indices_from_pool=PoolName.SWA,
                    device_pool=None,
                    components=[[torch.zeros((4, 5), dtype=torch.uint8)]],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=True,
                ),
            ],
            num_layers=1,
            page_size=self.page_size,
        )

    def test_offload_is_chunked_on_logical_page_boundaries(self):
        connector = self.make_connector()
        # Offload puts one range per layer on an object, so a budget of
        # 2 * num_layers ranges is a budget of 2 objects.
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.RANGES_PER_CALL",
            2 * self.num_layers,
        ):
            self.assertTrue(connector.offload([self.transfer(pages=5)]))
            self.wait_for_offloads(connector)

        # 3 chunks per pool, 2 objects each. A chunk boundary may
        # fall between objects but never inside one: the tier requires an
        # object's ranges to tile it exactly, so a split object could never be
        # published.
        calls = self.client.batch_put_ranges_from_ptr.call_args_list
        self.assertEqual(len(calls), 6)
        self.assertEqual([len(call.args[0]) for call in calls], [2, 2, 1, 2, 2, 1])
        for call in calls:
            for object_size, sizes in zip(call.args[1], call.args[3]):
                self.assertEqual(sum(sizes), object_size)

    def test_offload_keeps_key_range_and_object_size_pairing(self):
        """Every key must leave with its own ranges and its own declared size.

        Desyncing them would store one object's bytes under another object's
        key -- silent corruption that no return value would reveal. There is no
        longer a sort to disturb the pairing (a page object spans every layer
        buffer, so it has no single device address to sort by), but chunking
        still slices five parallel lists.
        """
        connector = self.make_connector()

        expected = {}
        for transfer in connector.pool_group.resolve_transfers(
            [self.transfer(pages=3)]
        ):
            plan = connector._build_load_plans([[transfer]])[0]
            ptrs, sizes, offsets = connector._all_layer_ranges(plan)
            for index, key in enumerate(plan.keys):
                expected[key] = (ptrs[index], sizes[index], offsets[index])

        self.client.batch_put_ranges_from_ptr.reset_mock()
        self.assertTrue(connector.offload([self.transfer(pages=3)]))
        self.wait_for_offloads(connector)

        seen = {}
        for call in self.client.batch_put_ranges_from_ptr.call_args_list:
            keys, object_sizes = call.args[0], call.args[1]
            ptrs, sizes, offsets = call.args[2], call.args[3], call.args[4]
            self.assertEqual(len(keys), len(object_sizes))
            self.assertEqual(len(keys), len(ptrs))
            self.assertEqual(len(keys), len(sizes))
            self.assertEqual(len(keys), len(offsets))
            for index, key in enumerate(keys):
                seen[key] = (ptrs[index], sizes[index], offsets[index])
                # The declared object size must match the tiling, or the tier's
                # exact-tiling check -- the only write-time guard against a
                # dropped trailing layer -- has nothing to compare against.
                self.assertEqual(
                    object_sizes[index],
                    max(o + z for o, z in zip(offsets[index], sizes[index])),
                )
                self.assertEqual(sum(sizes[index]), object_sizes[index])

        self.assertEqual(seen, expected)

    def test_offload_success_produces_exactly_one_result(self):
        event_calls = []

        class _Event:
            def record(self):
                event_calls.append("record")

            def synchronize(self):
                event_calls.append("synchronize")

        def put_after_event(keys, *args):
            self.assertEqual(event_calls, ["record", "synchronize"])
            return [True] * len(keys)

        self.client.batch_put_ranges_from_ptr.side_effect = put_after_event
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.device_module.Event",
            _Event,
        ):
            connector = self.make_connector()

            self.assertTrue(connector.offload([self.transfer(pages=1)]))
            self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertTrue(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_offload_failure_produces_exactly_one_false_result(self):
        self.client.batch_put_ranges_from_ptr.return_value = [False]
        self.client.batch_put_ranges_from_ptr.side_effect = None
        connector = self.make_connector()

        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_offload_exceptions_produce_exactly_one_false_result(self):
        self.client.batch_put_ranges_from_ptr.side_effect = RuntimeError("put failed")
        connector = self.make_connector()

        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

        class _FailingEvent:
            def record(self):
                pass

            def synchronize(self):
                raise RuntimeError("event failed")

        self.client.batch_put_ranges_from_ptr.reset_mock()
        self.client.batch_put_ranges_from_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.device_module.Event",
            _FailingEvent,
        ):
            connector = self.make_connector()
            self.assertTrue(connector.offload([self.transfer(pages=1)]))
            self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.client.batch_put_ranges_from_ptr.assert_not_called()

    def test_offload_results_are_fifo(self):
        def result_for_key(keys, *args):
            return [not keys[0].startswith("fail-")] * len(keys)

        self.client.batch_put_ranges_from_ptr.side_effect = result_for_key
        connector = self.make_connector()
        success = self.transfer(pages=1)
        success.keys = ["success-page"]
        failure = self.transfer(pages=1)
        failure.keys = ["fail-page"]

        self.assertTrue(connector.offload([success]))
        self.assertTrue(connector.offload([failure]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 2)
        self.assertTrue(connector.pop_completed_offload())
        self.assertFalse(connector.pop_completed_offload())

    @staticmethod
    def _drain_fixture(
        pending,
        local_completed,
        slowest_rank_completed,
        *,
        local_results=None,
        peer_results=None,
    ):
        """A stand-in tree with `pending` queued offloads, whose connector reports
        `local_completed` finished ones while the slowest rank reports fewer."""
        local_results = list(local_results or [True] * local_completed)
        peer_results = list(peer_results or [True] * slowest_rank_completed)
        nodes = [
            SimpleNamespace(id=i, connector_offloaded=False) for i in range(pending)
        ]
        cache = SimpleNamespace(
            connector=SimpleNamespace(
                num_completed_offloads=lambda: local_completed,
                pop_completed_offload=lambda: local_results.pop(0),
            ),
            nodes=nodes,
            connector_offloads=[(node, object()) for node in nodes],
            released=[],
        )
        cache.dec_lock_ref = lambda node, params: cache.released.append(node.id)
        cache._connector_sync_min = lambda value: (
            UnifiedCacheConnectorMixin._connector_sync_min(cache, value)
        )
        collective_index = 0

        def all_reduce(tensor, op):
            nonlocal collective_index
            if collective_index == 0:
                tensor.fill_(min(int(tensor.item()), slowest_rank_completed))
            else:
                peer = torch.tensor(peer_results[: tensor.numel()], dtype=tensor.dtype)
                tensor.copy_(torch.minimum(tensor, peer))
            collective_index += 1

        cache._all_reduce_attn_groups = all_reduce
        return cache

    def test_drain_consumes_the_cross_rank_minimum(self):
        """Draining a rank-local count desyncs lock_ref across TP ranks, which
        desyncs evictable_size() and therefore the admission decision -- ranks
        then disagree on whether to run a forward at all and deadlock."""
        cache = self._drain_fixture(
            pending=5, local_completed=4, slowest_rank_completed=2
        )

        UnifiedCacheConnectorMixin.drain_connector_offloads(cache)

        self.assertEqual(cache.released, [0, 1])
        self.assertEqual(len(cache.connector_offloads), 3)

    def test_drain_skips_the_collective_when_nothing_is_queued(self):
        """The empty-queue gate must be rank-consistent on its own: issuing the
        MIN-reduce on only some ranks would itself hang."""
        cache = self._drain_fixture(
            pending=0, local_completed=3, slowest_rank_completed=0
        )
        cache._all_reduce_attn_groups = lambda tensor, op: self.fail(
            "collective issued with an empty offload queue"
        )

        UnifiedCacheConnectorMixin.drain_connector_offloads(cache)

        self.assertEqual(cache.released, [])

    def test_drain_synchronizes_offload_failures_across_ranks(self):
        cache = self._drain_fixture(
            pending=1,
            local_completed=1,
            slowest_rank_completed=1,
            local_results=[True],
            peer_results=[False],
        )

        UnifiedCacheConnectorMixin.drain_connector_offloads(cache)

        self.assertEqual(cache.released, [0])
        self.assertFalse(cache.connector_offloads)
        self.assertFalse(cache.nodes[0].connector_offloaded)

    def test_reset_waits_for_offload_and_discards_stale_result(self):
        worker_started = threading.Event()
        release_worker = threading.Event()
        reset_started = threading.Event()
        reset_done = threading.Event()

        def blocked_put(keys, *args):
            worker_started.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("test did not release offload worker")
            return [True] * len(keys)

        self.client.batch_put_ranges_from_ptr.side_effect = blocked_put
        connector = self.make_connector()
        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.assertTrue(worker_started.wait(timeout=5))

        def run_reset():
            reset_started.set()
            connector.reset()
            reset_done.set()

        reset_thread = threading.Thread(target=run_reset)
        reset_thread.start()
        self.assertTrue(reset_started.wait(timeout=5))
        try:
            self.assertTrue(reset_thread.is_alive())
            self.assertFalse(reset_done.is_set())
        finally:
            release_worker.set()
            reset_thread.join(timeout=5)

        self.assertFalse(reset_thread.is_alive())
        self.assertTrue(reset_done.is_set())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_background_load_completes_each_layer(self):
        connector = self.make_connector()
        # The finest granularity, one call per layer. Grouping is covered by
        # test_layer_group_folds_layers_into_one_call_per_object.
        connector.layer_group = 1
        self.assertTrue(connector.load("rid", [self.transfer(pages=3)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        connector.layer_done_counter.wait_until(self.num_layers - 1)

        # One get per layer and pool at this size.
        self.assertEqual(
            self.client.batch_get_ranges_into_ptr.call_count,
            self.num_layers * len(self.pools),
        )

    def test_prefetch_loads_synchronously_on_worker(self):
        caller_thread = threading.get_ident()
        worker_threads = []

        def record_worker(keys, *args):
            worker_threads.append(threading.get_ident())
            return [True] * len(keys)

        self.client.batch_get_ranges_into_ptr.side_effect = record_worker
        connector = self.make_connector(load_strategy="prefetch")
        connector.layer_done_counter.update_producer = MagicMock(
            wraps=connector.layer_done_counter.update_producer
        )

        self.assertTrue(connector.load("rid", [self.transfer(pages=3)]))

        self.assertTrue(worker_threads)
        self.assertEqual(set(worker_threads), {connector._load_thread.ident})
        self.assertNotEqual(worker_threads[0], caller_thread)
        self.assertFalse(connector._pending)
        self.assertEqual(connector.start_layer_wise_loading(), -1)
        connector.layer_done_counter.update_producer.assert_not_called()
        self.assertEqual(connector._stats["load"], 1)
        self.freeze_gc_mock.assert_called_once_with("UMBP connector")

    def test_prefetch_chunks_aligned_object_triples_once(self):
        connector = self.make_connector(load_strategy="prefetch")
        transfer = self.transfer(pages=7)
        expanded = connector.pool_group.resolve_transfers([transfer])
        plans = connector._build_load_plans([expanded])
        expected = []
        for plan in plans:
            ptrs, sizes, offsets = connector._all_layer_ranges(plan)
            expected.extend(zip(plan.keys, ptrs, sizes, offsets))

        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.RANGES_PER_CALL",
            2 * self.num_layers,
        ):
            self.assertTrue(connector.load("rid", [transfer]))

        calls = self.client.batch_get_ranges_into_ptr.call_args_list
        actual = [
            item
            for call in calls
            for item in zip(call.args[0], call.args[1], call.args[2], call.args[3])
        ]
        # Chunking must not disturb the key-to-range pairing, and prefetch reads
        # every layer of an object in the one call that covers it.
        self.assertEqual(actual, expected)
        self.assertEqual([len(call.args[0]) for call in calls], [2, 2, 2, 1] * 2)
        for _, ranges, _, _ in actual:
            self.assertEqual(len(ranges), self.num_layers)

    def test_prefetch_failure_paths_finish_or_fail_before_enqueue(self):
        self.client.batch_get_ranges_into_ptr.side_effect = lambda keys, *args: [
            False
        ] * len(keys)
        connector = self.make_connector(load_strategy="prefetch")
        self.assertFalse(connector.load("false", [self.transfer(pages=1)]))

        self.client.batch_get_ranges_into_ptr.side_effect = RuntimeError("get failed")
        connector = self.make_connector(load_strategy="prefetch")
        self.assertFalse(connector.load("error", [self.transfer(pages=1)]))
        self.assertTrue(connector._load_thread.is_alive())

        self.client.batch_get_ranges_into_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        connector = self.make_connector(load_strategy="prefetch")
        with patch.object(
            connector, "_build_load_plans", side_effect=ValueError("bad metadata")
        ):
            with self.assertRaisesRegex(ValueError, "bad metadata"):
                connector.load("metadata", [self.transfer(pages=1)])
        self.assertTrue(connector._load_queue.empty())

    def test_load_and_offload_share_one_gc_freeze(self):
        self.freeze_gc_mock.reset_mock()
        connector = self.make_connector()

        self.assertTrue(connector.load("rid", [self.transfer(pages=1)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        connector.layer_done_counter.wait_until(self.num_layers - 1)
        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.freeze_gc_mock.assert_called_once_with("UMBP connector")

    def test_background_load_uses_full_object_budget_per_call(self):
        connector = self.make_connector()
        connector.layer_group = 1
        self.assertTrue(connector.load("rid", [self.transfer(pages=7)]))
        # Layer-wise puts a single range on an object, so a budget of 2 ranges
        # is a budget of 2 objects.
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.RANGES_PER_CALL",
            2,
        ):
            counter = connector.start_layer_wise_loading()
            connector.layer_done_counter.set_consumer(counter)
            connector.layer_done_counter.wait_until(self.num_layers - 1)

        # 7 pages go out as [2, 2, 2, 1] per layer and pool.
        calls = self.client.batch_get_ranges_into_ptr.call_args_list
        self.assertEqual(len(calls), self.num_layers * len(self.pools) * 4)
        self.assertEqual(
            [len(call.args[0]) for call in calls],
            [2, 2, 2, 1] * (self.num_layers * len(self.pools)),
        )
        # Every key carries exactly one range per layer for these pools.
        for call in calls:
            for ranges in call.args[1]:
                self.assertEqual(len(ranges), 1)

    def test_background_load_does_not_split_a_load_that_fits_the_budget(self):
        """Layer-wise loading must spend its whole range budget on objects.

        A layer-wise call carries one range per object, so the number of
        objects it may hold is the budget itself. Budgeting it in pages instead
        divided that by the layer count and split every load into that many
        more round trips -- on GLM-5.1, 1248 RPCs where 156 sufficed, which
        dominated the restore.
        """
        connector = self.make_connector()
        connector.layer_group = 1  # the budget, not the group, is under test
        pages = 7
        self.assertTrue(connector.load("rid", [self.transfer(pages=pages)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        connector.layer_done_counter.wait_until(self.num_layers - 1)

        calls = self.client.batch_get_ranges_into_ptr.call_args_list
        self.assertEqual(len(calls), self.num_layers * len(self.pools))
        for call in calls:
            self.assertEqual(len(call.args[0]), pages)

    def test_layer_group_folds_layers_into_one_call_per_object(self):
        """A group of G layers must cost one call, not G, and still release all G.

        Naming an object on the wire is what a load pays per call -- key bytes,
        key hashing, and the per-object vectors the server rebuilds -- and under
        concurrency that is most of the RPC. Grouping amortizes it, but only if
        every layer in the group still completes: the forward pass waits on each
        one individually.
        """
        connector = self.make_connector()
        group = 2
        connector.layer_group = group
        pages = 3
        self.assertTrue(connector.load("rid", [self.transfer(pages=pages)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        # Completing every layer is the property the forward thread depends on;
        # waiting on the last one would hang if any were skipped.
        for layer in range(self.num_layers):
            connector.layer_done_counter.wait_until(layer)

        # 3 layers in groups of 2 is a ragged split, [0,1] then [2], which is
        # the normal case: no model's layer count divides by the group size.
        group_sizes = [
            min(group, self.num_layers - start)
            for start in range(0, self.num_layers, group)
        ]
        self.assertEqual(group_sizes, [2, 1])
        calls = self.client.batch_get_ranges_into_ptr.call_args_list
        self.assertEqual(len(calls), len(group_sizes) * len(self.pools))
        for call, expected_ranges in zip(
            calls, [size for size in group_sizes for _ in self.pools]
        ):
            self.assertEqual(len(call.args[0]), pages)
            for ranges in call.args[1]:
                self.assertEqual(len(ranges), expected_ranges)

    def test_layer_group_reads_the_same_bytes_as_ungrouped(self):
        """Grouping must not disturb which range lands at which pointer."""
        connector = self.make_connector()
        transfer = self.transfer(pages=3)

        def ranges_for(layer_group):
            self.client.batch_get_ranges_into_ptr.reset_mock()
            connector.layer_group = layer_group
            self.assertTrue(connector.load(f"rid{layer_group}", [transfer]))
            counter = connector.start_layer_wise_loading()
            connector.layer_done_counter.set_consumer(counter)
            connector.layer_done_counter.wait_until(self.num_layers - 1)
            triples = set()
            for call in self.client.batch_get_ranges_into_ptr.call_args_list:
                for key, ptrs, sizes, offsets in zip(*call.args[:4]):
                    triples.update(zip([key] * len(ptrs), ptrs, sizes, offsets))
            return triples

        self.assertEqual(ranges_for(2), ranges_for(1))

    def test_background_load_failure_reaches_consumer(self):
        connector = self.make_connector()
        # One call per layer, so the failure names a single layer.
        connector.layer_group = 1
        call_index = 0

        def fail_second_layer(keys, *args):
            nonlocal call_index
            call_index += 1
            if call_index == 3:
                return [False] * len(keys)
            return [True] * len(keys)

        self.client.batch_get_ranges_into_ptr.side_effect = fail_second_layer
        self.assertTrue(connector.load("rid", [self.transfer(pages=2)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        with self.assertRaisesRegex(
            RuntimeError, "UMBP layer-wise KV load failed"
        ) as raised:
            connector.layer_done_counter.wait_until(1)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(
            str(raised.exception.__cause__),
            "UMBP get failed for pool=kv, layer=1: success=0/2.",
        )

    def test_rejects_unsafe_storage_options(self):
        with self.assertRaisesRegex(ValueError, "ssd_enabled=false"):
            self.make_connector({"ssd_enabled": True})
        with self.assertRaisesRegex(ValueError, "cache_remote_fetches=false"):
            self.make_connector({"cache_remote_fetches": True})
        with self.assertRaisesRegex(ValueError, "smallest per-layer object"):
            self.make_connector({"dram_page_size": 1024})
        with self.assertRaisesRegex(ValueError, "smallest per-layer object"):
            self.make_connector({"dram_page_size": 0})

        connector = self.make_connector({"dram_page_size": 6})
        self.assertFalse(connector._closed)

    def test_standalone_process_skips_distributed_page_config(self):
        self.client.is_distributed.return_value = False
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess

        connector = self.make_connector(
            {
                "standalone_address": "unix:///tmp/umbp-test.sock",
                "dram_page_size": 1024,
            }
        )

        self.assertEqual(connector.deployment_mode, _DeploymentMode.StandaloneProcess)

    def test_rejects_deployment_modes_without_ranged_io(self):
        """Page-granular objects need ranged multi-buffer I/O.

        The worker-facing client must remain StandaloneProcess. The inner
        backend is checked separately through its advertised capability.
        """
        for mode in (_DeploymentMode.Local, _DeploymentMode.Distributed):
            with self.subTest(mode=mode):
                self.client.get_deployment_mode.return_value = mode
                with self.assertRaisesRegex(ValueError, "StandaloneProcess"):
                    self.make_connector()

    def test_rejects_standalone_server_without_ranged_capability(self):
        self.client.supports_ranged_io.return_value = False

        with self.assertRaisesRegex(ValueError, "ranged multi-buffer I/O"):
            self.make_connector()

        self.storage.close.assert_called_once()

    def test_external_kv_reconcile_checks_every_rank_and_pool(self):
        self.client.get_backend_mode.return_value = _DeploymentMode.Distributed
        self.server_args.tp_size = 2
        with patch.dict(
            os.environ,
            {
                "UMBP_EXTKV_REPORT": "1",
                "UMBP_EXTKV_FLUSH_MS": "3600000",
                "UMBP_EXTKV_RECONCILE_SECONDS": "3600",
            },
        ):
            connector = self.make_connector()

        connector._queue_extkv_pages(["0123456789abcdef0123456789abcdef"])
        connector._flush_extkv()

        self.client.report_external_kv_blocks.assert_called_once()
        reported_hash = self.client.report_external_kv_blocks.call_args.args[0][0]
        required = connector._extkv_required_keys[reported_hash]
        self.assertEqual(len(required), 4)  # 2 TP ranks x (KV + INDEXER)
        self.assertTrue(any("tp1_cp0_pp0_kv" in key for key in required))
        self.assertTrue(any("tp0_cp0_pp0_indexer" in key for key in required))

        # Losing a non-anchor object must revoke the logical block even while
        # every KV anchor remains present.
        self.client.batch_exists.side_effect = lambda keys: [
            "indexer" not in key for key in keys
        ]
        connector._reconcile_extkv()

        self.client.revoke_external_kv_blocks.assert_called_once()
        self.assertNotIn(reported_hash, connector._extkv_reported)

        # Re-report the same block, then lose only a non-tp0 KV key. The full
        # required-key AND must revoke it again.
        self.client.batch_exists.side_effect = lambda keys: [True] * len(keys)
        connector._queue_extkv_pages(["0123456789abcdef0123456789abcdef"])
        connector._flush_extkv()
        self.client.batch_exists.side_effect = lambda keys: [
            "tp1_cp0_pp0_kv" not in key for key in keys
        ]
        connector._reconcile_extkv()
        self.assertEqual(self.client.revoke_external_kv_blocks.call_count, 2)
        self.assertNotIn(reported_hash, connector._extkv_reported)

    def test_external_kv_reset_revokes_all_and_clears_state(self):
        self.client.get_backend_mode.return_value = _DeploymentMode.Distributed
        with patch.dict(
            os.environ,
            {
                "UMBP_EXTKV_REPORT": "1",
                "UMBP_EXTKV_FLUSH_MS": "3600000",
                "UMBP_EXTKV_RECONCILE_SECONDS": "3600",
            },
        ):
            connector = self.make_connector()
        connector._queue_extkv_pages(["fedcba9876543210fedcba9876543210"])
        connector._flush_extkv()
        self.assertEqual(len(connector._extkv_reported), 1)
        connector._queue_extkv_pages(["00112233445566778899aabbccddeeff"])
        self.assertEqual(len(connector._extkv_pending), 1)

        connector.reset()

        # Publishing a pending tail immediately before revoke-all only wastes a
        # report RPC. reset clears both reported and pending state directly.
        self.client.report_external_kv_blocks.assert_called_once()
        self.client.revoke_all_external_kv_blocks_at_tier.assert_called_once()
        self.assertFalse(connector._extkv_pending)
        self.assertFalse(connector._extkv_reported)
        self.assertFalse(connector._extkv_required_keys)

    def test_external_kv_repeated_report_revoke_keeps_live_state_bounded(self):
        self.client.get_backend_mode.return_value = _DeploymentMode.Distributed
        live_hashes = set()

        def report(hashes, _tier):
            live_hashes.update(hashes)
            return True

        def revoke(hashes, _tier):
            live_hashes.difference_update(hashes)
            return True

        self.client.report_external_kv_blocks.side_effect = report
        self.client.revoke_external_kv_blocks.side_effect = revoke
        self.client.batch_exists.side_effect = lambda keys: [False] * len(keys)
        with patch.dict(
            os.environ,
            {
                "UMBP_EXTKV_REPORT": "1",
                "UMBP_EXTKV_FLUSH_MS": "3600000",
                "UMBP_EXTKV_RECONCILE_SECONDS": "3600",
            },
        ):
            connector = self.make_connector()

        for index in range(5):
            page_hash = f"{index:032x}"
            connector._queue_extkv_pages([page_hash])
            connector._flush_extkv()
            self.assertEqual(len(live_hashes), 1)

            connector._reconcile_extkv()

            self.assertFalse(live_hashes)
            self.assertFalse(connector._extkv_reported)
            self.assertFalse(connector._extkv_required_keys)

    def test_external_kv_is_disabled_by_default(self):
        self.client.get_backend_mode.return_value = _DeploymentMode.Distributed
        connector = self.make_connector()

        connector._queue_extkv_pages(["page-extkv-disabled"])

        self.client.report_external_kv_blocks.assert_not_called()
        self.assertIsNone(connector._extkv_thread)

    def test_standalone_close_deregisters_before_storage_close(self):
        self.client.is_distributed.return_value = False
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess
        events = []
        self.client.deregister_memory.side_effect = lambda ptr: events.append(
            ("deregister", ptr)
        )
        self.storage.close.side_effect = lambda: events.append(("storage_close", None))
        connector = self.make_connector(
            {"standalone_address": "unix:///tmp/umbp-test.sock"}
        )

        connector.close()

        self.assertGreater(len(connector._registered), 0)
        self.assertEqual(events[0][0], "deregister")
        self.assertEqual(events[1][0], "storage_close")
        self.client.deregister_memory.assert_called_once_with(
            connector._registered[0][0]
        )

    def test_standalone_close_can_retry_deregistration(self):
        self.client.is_distributed.return_value = False
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess
        self.client.deregister_memory.side_effect = [RuntimeError("rpc failed"), None]
        connector = self.make_connector(
            {"standalone_address": "unix:///tmp/umbp-test.sock"}
        )

        with self.assertRaisesRegex(RuntimeError, "rpc failed"):
            connector.close()
        self.assertFalse(connector._closed)
        self.assertFalse(connector._offload_thread.is_alive())
        self.assertFalse(connector._load_thread.is_alive())

        connector.close()
        self.assertTrue(connector._closed)
        self.assertEqual(self.client.deregister_memory.call_count, 2)

    def test_does_not_retain_second_client_reference(self):
        connector = self.make_connector()
        self.assertNotIn("client", vars(connector))

    def test_close_is_idempotent_and_does_not_poison_queue(self):
        connector = self.make_connector()

        connector.close()
        connector.close()
        connector.reset()

        self.storage.close.assert_called_once()

    def test_store_component_keys_match_connector_pool_names(self):
        from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

        store = UMBPStore.__new__(UMBPStore)
        store.registered_pools = self.pools
        store.is_mla_backend = True
        store.mla_suffix = "tp0_cp0_pp0"
        store.config_prefix = None
        transfer = PoolTransfer(name=PoolName.INDEXER)

        keys, multiplier = store._get_hybrid_page_component_keys(
            ["page-0", "page-1"], transfer
        )

        self.assertEqual(multiplier, 1)
        self.assertEqual(
            keys,
            [
                "page-0_tp0_cp0_pp0_indexer",
                "page-1_tp0_cp0_pp0_indexer",
            ],
        )

        store.config_prefix = "model-a"
        tagged_keys, _ = store._get_hybrid_page_component_keys(["page-0"], transfer)
        self.assertEqual(tagged_keys, ["model-a_page-0_tp0_cp0_pp0_indexer"])

    def test_extra_backend_tag_is_a_known_store_option(self):
        from sglang.srt.mem_cache.storage.umbp.umbp_store import _COMMON_EXTRA_KEYS

        self.assertIn("extra_backend_tag", _COMMON_EXTRA_KEYS)

    def test_connector_backend_dispatches_to_mori(self):
        cache = SimpleNamespace(tree_components=(ComponentType.FULL,))
        args = SimpleNamespace(unified_tree_connector_backend="mori")
        params = MagicMock()
        expected = object()
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.UMBPTreeConnector",
            return_value=expected,
        ) as connector_cls:
            UnifiedCacheConnectorMixin.init_connector(cache, args, params)

        connector_cls.assert_called_once_with(args, params)
        self.assertIs(cache.connector, expected)
        self.assertEqual(cache.write_through_threshold, 1)


if __name__ == "__main__":
    unittest.main()
