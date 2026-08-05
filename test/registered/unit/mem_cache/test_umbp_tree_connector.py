import json
import threading
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_mappings import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.umbp.umbp_tree_connector import (
    LayerWiseLoadCounter,
    UMBPTreeConnector,
    _LayerObjectPlan,
    _ordered_layers,
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
        self.client.get_deployment_mode.return_value = _DeploymentMode.Distributed
        self.client.register_memory.return_value = True
        self.client.batch_put_from_ptr.side_effect = lambda keys, ptrs, sizes: (
            [True] * len(keys)
        )
        self.client.batch_get_into_ptr.side_effect = lambda keys, ptrs, sizes: (
            [True] * len(keys)
        )

        self.storage = MagicMock()
        self.storage.client = self.client
        self.storage._disable_zero_copy_register = False
        self.storage._get_hybrid_page_component_keys.side_effect = (
            lambda keys, transfer: (
                [f"{key}_rank_{transfer.name}" for key in keys],
                1,
            )
        )

        self.server_args = SimpleNamespace(
            hicache_storage_backend_extra_config=None,
            tp_size=1,
            model_path="test-model",
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

    def make_connector(self, extra_config=None):
        self.server_args.hicache_storage_backend_extra_config = (
            json.dumps(extra_config) if extra_config is not None else None
        )
        with (
            patch(
                "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector._resolve_umbp_pool_group",
                return_value=self.pool_group,
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
        ptrs, sizes = connector.pools[transfer.name].get_page_buffer_meta(
            transfer.host_indices
        )

        self.assertEqual(
            keys,
            [
                f"page-{page}_rank_kv_L{layer}"
                for page in range(2)
                for layer in range(self.num_layers)
            ],
        )
        self.assertEqual(len(keys), len(ptrs))
        self.assertEqual(len(keys), len(sizes))

    def test_dsa_transfer_resolution_matches_legacy_expansion(self):
        connector = self.make_connector()
        source = self.transfer(pages=2)

        resolved = connector.pool_group.resolve_transfers([source])

        self.assertEqual([transfer.name for transfer in resolved], list(self.pools))
        for transfer in resolved:
            self.assertEqual(transfer.keys, source.keys)
            self.assertTrue(torch.equal(transfer.host_indices, source.device_indices))
            self.assertIsNone(transfer.indices_from_pool)

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

        self.assertEqual(
            {plan.name: plan.logical_pages for plan in plans},
            {PoolName.KV: 2, PoolName.INDEXER: 1},
        )

    def test_layerwise_load_completes_logical_layers_without_objects(self):
        connector = UMBPTreeConnector.__new__(UMBPTreeConnector)
        connector.num_layers = 3
        connector.pool_layers = {PoolName.KV: [0, 2]}
        connector.storage = self.storage
        connector.layer_done_counter = LayerWiseLoadCounter(connector.num_layers)
        plan = _LayerObjectPlan(
            name=PoolName.KV,
            keys=["page_L0", "page_L2"],
            ptrs=[100, 200],
            sizes=[8, 8],
            logical_pages=1,
            component_count=1,
            pool_layer_count=2,
        )
        counter = connector.layer_done_counter.update_producer()
        connector.layer_done_counter.set_consumer(counter)

        connector._run_layer_wise_batch(counter, [plan])

        connector.layer_done_counter.wait_until(2)
        self.assertEqual(self.client.batch_get_into_ptr.call_count, 2)

    def test_lookup_stops_at_first_partial_page_across_chunks(self):
        connector = self.make_connector()
        # Per pool: first two pages complete, then one complete page plus one
        # layer of the fourth page. Both pools therefore expose 3 pages.
        self.client.batch_exists_consecutive.side_effect = [6, 4, 6, 4]
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.CHUNK_PAGES",
            2,
        ):
            hit = connector.lookup("rid", [self.transfer(pages=4)])

        self.assertEqual(hit, [1, 2, 3])
        self.assertEqual(self.client.batch_exists_consecutive.call_count, 4)

    def test_offload_is_chunked_on_logical_page_boundaries(self):
        connector = self.make_connector()
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.CHUNK_PAGES",
            2,
        ):
            self.assertTrue(connector.offload([self.transfer(pages=5)]))
            self.wait_for_offloads(connector)

        # 3 chunks per pool, and every non-tail chunk contains 2 * L objects.
        calls = self.client.batch_put_from_ptr.call_args_list
        self.assertEqual(len(calls), 6)
        self.assertEqual([len(call.args[0]) for call in calls], [6, 6, 3, 6, 6, 3])

    def test_offload_sends_objects_in_device_address_order(self):
        """Offload must go out sorted by GPU address.

        The storage tier allocates slots in arrival order, so this send order is
        what makes a later layer-wise load able to coalesce. See
        UMBP_STANDALONE_PROCESS_DESIGN.md 11.2.
        """
        connector = self.make_connector()
        self.assertTrue(connector.offload([self.transfer(pages=3)]))
        self.wait_for_offloads(connector)

        for call in self.client.batch_put_from_ptr.call_args_list:
            ptrs = call.args[1]
            self.assertEqual(
                ptrs, sorted(ptrs), "offload batch is not in device-address order"
            )

    def test_offload_reorder_keeps_key_pointer_pairing(self):
        """The permutation must move keys, pointers and sizes together.

        Desyncing them would store one object's bytes under another object's
        key -- silent corruption that no return value would reveal.
        """
        connector = self.make_connector()

        expected = {}
        for transfer in connector.pool_group.resolve_transfers(
            [self.transfer(pages=3)]
        ):
            keys = connector._object_keys(transfer)
            ptrs, sizes = connector.pools[transfer.name].get_page_buffer_meta(
                transfer.host_indices
            )
            for key, ptr, size in zip(keys, ptrs, sizes):
                expected[key] = (ptr, size)

        self.client.batch_put_from_ptr.reset_mock()
        self.assertTrue(connector.offload([self.transfer(pages=3)]))
        self.wait_for_offloads(connector)

        seen = {}
        for call in self.client.batch_put_from_ptr.call_args_list:
            keys, ptrs, sizes = call.args[0], call.args[1], call.args[2]
            self.assertEqual(len(keys), len(ptrs))
            self.assertEqual(len(keys), len(sizes))
            for key, ptr, size in zip(keys, ptrs, sizes):
                seen[key] = (ptr, size)

        self.assertEqual(seen, expected)

    def test_offload_success_produces_exactly_one_result(self):
        connector = self.make_connector()

        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertTrue(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_offload_failure_produces_exactly_one_false_result(self):
        self.client.batch_put_from_ptr.return_value = [False]
        self.client.batch_put_from_ptr.side_effect = None
        connector = self.make_connector()

        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_offload_exception_produces_exactly_one_false_result(self):
        self.client.batch_put_from_ptr.side_effect = RuntimeError("put failed")
        connector = self.make_connector()

        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_offload_results_are_fifo(self):
        def result_for_key(keys, ptrs, sizes):
            return [not keys[0].startswith("fail-")] * len(keys)

        self.client.batch_put_from_ptr.side_effect = result_for_key
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

        def blocked_put(keys, ptrs, sizes):
            worker_started.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("test did not release offload worker")
            return [True] * len(keys)

        self.client.batch_put_from_ptr.side_effect = blocked_put
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
        self.assertTrue(connector.load("rid", [self.transfer(pages=3)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        connector.layer_done_counter.wait_until(self.num_layers - 1)

        # One get per layer and pool at this size.
        self.assertEqual(
            self.client.batch_get_into_ptr.call_count,
            self.num_layers * len(self.pools),
        )

    def test_background_load_uses_full_object_budget_per_call(self):
        connector = self.make_connector()
        self.assertTrue(connector.load("rid", [self.transfer(pages=7)]))
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector.CHUNK_PAGES",
            2,
        ):
            counter = connector.start_layer_wise_loading()
            connector.layer_done_counter.set_consumer(counter)
            connector.layer_done_counter.wait_until(self.num_layers - 1)

        # max_objects_per_call = CHUNK_PAGES * layers = 6. Each layer/pool
        # therefore transfers 7 objects as [6, 1], rather than [2, 2, 2, 1].
        calls = self.client.batch_get_into_ptr.call_args_list
        self.assertEqual(len(calls), self.num_layers * len(self.pools) * 2)
        self.assertEqual(
            [len(call.args[0]) for call in calls],
            [6, 1] * (self.num_layers * len(self.pools)),
        )

    def test_background_load_failure_reaches_consumer(self):
        connector = self.make_connector()
        call_index = 0

        def fail_second_layer(keys, ptrs, sizes):
            nonlocal call_index
            call_index += 1
            if call_index == 3:
                return [False] * len(keys)
            return [True] * len(keys)

        self.client.batch_get_into_ptr.side_effect = fail_second_layer
        self.assertTrue(connector.load("rid", [self.transfer(pages=2)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        with self.assertRaisesRegex(RuntimeError, "UMBP layer-wise KV load failed"):
            connector.layer_done_counter.wait_until(1)

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

    def test_rejects_local_deployment_mode(self):
        self.client.is_distributed.return_value = False
        self.client.get_deployment_mode.return_value = _DeploymentMode.Local

        with self.assertRaisesRegex(ValueError, "Distributed or StandaloneProcess"):
            self.make_connector()

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
