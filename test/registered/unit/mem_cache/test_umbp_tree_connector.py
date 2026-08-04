import json
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.umbp.umbp_tree_connector import (
    UMBPTreeConnector,
    _LogicalPool,
    _PageRowsPool,
    _TokenRowsPool,
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
        self.pools = {
            PoolName.KV: _TokenRowsPool(self.kv_buffers, self.page_size),
            PoolName.INDEXER: _PageRowsPool(self.indexer_buffers, self.page_size),
        }
        self.anchor = _LogicalPool(self.page_size)

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
                "sglang.srt.mem_cache.storage.umbp.umbp_tree_connector._build_pools",
                return_value=(self.anchor, self.pools),
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

    def test_object_key_and_pointer_order_are_page_major(self):
        connector = self.make_connector()
        transfer = connector._expand([self.transfer(pages=2)])[0]

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

        # 3 chunks per pool, and every non-tail chunk contains 2 * L objects.
        calls = self.client.batch_put_from_ptr.call_args_list
        self.assertEqual(len(calls), 6)
        self.assertEqual([len(call.args[0]) for call in calls], [6, 6, 3, 6, 6, 3])

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
