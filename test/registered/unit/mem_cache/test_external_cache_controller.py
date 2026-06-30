import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.external_cache_controller import (
    BackupRequest,
    BackupResult,
    BaseExternalCacheController,
    ExternalCacheProgress,
    ExternalCacheTreeOps,
    LoadBackRequest,
    LoadBackResult,
    NoopExternalCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.test.test_utils import CustomTestCase


class _FakeFullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: True

    def redistribute_on_node_split(self, new_parent, child):
        return None

    def evict_component(
        self, node, target: EvictLayer = EvictLayer.DEVICE
    ) -> tuple[int, int]:
        return 0, 0

    def drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int]):
        return None

    def acquire_component_lock(self, node, result):
        return result

    def release_component_lock(self, node, params):
        return None


class _RecordingExternalCacheController(BaseExternalCacheController):
    def __init__(self):
        self.calls = []
        self.node_observations = []
        self.tree_ops = []
        self.reset_count = 0
        self.shutdown_count = 0

    def write_backup(
        self, request: BackupRequest, tree_ops: ExternalCacheTreeOps
    ) -> BackupResult:
        self.calls.append(("write_backup", request))
        self.node_observations.append(
            (
                request.node_id,
                tree_ops.contains_node(request.node_id),
                tree_ops.get_node_token_count(request.node_id),
            )
        )
        self.tree_ops.append(tree_ops)
        return BackupResult(backed_up_tokens=11)

    def load_back(
        self, request: LoadBackRequest, tree_ops: ExternalCacheTreeOps
    ) -> LoadBackResult:
        self.calls.append(("load_back", request))
        self.node_observations.append(
            (
                request.node_id,
                tree_ops.contains_node(request.node_id),
                tree_ops.get_node_token_count(request.node_id),
            )
        )
        self.tree_ops.append(tree_ops)
        return LoadBackResult(loaded=True)

    def begin_pending_loads(self) -> int:
        self.calls.append(("begin_pending_loads", None))
        return 17

    def poll(
        self, tree_ops: ExternalCacheTreeOps, *, wait: bool = False
    ) -> ExternalCacheProgress:
        self.calls.append(("poll", wait))
        self.tree_ops.append(tree_ops)
        return ExternalCacheProgress(completed_writes=1)

    def abort_request(
        self, request_id: str, tree_ops: ExternalCacheTreeOps
    ) -> None:
        self.calls.append(("abort_request", request_id))
        self.tree_ops.append(tree_ops)

    def reset(self) -> None:
        self.reset_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


def _build_tree() -> UnifiedRadixCache:
    return UnifiedRadixCache(params=_build_params())


def _build_params() -> CacheInitParams:
    params = CacheInitParams(
        req_to_token_pool=ReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
        ),
        token_to_kv_pool_allocator=None,
        page_size=1,
        disable=True,
        tree_components=(ComponentType.FULL,),
        component_registry_override={ComponentType.FULL: _FakeFullComponent},
    )
    return params


class TestExternalCacheController(CustomTestCase):
    def test_base_api_excludes_storage_specific_members(self):
        for name in (
            "write_storage",
            "prefetch_from_storage",
            "storage_enabled",
            "allow_host_tree_match",
        ):
            self.assertFalse(hasattr(BaseExternalCacheController, name), name)

    def test_public_requests_do_not_expose_scheduler_or_tree_objects(self):
        for request_type in (BackupRequest, LoadBackRequest):
            field_names = {field.name for field in dataclasses.fields(request_type)}
            self.assertNotIn("req", field_names)
            self.assertNotIn("node", field_names)
            self.assertIn("node_id", field_names)

    def test_unified_radix_cache_defaults_to_noop_external_controller(self):
        tree = _build_tree()

        self.assertIsInstance(
            tree.external_cache_controller, NoopExternalCacheController
        )
        self.assertEqual(tree.write_backup(tree.root_node), 0)
        self.assertFalse(tree.load_back(tree.root_node))
        self.assertEqual(tree.ready_to_load_host_cache(), 0)
        tree.check_hicache_events()

    def test_unified_radix_cache_delegates_to_installed_external_controller(self):
        tree = _build_tree()
        controller = _RecordingExternalCacheController()
        tree.external_cache_controller = controller

        self.assertEqual(tree.write_backup(tree.root_node, write_back=True), 11)
        self.assertTrue(tree.load_back(tree.root_node, mem_quota=3))
        self.assertEqual(tree.ready_to_load_host_cache(), 17)
        tree.check_hicache_events()

        self.assertEqual(
            controller.calls,
            [
                (
                    "write_backup",
                    BackupRequest(node_id=tree.root_node.id, write_back=True),
                ),
                ("load_back", LoadBackRequest(node_id=tree.root_node.id, mem_quota=3)),
                ("begin_pending_loads", None),
                ("poll", False),
            ],
        )
        self.assertEqual(
            controller.node_observations,
            [
                (tree.root_node.id, True, 0),
                (tree.root_node.id, True, 0),
            ],
        )
        self.assertTrue(controller.tree_ops)
        for tree_ops in controller.tree_ops:
            self.assertIsNot(tree_ops, tree)
            self.assertNotIsInstance(tree_ops, UnifiedTreeNode)

    def test_generic_external_cache_facade_names_keep_legacy_aliases(self):
        tree = _build_tree()

        with mock.patch.object(tree, "prefetch_from_storage") as prefetch:
            tree.prefetch_external_cache(
                "req", tree.root_node, [1, 2], "last_hash", ["prefix_hash"]
            )
            prefetch.assert_called_once_with(
                "req", tree.root_node, [1, 2], "last_hash", ["prefix_hash"]
            )

        with mock.patch.object(
            tree, "check_prefetch_progress", return_value=False
        ) as check_progress:
            self.assertFalse(tree.check_external_cache_prefetch_progress("req"))
            check_progress.assert_called_once_with("req")

        with mock.patch.object(tree, "terminate_prefetch") as terminate:
            tree.terminate_external_cache_prefetch("req")
            terminate.assert_called_once_with("req")

        with mock.patch.object(
            tree, "pop_prefetch_loaded_tokens", return_value=3
        ) as pop_loaded:
            self.assertEqual(tree.pop_external_cache_loaded_tokens("req"), 3)
            pop_loaded.assert_called_once_with("req")

        with mock.patch.object(tree, "release_aborted_request") as release:
            tree.release_aborted_external_cache_request("req")
            release.assert_called_once_with("req")

        with mock.patch.object(tree, "check_hicache_events") as check_events:
            tree.check_external_cache_events()
            check_events.assert_called_once_with()

    def test_ready_to_load_uses_hybrid_controller_begin_pending_loads(self):
        tree = _build_tree()
        controller = object.__new__(HybridCacheController)
        controller.begin_pending_loads = mock.Mock(return_value=23)
        tree.cache_controller = controller
        tree.external_cache_controller = controller

        self.assertEqual(tree.ready_to_load_host_cache(), 23)
        controller.begin_pending_loads.assert_called_once_with()

    def test_hybrid_controller_poll_drains_controller_owned_paths(self):
        controller = object.__new__(HybridCacheController)
        controller.enable_storage = True
        controller._poll_write_acks = mock.Mock(return_value=1)
        controller._poll_load_acks = mock.Mock(return_value=2)
        controller._drain_storage_control_queues = mock.Mock(return_value=3)
        tree = _build_tree()

        progress = controller.poll(tree.external_cache_tree_ops, wait=False)

        self.assertEqual(progress.completed_writes, 1)
        self.assertEqual(progress.completed_loads, 2)
        self.assertEqual(progress.completed_storage_ops, 3)
        controller._poll_write_acks.assert_called_once_with(tree)
        controller._poll_load_acks.assert_called_once_with(tree)
        controller._drain_storage_control_queues.assert_called_once_with(tree)

    def test_init_hicache_registers_hybrid_as_external_controller(self):
        params = _build_params()
        tree = UnifiedRadixCache(params=params)
        controller = object.__new__(HybridCacheController)
        controller.write_backup = mock.Mock(return_value=BackupResult(backed_up_tokens=5))
        controller.load_back = mock.Mock(return_value=LoadBackResult(loaded=True))
        controller.begin_pending_loads = mock.Mock(return_value=9)

        server_args = SimpleNamespace(
            hicache_io_backend="",
            hicache_mem_layout="",
            extra_metric_labels=None,
            hicache_storage_backend=None,
            hicache_storage_backend_extra_config=None,
            hicache_write_policy="write_through",
            hicache_storage_prefetch_policy="best_effort",
        )

        with mock.patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.attach_hybrid_pool_to_unified_cache",
            side_effect=lambda cache, *args, **kwargs: setattr(
                cache, "cache_controller", controller
            ),
        ):
            tree.init_hicache(server_args, params)

        self.assertIs(tree.external_cache_controller, controller)
        self.assertIs(controller._active_cache, tree)
        self.assertEqual(tree.write_backup(tree.root_node, write_back=True), 5)
        self.assertTrue(tree.load_back(tree.root_node, mem_quota=1))
        self.assertEqual(tree.ready_to_load_host_cache(), 9)
        controller.write_backup.assert_called_once()
        controller.load_back.assert_called_once()

    def test_check_hicache_events_uses_hybrid_controller_poll(self):
        tree = _build_tree()
        controller = object.__new__(HybridCacheController)
        controller.poll = mock.Mock(return_value=ExternalCacheProgress())
        tree.cache_controller = controller
        tree.external_cache_controller = controller

        tree.check_hicache_events()

        controller.poll.assert_called_once_with(tree.external_cache_tree_ops)

    def test_tree_ops_resolves_nodes_by_id_without_returning_nodes(self):
        tree = _build_tree()
        child = UnifiedTreeNode((ComponentType.FULL,))
        child.parent = tree.root_node
        child.component_data[ComponentType.FULL].value = [1, 2, 3]
        tree.root_node.children[0] = child

        tree_ops = tree.external_cache_tree_ops

        self.assertIsInstance(tree_ops, ExternalCacheTreeOps)
        self.assertTrue(tree_ops.contains_node(tree.root_node.id))
        self.assertTrue(tree_ops.contains_node(child.id))
        self.assertEqual(tree_ops.get_node_token_count(child.id), 3)
        self.assertFalse(tree_ops.contains_node(-1))
        with self.assertRaises(KeyError):
            tree_ops.get_node_token_count(-1)

    def test_write_through_state_is_owned_by_hybrid_controller(self):
        tree = _build_tree()
        controller = object.__new__(HybridCacheController)
        controller.ongoing_write_through = {}
        controller.ongoing_load_back = {}
        controller.ongoing_backup = {}
        controller.prefetch_loaded_tokens_by_reqid = {}
        controller.ongoing_prefetch = {}
        tree.cache_controller = controller

        self.assertFalse(hasattr(tree, "ongoing_write_through"))
        self.assertFalse(hasattr(tree, "ongoing_load_back"))
        self.assertFalse(hasattr(tree, "ongoing_backup"))
        self.assertFalse(hasattr(tree, "prefetch_loaded_tokens_by_reqid"))
        self.assertFalse(hasattr(tree, "ongoing_prefetch"))

        controller.track_write_through_node(tree.root_node, None)

        pending = controller.ongoing_write_through[tree.root_node.id]
        self.assertIs(pending.node, tree.root_node)
        self.assertEqual(pending.publish_nodes, [tree.root_node])
        self.assertEqual(
            tree.root_node.write_through_pending_id,
            tree.root_node.id,
        )

        left = UnifiedTreeNode((ComponentType.FULL,))
        right = UnifiedTreeNode((ComponentType.FULL,))
        controller.replace_pending_write_through_node(tree.root_node, [left, right])

        updated = controller.ongoing_write_through[tree.root_node.id]
        self.assertEqual(updated.publish_nodes, [left, right])
        self.assertEqual(left.write_through_pending_id, tree.root_node.id)
        self.assertEqual(right.write_through_pending_id, tree.root_node.id)

    def test_write_through_ack_completion_runs_in_hybrid_controller(self):
        tree = _build_tree()
        controller = object.__new__(HybridCacheController)
        controller.ongoing_write_through = {}
        controller.ongoing_backup = {}
        tree.cache_controller = controller
        tree.enable_storage = True
        lock_params = object()

        controller.track_write_through_node(tree.root_node, lock_params)
        left = UnifiedTreeNode((ComponentType.FULL,))
        right = UnifiedTreeNode((ComponentType.FULL,))
        controller.replace_pending_write_through_node(tree.root_node, [left, right])

        with (
            mock.patch.object(tree, "_record_store_event") as record_event,
            mock.patch.object(tree, "dec_lock_ref") as dec_lock_ref,
            mock.patch.object(controller, "write_backup_storage") as backup_storage,
        ):
            tree._finish_write_through_ack(tree.root_node.id)

        self.assertEqual(controller.ongoing_write_through, {})
        self.assertIsNone(left.write_through_pending_id)
        self.assertIsNone(right.write_through_pending_id)
        record_event.assert_has_calls(
            [
                mock.call(left, medium=StorageMedium.CPU),
                mock.call(right, medium=StorageMedium.CPU),
            ]
        )
        dec_lock_ref.assert_called_once_with(tree.root_node, lock_params)
        backup_storage.assert_has_calls(
            [
                mock.call(left),
                mock.call(right),
            ]
        )

    def test_load_back_ack_completion_runs_in_hybrid_controller(self):
        tree = _build_tree()
        controller = object.__new__(HybridCacheController)
        controller.ongoing_write_through = {}
        controller.ongoing_load_back = {}
        controller.ongoing_backup = {}
        tree.cache_controller = controller
        lock_params = object()
        host_lock_params = object()
        controller.ongoing_load_back[tree.root_node.id] = (
            tree.root_node,
            lock_params,
            host_lock_params,
        )

        with (
            mock.patch.object(tree, "dec_lock_ref") as dec_lock_ref,
            mock.patch.object(tree, "dec_host_lock_ref") as dec_host_lock_ref,
        ):
            tree._finish_load_back_ack(tree.root_node.id)

        self.assertEqual(controller.ongoing_load_back, {})
        dec_lock_ref.assert_called_once_with(tree.root_node, lock_params)
        dec_host_lock_ref.assert_called_once_with(tree.root_node, host_lock_params)

    def test_prefetch_progress_accounting_runs_in_hybrid_controller(self):
        tree = _build_tree()
        controller = object.__new__(HybridCacheController)
        controller.prefetch_loaded_tokens_by_reqid = {}
        controller.ongoing_prefetch = {}
        controller.prefetch_tokens_occupied = 4
        controller.mem_pool_host = mock.Mock()
        controller.append_host_mem_release = mock.Mock()
        host_indices = torch.arange(4)
        lock_params = object()
        operation = mock.Mock()
        controller.ongoing_prefetch["req"] = (
            tree.root_node,
            [1, 2, 3, 4],
            host_indices,
            operation,
            lock_params,
            {},
        )

        release_host_lock = mock.Mock()
        occupied = controller.finish_prefetch_progress(
            "req",
            completed_tokens=3,
            min_completed_tokens=2,
            matched_tokens=1,
            loaded_tokens=1,
            release_host_lock=release_host_lock,
        )

        self.assertEqual(occupied, 0)
        self.assertEqual(controller.ongoing_prefetch, {})
        self.assertEqual(controller.prefetch_loaded_tokens_by_reqid, {"req": 1})
        freed = controller.mem_pool_host.free.call_args.args[0]
        self.assertTrue(torch.equal(freed, host_indices[:1]))
        released = controller.append_host_mem_release.call_args.args[0]
        self.assertTrue(torch.equal(released, host_indices[2:3]))
        release_host_lock.assert_called_once_with(tree.root_node, lock_params)

    def test_prefetch_wrappers_delegate_abort_and_loaded_accounting(self):
        tree = _build_tree()
        controller = object.__new__(HybridCacheController)
        controller.prefetch_loaded_tokens_by_reqid = {"req": 7}
        controller.ongoing_prefetch = {}
        controller.prefetch_tokens_occupied = 4
        controller.terminate_prefetch = mock.Mock(return_value=(2, []))
        controller.append_host_mem_release = mock.Mock()
        tree.cache_controller = controller
        tree.external_cache_controller = controller
        host_indices = torch.arange(4)
        lock_params = object()
        operation = mock.Mock()
        operation.host_indices = host_indices
        controller.ongoing_prefetch["req"] = (
            tree.root_node,
            [1, 2, 3, 4],
            host_indices,
            operation,
            lock_params,
            {},
        )

        with (
            mock.patch.object(tree, "_barrier_attn_groups") as barrier,
            mock.patch.object(tree, "dec_host_lock_ref") as dec_host_lock_ref,
        ):
            tree.release_aborted_external_cache_request("req")

        self.assertEqual(controller.prefetch_loaded_tokens_by_reqid, {})
        self.assertEqual(controller.ongoing_prefetch, {})
        self.assertEqual(controller.prefetch_tokens_occupied, 0)
        barrier.assert_called_once_with()
        dec_host_lock_ref.assert_called_once_with(tree.root_node, lock_params)
        released = controller.append_host_mem_release.call_args.kwargs["host_indices"]
        self.assertTrue(torch.equal(released, host_indices[:2]))
        self.assertEqual(
            controller.append_host_mem_release.call_args.kwargs["extra_pools"],
            [],
        )

        operation = mock.Mock()
        operation.host_indices = host_indices
        controller.ongoing_prefetch["req"] = (
            tree.root_node,
            [1],
            host_indices,
            operation,
            lock_params,
            {},
        )
        tree.terminate_prefetch("req")
        operation.mark_terminate.assert_called_once_with()

        controller.prefetch_loaded_tokens_by_reqid["req"] = 5
        self.assertEqual(tree.pop_prefetch_loaded_tokens("req"), 5)


if __name__ == "__main__":
    unittest.main()
