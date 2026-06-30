import dataclasses
import unittest
from unittest import mock

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
    return UnifiedRadixCache(params=params)


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

    def test_ready_to_load_uses_hybrid_controller_begin_pending_loads(self):
        tree = _build_tree()
        controller = object.__new__(HybridCacheController)
        controller.begin_pending_loads = mock.Mock(return_value=23)
        tree.cache_controller = controller

        self.assertEqual(tree.ready_to_load_host_cache(), 23)
        controller.begin_pending_loads.assert_called_once_with()

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
        tree.cache_controller = controller

        self.assertIs(tree.ongoing_write_through, controller.ongoing_write_through)
        self.assertIs(tree.ongoing_load_back, controller.ongoing_load_back)
        self.assertIs(tree.ongoing_backup, controller.ongoing_backup)

        tree._track_write_through_node(tree.root_node, None)

        pending = controller.ongoing_write_through[tree.root_node.id]
        self.assertIs(pending.node, tree.root_node)
        self.assertEqual(pending.publish_nodes, [tree.root_node])
        self.assertEqual(
            tree.root_node.write_through_pending_id,
            tree.root_node.id,
        )

        left = UnifiedTreeNode((ComponentType.FULL,))
        right = UnifiedTreeNode((ComponentType.FULL,))
        tree._replace_pending_write_through_node(tree.root_node, [left, right])

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

        tree._track_write_through_node(tree.root_node, lock_params)
        left = UnifiedTreeNode((ComponentType.FULL,))
        right = UnifiedTreeNode((ComponentType.FULL,))
        tree._replace_pending_write_through_node(tree.root_node, [left, right])

        with (
            mock.patch.object(tree, "_record_store_event") as record_event,
            mock.patch.object(tree, "dec_lock_ref") as dec_lock_ref,
            mock.patch.object(tree, "write_backup_storage") as backup_storage,
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


if __name__ == "__main__":
    unittest.main()
