import dataclasses
import unittest

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
        self.tree_ops = []
        self.reset_count = 0
        self.shutdown_count = 0

    def write_backup(
        self, request: BackupRequest, tree_ops: ExternalCacheTreeOps
    ) -> BackupResult:
        self.calls.append(("write_backup", request))
        self.tree_ops.append(tree_ops)
        return BackupResult(backed_up_tokens=11)

    def load_back(
        self, request: LoadBackRequest, tree_ops: ExternalCacheTreeOps
    ) -> LoadBackResult:
        self.calls.append(("load_back", request))
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
        self.assertTrue(controller.tree_ops)
        for tree_ops in controller.tree_ops:
            self.assertIsNot(tree_ops, tree)
            self.assertNotIsInstance(tree_ops, UnifiedTreeNode)


if __name__ == "__main__":
    unittest.main()
