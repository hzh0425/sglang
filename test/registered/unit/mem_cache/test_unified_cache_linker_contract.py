from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertResult
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _SWAAllocator:
    def __init__(self):
        self.freed = []
        self.mappings = []

    def free(self, value):
        self.freed.append(value.clone())

    def set_full_to_swa_mapping(self, full, swa):
        self.mappings.append((full.clone(), swa.clone()))


def test_insert_result_merges_adopted_ranges():
    result = InsertResult(prefix_len=0, adopted_ranges={})

    result.record_adopted_range(ComponentType.FULL, 4, 8)
    result.record_adopted_range(ComponentType.FULL, 8, 12)
    result.record_adopted_range(ComponentType.FULL, 16, 20)

    assert result.adopted_ranges == {ComponentType.FULL: [(4, 12), (16, 20)]}


def test_swa_linker_prepare_updates_request_kv_ownership():
    swa_allocator = _SWAAllocator()
    allocator = SimpleNamespace(
        swa_attn_allocator=swa_allocator,
        set_full_to_swa_mapping=swa_allocator.set_full_to_swa_mapping,
    )
    component = SWAComponent.__new__(SWAComponent)
    component.cache = SimpleNamespace(
        page_size=1,
        token_to_kv_pool_allocator=allocator,
    )
    component.sliding_window_size = 2
    req = SimpleNamespace(
        kv=SimpleNamespace(
            cache_protected_len=0,
            kv_allocated_len=0,
            swa_evicted_seqlen=0,
        )
    )
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1, 2, 3, 4]))
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([20, 21]))

    component.update_external_linker_load(
        ExternalLinkerLoadPhase.PREPARE,
        req,
        full,
        swa,
        prefix_len=4,
    )

    mapped_full, mapped_swa = swa_allocator.mappings[0]
    assert mapped_full.tolist() == [3, 4]
    assert mapped_swa.tolist() == [20, 21]
    assert req.kv.kv_allocated_len == 4
    assert req.kv.swa_evicted_seqlen == 2


def test_swa_linker_abort_releases_allocated_slots():
    swa_allocator = _SWAAllocator()
    component = SWAComponent.__new__(SWAComponent)
    component.cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            swa_attn_allocator=swa_allocator,
        )
    )
    transfer = PoolTransfer(
        name=PoolName.SWA,
        device_indices=torch.tensor([30, 31]),
    )

    result = component.update_external_linker_load(
        ExternalLinkerLoadPhase.ABORT,
        SimpleNamespace(),
        PoolTransfer(name=PoolName.KV),
        transfer,
        prefix_len=0,
    )

    assert result is None
    assert swa_allocator.freed[0].tolist() == [30, 31]
