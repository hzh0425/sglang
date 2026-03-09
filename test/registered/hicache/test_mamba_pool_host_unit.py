import os
import threading
import unittest

import torch

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from sglang.srt.managers.cache_controller import AuxiliaryTransfer, HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MambaPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.mem_cache.memory_pool_host import (
    ALLOC_MEMORY_FUNCS,
    HostPoolGroup,
    MHATokenToKVPoolHost,
    MambaPoolHost,
    PoolEntry,
    alloc_with_pin_memory,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")


def _make_mamba_pool(size: int, num_layers: int = 3) -> MambaPool:
    shape = Mamba2StateShape(
        conv=[(4, 2)],
        temporal=(2, 2, 2),
        intermediate_size=4,
        conv_dim=4,
        ssm_state_size=2,
        num_heads=2,
        head_dim=2,
        state_size=2,
        conv_kernel=3,
    )
    cache_params = Mamba2CacheParams(
        shape=shape,
        layers=list(range(num_layers)),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
    )
    return MambaPool(
        size=size,
        spec_state_size=size,
        cache_params=cache_params,
        device="cuda",
        enable_memory_saver=False,
    )


class TestMambaPoolHostTransfer(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Mamba host transfer tests.")
        if is_npu() or is_xpu():
            self.skipTest("Mamba host transfer tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    def _run_roundtrip(self, io_backend: str):
        device_pool = _make_mamba_pool(size=4, num_layers=3)
        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        if io_backend == "kernel":
            ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            host_pool = MambaPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                pin_memory=io_backend == "kernel",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        device_indices = torch.tensor([1, 3], device="cuda", dtype=torch.int64)
        host_indices = torch.tensor(
            [0, 2],
            device="cuda" if io_backend == "kernel" else "cpu",
            dtype=torch.int64,
        )

        for layer_id in range(device_pool.num_mamba_layers):
            for conv_idx, conv_state in enumerate(device_pool.mamba_cache.conv):
                value = (
                    torch.arange(
                        conv_state[layer_id, device_indices].numel(),
                        device="cuda",
                        dtype=conv_state.dtype,
                    ).view_as(conv_state[layer_id, device_indices])
                    + (layer_id + 1) * 10
                    + conv_idx
                )
                conv_state[layer_id, device_indices] = value
            temporal = device_pool.mamba_cache.temporal
            temporal[layer_id, device_indices] = (
                torch.arange(
                    temporal[layer_id, device_indices].numel(),
                    device="cuda",
                    dtype=temporal.dtype,
                ).view_as(temporal[layer_id, device_indices])
                + (layer_id + 1) * 100
            )

        expected_conv = [
            conv[:, device_indices].detach().cpu().clone()
            for conv in device_pool.mamba_cache.conv
        ]
        expected_temporal = (
            device_pool.mamba_cache.temporal[:, device_indices].detach().cpu().clone()
        )

        host_pool.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )

        for conv_state in device_pool.mamba_cache.conv:
            conv_state[:, device_indices] = 0
        device_pool.mamba_cache.temporal[:, device_indices] = 0

        for layer_id in range(device_pool.num_mamba_layers):
            host_pool.load_to_device_per_layer(
                device_pool, host_indices, device_indices, layer_id, io_backend
            )

        for conv_idx, conv_state in enumerate(device_pool.mamba_cache.conv):
            self.assertTrue(
                torch.equal(
                    conv_state[:, device_indices].detach().cpu(), expected_conv[conv_idx]
                )
            )
        self.assertTrue(
            torch.equal(
                device_pool.mamba_cache.temporal[:, device_indices].detach().cpu(),
                expected_temporal,
            )
        )

    def test_roundtrip_direct(self):
        self._run_roundtrip(io_backend="direct")

    def test_roundtrip_kernel(self):
        self._run_roundtrip(io_backend="kernel")

    def test_page_serialization_preserves_mixed_dtypes(self):
        device_pool = _make_mamba_pool(size=4, num_layers=3)
        host_pool = MambaPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            pin_memory=False,
        )

        host_index = 1
        for local_layer in range(device_pool.num_mamba_layers):
            host_pool.temporal_buffer[local_layer, host_index] = (
                torch.arange(
                    host_pool.temporal_buffer[local_layer, host_index].numel(),
                    dtype=host_pool.temporal_dtype,
                ).view_as(host_pool.temporal_buffer[local_layer, host_index])
                + local_layer
            )
            for conv_idx, conv_buffer in enumerate(host_pool.conv_buffer):
                conv_buffer[local_layer, host_index] = (
                    torch.arange(
                        conv_buffer[local_layer, host_index].numel(),
                        dtype=conv_buffer.dtype,
                    ).view_as(conv_buffer[local_layer, host_index])
                    + 10 * (conv_idx + 1)
                    + local_layer
                )

        expected_temporal = host_pool.temporal_buffer[:, host_index].clone()
        expected_conv = [conv[:, host_index].clone() for conv in host_pool.conv_buffer]

        serialized_page = host_pool.get_data_page(host_index, flat=True)
        self.assertEqual(serialized_page.dtype, torch.uint8)
        self.assertEqual(serialized_page.numel(), host_pool.size_per_token)

        host_pool.temporal_buffer[:, host_index] = 0
        for conv_buffer in host_pool.conv_buffer:
            conv_buffer[:, host_index] = 0

        host_pool.set_from_flat_data_page(host_index, serialized_page)

        self.assertTrue(torch.equal(host_pool.temporal_buffer[:, host_index], expected_temporal))
        for conv_idx, conv_buffer in enumerate(host_pool.conv_buffer):
            self.assertTrue(torch.equal(conv_buffer[:, host_index], expected_conv[conv_idx]))


class TestHostPoolGroupLayerDispatch(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for HostPoolGroup tests.")

    def test_group_dispatches_kv_and_mamba_by_model_layer(self):
        kv_device_pool = MHATokenToKVPool(
            size=8,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=4,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        kv_host_pool = MHATokenToKVPoolHost(
            kv_device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )

        mamba_device_pool = _make_mamba_pool(size=4, num_layers=3)
        mamba_host_pool = MambaPoolHost(
            mamba_device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            pin_memory=False,
        )

        kv_host_indices = torch.tensor([0], dtype=torch.int64)
        kv_device_indices = torch.tensor([2], device="cuda", dtype=torch.int64)
        mamba_host_indices = torch.tensor([1], dtype=torch.int64)
        mamba_device_indices = torch.tensor([3], device="cuda", dtype=torch.int64)

        kv_host_pool.k_buffer[0][kv_host_indices] = 7
        kv_host_pool.v_buffer[0][kv_host_indices] = 9
        for local_layer in range(3):
            for conv_state in mamba_host_pool.conv_buffer:
                conv_state[local_layer, mamba_host_indices] = local_layer + 1
            mamba_host_pool.temporal_buffer[local_layer, mamba_host_indices] = (
                local_layer + 1
            ) * 10

        def kv_index_resolver(ctx: dict):
            entry_ctx = ctx["entries"]["kv"]
            return entry_ctx["host_indices"], entry_ctx["device_indices"]

        def mamba_index_resolver(ctx: dict):
            entry_ctx = ctx["entries"]["mamba"]
            return entry_ctx["host_indices"], entry_ctx["device_indices"]

        model_layer_ids = [0, 1, 2, 3]
        kv_map = {1: 0}
        mamba_map = {0: 0, 2: 1, 3: 2}

        group = HostPoolGroup(
            [
                PoolEntry(
                    name="kv",
                    host_pool=kv_host_pool,
                    device_pool=kv_device_pool,
                    index_resolver=kv_index_resolver,
                    layer_mapper=lambda model_layer_local_id: kv_map.get(
                        model_layer_ids[model_layer_local_id]
                    ),
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name="mamba",
                    host_pool=mamba_host_pool,
                    device_pool=mamba_device_pool,
                    index_resolver=mamba_index_resolver,
                    layer_mapper=lambda model_layer_local_id: mamba_map.get(
                        model_layer_ids[model_layer_local_id]
                    ),
                ),
            ]
        )
        group.set_transfer_context(
            {
                "entries": {
                    "mamba": {
                        "host_indices": mamba_host_indices,
                        "device_indices": mamba_device_indices,
                    }
                }
            }
        )
        try:
            for layer_id in range(len(model_layer_ids)):
                group.load_to_device_per_layer(
                    kv_device_pool,
                    kv_host_indices,
                    kv_device_indices,
                    layer_id,
                    "direct",
                )
        finally:
            group.clear_transfer_context()

        self.assertTrue(
            torch.equal(
                kv_device_pool.k_buffer[0][kv_device_indices].detach().cpu(),
                kv_host_pool.k_buffer[0][kv_host_indices],
            )
        )
        self.assertTrue(
            torch.equal(
                kv_device_pool.v_buffer[0][kv_device_indices].detach().cpu(),
                kv_host_pool.v_buffer[0][kv_host_indices],
            )
        )
        for local_layer in range(3):
            self.assertTrue(
                torch.equal(
                    mamba_device_pool.mamba_cache.temporal[
                        local_layer, mamba_device_indices
                    ]
                    .detach()
                    .cpu(),
                    mamba_host_pool.temporal_buffer[local_layer, mamba_host_indices],
                )
            )

    def test_group_page_serialization_includes_mamba_pages(self):
        kv_device_pool = MHATokenToKVPool(
            size=8,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=4,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        kv_host_pool = MHATokenToKVPoolHost(
            kv_device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        mamba_device_pool = _make_mamba_pool(size=4, num_layers=3)
        mamba_host_pool = MambaPoolHost(
            mamba_device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            pin_memory=False,
        )

        kv_host_pool.k_buffer[0][0] = 3
        kv_host_pool.v_buffer[0][0] = 5
        for local_layer in range(3):
            mamba_host_pool.temporal_buffer[local_layer, 0] = local_layer + 11
            for conv_idx, conv_buffer in enumerate(mamba_host_pool.conv_buffer):
                conv_buffer[local_layer, 0] = local_layer + conv_idx + 21

        group = HostPoolGroup(
            [
                PoolEntry(
                    name="kv",
                    host_pool=kv_host_pool,
                    device_pool=kv_device_pool,
                    index_resolver=lambda ctx: (
                        ctx["entries"]["kv"]["host_indices"],
                        None,
                    ),
                    layer_mapper=lambda _: None,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name="mamba",
                    host_pool=mamba_host_pool,
                    device_pool=mamba_device_pool,
                    index_resolver=lambda ctx: (
                        ctx["entries"]["mamba"]["host_indices"],
                        None,
                    ),
                    layer_mapper=lambda _: None,
                ),
            ]
        )
        group.set_transfer_context(
            {
                "entries": {
                    "kv": {"host_indices": torch.tensor([0], dtype=torch.int64)},
                    "mamba": {"host_indices": torch.tensor([0], dtype=torch.int64)},
                }
            }
        )

        serialized_page = group.get_data_page(0, flat=True)
        dummy_page = group.get_dummy_flat_data_page()
        transfer_view = group.get_transfer_view(torch.tensor([0], dtype=torch.int64))
        self.assertEqual(serialized_page.dtype, torch.uint8)
        self.assertEqual(serialized_page.numel(), dummy_page.numel())
        self.assertIn("kv", transfer_view.subviews)
        self.assertIn("mamba", transfer_view.subviews)

        kv_host_pool.k_buffer[0][0] = 0
        kv_host_pool.v_buffer[0][0] = 0
        mamba_host_pool.temporal_buffer[:, 0] = 0
        for conv_buffer in mamba_host_pool.conv_buffer:
            conv_buffer[:, 0] = 0

        group.set_from_flat_data_page(0, serialized_page)
        self.assertTrue(
            torch.equal(
                kv_host_pool.k_buffer[0][0],
                torch.full_like(kv_host_pool.k_buffer[0][0], 3),
            )
        )
        self.assertTrue(
            torch.equal(
                kv_host_pool.v_buffer[0][0],
                torch.full_like(kv_host_pool.v_buffer[0][0], 5),
            )
        )
        for local_layer in range(3):
            self.assertTrue(
                torch.equal(
                    mamba_host_pool.temporal_buffer[local_layer, 0],
                    torch.full_like(mamba_host_pool.temporal_buffer[local_layer, 0], local_layer + 11),
                )
            )


class TestHiCacheControllerHybridRoundTrip(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for HiCache hybrid offload tests.")
        if is_npu() or is_xpu():
            self.skipTest("Hybrid offload tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    def test_controller_roundtrip_restores_kv_and_mamba(self):
        model_layer_ids = [0, 1, 2, 3]
        full_attention_layer_ids = [1]

        mamba_cache_params = Mamba2CacheParams(
            shape=Mamba2StateShape(
                conv=[(4, 2)],
                temporal=(2, 2, 2),
                intermediate_size=4,
                conv_dim=4,
                ssm_state_size=2,
                num_heads=2,
                head_dim=2,
                state_size=2,
                conv_kernel=3,
            ),
            layers=[0, 2, 3],
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        )
        req_pool = HybridReqToTokenPool(
            size=4,
            mamba_size=4,
            mamba_spec_state_size=4,
            max_context_len=8,
            device="cuda",
            enable_memory_saver=False,
            cache_params=mamba_cache_params,
            enable_mamba_extra_buffer=False,
        )
        kv_pool = HybridLinearKVPool(
            size=8,
            dtype=torch.float16,
            page_size=1,
            head_num=1,
            head_dim=4,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device="cuda",
            enable_memory_saver=False,
            mamba_pool=req_pool.mamba_pool,
        )
        kv_pool.set_model_layer_id_mapping(model_layer_ids)
        req_pool.set_model_layer_id_mapping(model_layer_ids)

        kv_host_pool = MHATokenToKVPoolHost(
            kv_pool.full_kv_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        mamba_host_pool = MambaPoolHost(
            req_pool.mamba_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            pin_memory=False,
        )

        def kv_index_resolver(ctx: dict):
            entry_ctx = ctx["entries"]["kv"]
            return entry_ctx["host_indices"], entry_ctx["device_indices"]

        def mamba_index_resolver(ctx: dict):
            entry_ctx = ctx["entries"]["mamba"]
            return entry_ctx["host_indices"], entry_ctx["device_indices"]

        kv_map = {1: 0}
        mamba_map = dict(req_pool.mamba_map)
        host_group = HostPoolGroup(
            [
                PoolEntry(
                    name="kv",
                    host_pool=kv_host_pool,
                    device_pool=kv_pool.full_kv_pool,
                    index_resolver=kv_index_resolver,
                    layer_mapper=lambda model_layer_local_id: kv_map.get(
                        model_layer_ids[model_layer_local_id]
                    ),
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name="mamba",
                    host_pool=mamba_host_pool,
                    device_pool=req_pool.mamba_pool,
                    index_resolver=mamba_index_resolver,
                    layer_mapper=lambda model_layer_local_id: mamba_map.get(
                        model_layer_ids[model_layer_local_id]
                    ),
                ),
            ]
        )

        allocator = TokenToKVPoolAllocator(
            size=8,
            dtype=torch.float16,
            device="cuda",
            kvcache=kv_pool,
            need_sort=False,
        )
        controller = HiCacheController(
            allocator,
            host_group,
            page_size=1,
            tp_group=None,
            load_cache_event=threading.Event(),
            write_policy="write_through",
            io_backend="direct",
            transfer_layer_num=len(model_layer_ids),
        )
        req_pool.register_layer_transfer_counter(controller.layer_done_counter)

        kv_device_indices = torch.tensor([2], device="cuda", dtype=torch.int64)
        mamba_device_indices = torch.tensor([1], device="cuda", dtype=torch.int64)
        mamba_host_indices = torch.tensor([0], dtype=torch.int64)

        expected_k = torch.arange(4, device="cuda", dtype=torch.float16).view(1, 1, 4)
        expected_v = (expected_k + 10).clone()
        kv_pool.get_key_buffer(1)[kv_device_indices] = expected_k
        kv_pool.get_value_buffer(1)[kv_device_indices] = expected_v

        expected_conv = {}
        expected_temporal = {}
        for model_layer_id, local_layer_id in req_pool.mamba_map.items():
            layer_conv_expected = []
            for conv_idx, conv_state in enumerate(req_pool.mamba_pool.mamba_cache.conv):
                value = (
                    torch.arange(
                        conv_state[local_layer_id, mamba_device_indices].numel(),
                        device="cuda",
                        dtype=conv_state.dtype,
                    ).view_as(conv_state[local_layer_id, mamba_device_indices])
                    + 10 * (local_layer_id + 1)
                    + conv_idx
                )
                conv_state[local_layer_id, mamba_device_indices] = value
                layer_conv_expected.append(value.detach().cpu().clone())
            expected_conv[model_layer_id] = layer_conv_expected

            temporal_value = (
                torch.arange(
                    req_pool.mamba_pool.mamba_cache.temporal[
                        local_layer_id, mamba_device_indices
                    ].numel(),
                    device="cuda",
                    dtype=req_pool.mamba_pool.mamba_cache.temporal.dtype,
                ).view_as(
                    req_pool.mamba_pool.mamba_cache.temporal[
                        local_layer_id, mamba_device_indices
                    ]
                )
                + 100 * (local_layer_id + 1)
            )
            req_pool.mamba_pool.mamba_cache.temporal[
                local_layer_id, mamba_device_indices
            ] = temporal_value
            expected_temporal[model_layer_id] = temporal_value.detach().cpu().clone()

        kv_host_indices = controller.write(
            kv_device_indices,
            auxiliary_transfers=[
                AuxiliaryTransfer(
                    name="mamba",
                    host_indices=mamba_host_indices,
                    device_indices=mamba_device_indices,
                )
            ],
            node_id=1,
        )
        self.assertIsNotNone(kv_host_indices)
        controller.ack_write_queue[-1].finish_event.synchronize()

        kv_pool.get_key_buffer(1)[kv_device_indices] = 0
        kv_pool.get_value_buffer(1)[kv_device_indices] = 0
        for conv_state in req_pool.mamba_pool.mamba_cache.conv:
            conv_state[:, mamba_device_indices] = 0
        req_pool.mamba_pool.mamba_cache.temporal[:, mamba_device_indices] = 0

        reloaded_mamba_device_indices = req_pool.mamba_pool.alloc(1)
        self.assertIsNotNone(reloaded_mamba_device_indices)
        reloaded_kv_device_indices = controller.load(
            kv_host_indices,
            auxiliary_transfers=lambda _device_indices: [
                AuxiliaryTransfer(
                    name="mamba",
                    host_indices=mamba_host_indices,
                    device_indices=reloaded_mamba_device_indices,
                )
            ],
            node_id=2,
        )
        self.assertIsNotNone(reloaded_kv_device_indices)
        producer_id = controller.start_loading()
        self.assertGreaterEqual(producer_id, 0)
        controller.layer_done_counter.set_consumer(producer_id)
        controller.ack_load_queue[-1].finish_event.synchronize()

        self.assertTrue(
            torch.equal(
                kv_pool.get_key_buffer(1)[reloaded_kv_device_indices].detach().cpu(),
                expected_k.detach().cpu(),
            )
        )
        self.assertTrue(
            torch.equal(
                kv_pool.get_value_buffer(1)[reloaded_kv_device_indices]
                .detach()
                .cpu(),
                expected_v.detach().cpu(),
            )
        )

        for model_layer_id in [0, 2, 3]:
            state = req_pool.mamba2_layer_cache(model_layer_id)
            self.assertTrue(
                torch.equal(
                    state.temporal[reloaded_mamba_device_indices].detach().cpu(),
                    expected_temporal[model_layer_id],
                )
            )
            for conv_idx, conv_state in enumerate(state.conv):
                self.assertTrue(
                    torch.equal(
                        conv_state[reloaded_mamba_device_indices].detach().cpu(),
                        expected_conv[model_layer_id][conv_idx],
                    )
                )


class TestHiMambaRadixCacheRoundTrip(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for HiMambaRadixCache tests.")
        if is_npu() or is_xpu():
            self.skipTest("HiMambaRadixCache tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    def test_himamba_tree_write_backup_and_load_back(self):
        set_global_server_args_for_scheduler(
            ServerArgs(
                model_path="dummy",
                page_size=1,
                enable_hierarchical_cache=True,
                hicache_mem_layout="layer_first",
                hicache_io_backend="direct",
                hicache_ratio=2.0,
                hicache_size=0,
                hicache_write_policy="write_through_selective",
                disable_hicache_numa_detect=True,
            )
        )

        model_layer_ids = [0, 1, 2, 3]
        mamba_cache_params = Mamba2CacheParams(
            shape=Mamba2StateShape(
                conv=[(4, 2)],
                temporal=(2, 2, 2),
                intermediate_size=4,
                conv_dim=4,
                ssm_state_size=2,
                num_heads=2,
                head_dim=2,
                state_size=2,
                conv_kernel=3,
            ),
            layers=[0, 2, 3],
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        )
        req_pool = HybridReqToTokenPool(
            size=8,
            mamba_size=8,
            mamba_spec_state_size=8,
            max_context_len=8,
            device="cuda",
            enable_memory_saver=False,
            cache_params=mamba_cache_params,
            enable_mamba_extra_buffer=False,
        )
        kv_pool = HybridLinearKVPool(
            size=16,
            dtype=torch.float16,
            page_size=1,
            head_num=1,
            head_dim=4,
            full_attention_layer_ids=[1],
            enable_kvcache_transpose=False,
            device="cuda",
            enable_memory_saver=False,
            mamba_pool=req_pool.mamba_pool,
        )
        kv_pool.set_model_layer_id_mapping(model_layer_ids)
        req_pool.set_model_layer_id_mapping(model_layer_ids)
        allocator = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float16,
            device="cuda",
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
        tree = HiMambaRadixCache(
            params=params,
            server_args=ServerArgs(
                model_path="dummy",
                page_size=1,
                enable_hierarchical_cache=True,
                hicache_mem_layout="layer_first",
                hicache_io_backend="direct",
                hicache_ratio=2.0,
                hicache_size=0,
                hicache_write_policy="write_through_selective",
                disable_hicache_numa_detect=True,
            ),
        )
        tree.load_back_threshold = 1

        kv_indices = allocator.alloc(3)
        self.assertIsNotNone(kv_indices)
        mamba_index = req_pool.mamba_pool.alloc(1)
        self.assertIsNotNone(mamba_index)

        expected_k = torch.arange(12, device="cuda", dtype=torch.float16).view(3, 1, 4)
        expected_v = (expected_k + 50).clone()
        kv_pool.get_key_buffer(1)[kv_indices] = expected_k
        kv_pool.get_value_buffer(1)[kv_indices] = expected_v

        expected_temporal = {}
        expected_conv = {}
        for model_layer_id, local_layer_id in req_pool.mamba_map.items():
            temporal_value = (
                torch.arange(
                    req_pool.mamba_pool.mamba_cache.temporal[
                        local_layer_id, mamba_index
                    ].numel(),
                    device="cuda",
                    dtype=req_pool.mamba_pool.mamba_cache.temporal.dtype,
                ).view_as(
                    req_pool.mamba_pool.mamba_cache.temporal[
                        local_layer_id, mamba_index
                    ]
                )
                + 100 * (local_layer_id + 1)
            )
            req_pool.mamba_pool.mamba_cache.temporal[local_layer_id, mamba_index] = (
                temporal_value
            )
            expected_temporal[model_layer_id] = temporal_value.detach().cpu().clone()
            layer_conv = []
            for conv_idx, conv_state in enumerate(req_pool.mamba_pool.mamba_cache.conv):
                value = (
                    torch.arange(
                        conv_state[local_layer_id, mamba_index].numel(),
                        device="cuda",
                        dtype=conv_state.dtype,
                    ).view_as(conv_state[local_layer_id, mamba_index])
                    + 10 * (local_layer_id + 1)
                    + conv_idx
                )
                conv_state[local_layer_id, mamba_index] = value
                layer_conv.append(value.detach().cpu().clone())
            expected_conv[model_layer_id] = layer_conv

        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3], None),
                value=kv_indices,
                mamba_value=mamba_index,
            )
        )
        node = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3], None))).last_device_node

        self.assertEqual(tree.write_backup(node), len(kv_indices))
        tree.writing_check(write_back=True)
        self.assertIsNotNone(node.host_value)
        self.assertIsNotNone(node.mamba_host_value)

        self.assertEqual(tree._evict_to_host(node), len(kv_indices))
        if tree.mamba_lru_list.in_list(node):
            tree.mamba_lru_list.remove_node(node)
            tree.mamba_evictable_size_ -= len(node.mamba_value)
        req_pool.mamba_pool.free(node.mamba_value)
        node.mamba_value = None
        self.assertIsNone(node.value)
        self.assertIsNone(node.mamba_value)
        self.assertIsNotNone(node.host_value)
        self.assertIsNotNone(node.mamba_host_value)

        _, matched_node, _, _ = tree._match_prefix_helper(RadixKey([1, 2, 3], None))
        self.assertIs(matched_node, node)
        self.assertIsNone(node.mamba_value)

        device_indices = tree.load_back(node)
        self.assertIsNotNone(device_indices)
        producer_id = tree.ready_to_load_host_cache()
        tree.cache_controller.layer_done_counter.set_consumer(producer_id)
        tree.cache_controller.ack_load_queue[-1].finish_event.synchronize()
        torch.cuda.synchronize()
        tree.loading_check()

        self.assertIsNotNone(node.value)
        self.assertIsNotNone(node.mamba_value)
        self.assertFalse(node.evicted)
        self.assertEqual(len(tree.ongoing_load_back), 0)
        self.assertEqual(node.value.numel(), len(kv_indices))
        self.assertEqual(node.mamba_value.numel(), 1)


if __name__ == "__main__":
    unittest.main()
