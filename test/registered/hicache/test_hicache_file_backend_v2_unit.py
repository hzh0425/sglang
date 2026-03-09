import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.host_pool_base import HostPoolBase
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.transfer_view import TransferView


class FakeHostPool(HostPoolBase):
    def __init__(self, page_bytes: int, page_size: int = 1):
        self.page_size = page_size
        self.layout = "layer_first"
        self.device = "cpu"
        self.dtype = torch.uint8
        self.size = 16
        self.page_bytes = page_bytes
        self.pages = {
            idx: torch.zeros(page_bytes, dtype=torch.uint8) for idx in range(self.size)
        }

    def alloc(self, need_size: int):
        return torch.arange(need_size, dtype=torch.int64)

    def free(self, indices: torch.Tensor) -> int:
        return indices.numel()

    def get_transfer_view(self, indices: torch.Tensor) -> TransferView:
        return TransferView(
            offsets=indices.cpu(),
            element_size_bytes=self.page_bytes,
            dtype=torch.uint8,
            device="cpu",
            layout=self.layout,
            page_size=self.page_size,
        )

    def load_from_device(self, device_pool, host_indices, device_indices, **kwargs) -> None:
        pass

    def backup_to_device(self, device_pool, host_indices, device_indices, **kwargs) -> None:
        pass

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id: int, io_backend: str
    ) -> None:
        pass

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend: str
    ) -> None:
        pass

    def get_data_page(self, index: int, flat: bool = True):
        page = self.pages[index].clone()
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.pages[index] = data_page.contiguous().view(torch.uint8).clone()

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.page_bytes, dtype=torch.uint8)


class FakeTypedHostPool(HostPoolBase):
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.page_size = 1
        self.layout = "layer_first"
        self.device = "cpu"
        self.dtype = dtype
        self.size = 16
        self.shape = shape
        self.pages = {
            idx: torch.zeros(shape, dtype=dtype) for idx in range(self.size)
        }

    def alloc(self, need_size: int):
        return torch.arange(need_size, dtype=torch.int64)

    def free(self, indices: torch.Tensor) -> int:
        return indices.numel()

    def get_transfer_view(self, indices: torch.Tensor) -> TransferView:
        return TransferView(
            offsets=indices.cpu(),
            element_size_bytes=self.get_dummy_flat_data_page().numel()
            * self.get_dummy_flat_data_page().element_size(),
            dtype=self.dtype,
            device="cpu",
            layout=self.layout,
            page_size=self.page_size,
        )

    def load_from_device(self, device_pool, host_indices, device_indices, **kwargs) -> None:
        pass

    def backup_to_device(self, device_pool, host_indices, device_indices, **kwargs) -> None:
        pass

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id: int, io_backend: str
    ) -> None:
        pass

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend: str
    ) -> None:
        pass

    def get_data_page(self, index: int, flat: bool = True):
        page = self.pages[index].clone()
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.pages[index] = data_page.reshape(self.shape).clone()

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.shape, dtype=self.dtype).flatten()


class FakeGroupedHostPool(FakeHostPool):
    def __init__(self):
        super().__init__(page_bytes=12)
        self.entries = [
            type("Entry", (), {"name": "kv", "host_pool": FakeHostPool(4)})(),
            type("Entry", (), {"name": "mamba", "host_pool": FakeHostPool(8)})(),
        ]


class TestHiCacheFileTransferViewV2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="hicache-file-v2-")
        self.storage = HiCacheFile(
            HiCacheStorageConfig(
                tp_rank=0,
                tp_size=2,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="Qwen/Test",
            ),
            file_path=self.temp_dir,
        )

    def tearDown(self):
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def test_transfer_signature_isolated_from_legacy_layout(self):
        legacy_key = "prefix-key"
        legacy_path = os.path.join(
            self.temp_dir, f"{legacy_key}_Qwen-Test_0_2.bin"
        )
        with open(legacy_path, "wb") as f:
            f.write(b"x" * 4)

        grouped_pool = FakeGroupedHostPool()
        grouped_pool.pages[0] = torch.arange(12, dtype=torch.uint8)
        self.storage.register_host_pool(grouped_pool)

        transfer_view = grouped_pool.get_transfer_view(torch.tensor([0], dtype=torch.int64))
        result = self.storage.batch_set_v2([legacy_key], transfer_view)
        self.assertEqual(result, [True])
        self.assertTrue(self.storage.transfer_suffix.startswith("_tv2_"))

    def test_batch_set_overwrites_stale_same_suffix_entry(self):
        pool = FakeHostPool(page_bytes=12)
        pool.pages[0] = torch.arange(12, dtype=torch.uint8)
        self.storage.register_host_pool(pool)

        key = "rewrite-key"
        stale_path = self.storage._get_component_path(key)
        with open(stale_path, "wb") as f:
            f.write(b"bad!")

        transfer_view = pool.get_transfer_view(torch.tensor([0], dtype=torch.int64))
        result = self.storage.batch_set_v2([key], transfer_view)
        self.assertEqual(result, [True])
        self.assertGreater(os.path.getsize(stale_path), 12)

        read_result = self.storage.batch_get_v2([key], transfer_view)
        self.assertEqual(read_result, [True])
        self.assertTrue(torch.equal(pool.pages[0], torch.arange(12, dtype=torch.uint8)))

    def test_grouped_transfer_ctx_serializes_auxiliary_mamba_page(self):
        kv_pool = FakeHostPool(page_bytes=4)
        mamba_pool = FakeHostPool(page_bytes=8)
        kv_pool.pages[0] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        mamba_pool.pages[5] = torch.tensor(
            [11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.uint8
        )

        group = HostPoolGroup(
            [
                PoolEntry(
                    name="kv",
                    host_pool=kv_pool,
                    device_pool=None,
                    index_resolver=lambda ctx: (
                        ctx["entries"]["kv"]["host_indices"],
                        None,
                    ),
                    layer_mapper=lambda _: None,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name="mamba",
                    host_pool=mamba_pool,
                    device_pool=None,
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
                    "mamba": {"host_indices": torch.tensor([5], dtype=torch.int64)},
                }
            }
        )
        self.storage.register_host_pool(group)

        transfer_view = group.get_transfer_view(torch.tensor([0], dtype=torch.int64))
        result = self.storage.batch_set_v2(["group-key"], transfer_view)
        self.assertEqual(result, [True])

        kv_pool.pages[0].zero_()
        mamba_pool.pages[5].zero_()
        read_result = self.storage.batch_get_v2(["group-key"], transfer_view)
        self.assertEqual(read_result, [True])
        self.assertTrue(torch.equal(kv_pool.pages[0], torch.tensor([1, 2, 3, 4], dtype=torch.uint8)))
        self.assertTrue(
            torch.equal(
                mamba_pool.pages[5],
                torch.tensor([11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.uint8),
            )
        )

    def test_mamba_sidecar_uses_last_page_key(self):
        kv_pool = FakeHostPool(page_bytes=4)
        mamba_pool = FakeHostPool(page_bytes=8)
        kv_pool.pages[0] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        kv_pool.pages[1] = torch.tensor([5, 6, 7, 8], dtype=torch.uint8)
        mamba_pool.pages[5] = torch.tensor(
            [21, 22, 23, 24, 25, 26, 27, 28], dtype=torch.uint8
        )

        group = HostPoolGroup(
            [
                PoolEntry(
                    name="kv",
                    host_pool=kv_pool,
                    device_pool=None,
                    index_resolver=lambda ctx: (
                        ctx["entries"]["kv"]["host_indices"],
                        None,
                    ),
                    layer_mapper=lambda _: None,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name="mamba",
                    host_pool=mamba_pool,
                    device_pool=None,
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
                    "kv": {"host_indices": torch.tensor([0, 1], dtype=torch.int64)},
                    "mamba": {
                        "host_indices": torch.tensor([5], dtype=torch.int64),
                        "page_ordinals": [1],
                    },
                }
            }
        )
        self.storage.register_host_pool(group)

        transfer_view = group.get_transfer_view(torch.tensor([0, 1], dtype=torch.int64))
        result = self.storage.batch_set_v2(["page0", "page1"], transfer_view)
        self.assertEqual(result, [True, True])

        self.assertTrue(os.path.exists(self.storage._get_component_path("page0")))
        self.assertFalse(os.path.exists(self.storage._get_component_path("page0", "mamba")))
        self.assertTrue(os.path.exists(self.storage._get_component_path("page1")))
        self.assertTrue(os.path.exists(self.storage._get_component_path("page1", "mamba")))

        kv_pool.pages[0].zero_()
        kv_pool.pages[1].zero_()
        mamba_pool.pages[5].zero_()
        read_result = self.storage.batch_get_v2(["page0", "page1"], transfer_view)
        self.assertEqual(read_result, [True, True])
        self.assertTrue(torch.equal(kv_pool.pages[0], torch.tensor([1, 2, 3, 4], dtype=torch.uint8)))
        self.assertTrue(torch.equal(kv_pool.pages[1], torch.tensor([5, 6, 7, 8], dtype=torch.uint8)))
        self.assertTrue(
            torch.equal(
                mamba_pool.pages[5],
                torch.tensor([21, 22, 23, 24, 25, 26, 27, 28], dtype=torch.uint8),
            )
        )

    def test_batch_exists_v2_finds_longest_common_prefix(self):
        kv_pool = FakeHostPool(page_bytes=4)
        mamba_pool = FakeHostPool(page_bytes=8)
        kv_pool.pages[0] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        kv_pool.pages[1] = torch.tensor([5, 6, 7, 8], dtype=torch.uint8)
        mamba_pool.pages[5] = torch.tensor(
            [21, 22, 23, 24, 25, 26, 27, 28], dtype=torch.uint8
        )

        group = HostPoolGroup(
            [
                PoolEntry(
                    name="kv",
                    host_pool=kv_pool,
                    device_pool=None,
                    index_resolver=lambda ctx: (
                        ctx["entries"]["kv"]["host_indices"],
                        None,
                    ),
                    layer_mapper=lambda _: None,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name="mamba",
                    host_pool=mamba_pool,
                    device_pool=None,
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
                    "kv": {"host_indices": torch.tensor([0, 1], dtype=torch.int64)},
                    "mamba": {
                        "host_indices": torch.tensor([5], dtype=torch.int64),
                        "page_ordinals": [1],
                    },
                }
            }
        )
        self.storage.register_host_pool(group)
        transfer_view = group.get_transfer_view(torch.tensor([0, 1], dtype=torch.int64))
        self.assertEqual(
            self.storage.batch_set_v2(["page0", "page1"], transfer_view), [True, True]
        )

        self.assertEqual(
            self.storage.batch_exists_v2(
                ["page0", "page1"],
                [{"name": "mamba", "policy": "trailing_pages", "trailing_pages": 1}],
            ),
            2,
        )
        self.assertEqual(
            self.storage.batch_exists_v2(
                ["page0", "page1"],
                [{"name": "mamba", "policy": "all_pages"}],
            ),
            0,
        )

    def test_grouped_transfer_ctx_with_entry_scoped_mamba_sidecar(self):
        kv_pool = FakeHostPool(page_bytes=4)
        mamba_pool = FakeHostPool(page_bytes=8)
        kv_pool.pages[0] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        kv_pool.pages[1] = torch.tensor([5, 6, 7, 8], dtype=torch.uint8)
        mamba_pool.pages[5] = torch.tensor(
            [31, 32, 33, 34, 35, 36, 37, 38], dtype=torch.uint8
        )

        def kv_index_resolver(ctx):
            host_indices = ctx.get("anchor_host_indices")
            if host_indices is None:
                host_indices = ctx.get("entries", {}).get("kv", {}).get("host_indices")
            if host_indices is None:
                return None
            return host_indices, None

        def mamba_index_resolver(ctx):
            entry = ctx.get("entries", {}).get("mamba", {})
            host_indices = entry.get("host_indices", ctx.get("mamba_host_indices"))
            if host_indices is None:
                return None
            return host_indices, None

        group = HostPoolGroup(
            [
                PoolEntry(
                    name="kv",
                    host_pool=kv_pool,
                    device_pool=None,
                    index_resolver=kv_index_resolver,
                    layer_mapper=lambda _: None,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name="mamba",
                    host_pool=mamba_pool,
                    device_pool=None,
                    index_resolver=mamba_index_resolver,
                    layer_mapper=lambda _: None,
                ),
            ]
        )
        group.set_transfer_context(
            {
                "anchor_host_indices": torch.tensor([0, 1], dtype=torch.int64),
                "entries": {
                    "mamba": {
                        "host_indices": torch.tensor([5], dtype=torch.int64),
                        "page_ordinals": [1],
                    },
                },
            }
        )
        self.storage.register_host_pool(group)

        transfer_view = group.get_transfer_view(torch.tensor([0, 1], dtype=torch.int64))
        self.assertEqual(
            self.storage.batch_set_v2(["page0", "page1"], transfer_view), [True, True]
        )
        self.assertTrue(os.path.exists(self.storage._get_component_path("page1", "mamba")))

    def test_set_from_page_components_reinterprets_component_dtype(self):
        kv_pool = FakeTypedHostPool(shape=(2, 1, 2), dtype=torch.bfloat16)
        mamba_pool = FakeHostPool(page_bytes=8)
        kv_pool.pages[0] = torch.tensor(
            [[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.bfloat16
        )
        mamba_pool.pages[5] = torch.tensor(
            [41, 42, 43, 44, 45, 46, 47, 48], dtype=torch.uint8
        )

        group = HostPoolGroup(
            [
                PoolEntry(
                    name="kv",
                    host_pool=kv_pool,
                    device_pool=None,
                    index_resolver=lambda ctx: (ctx["anchor_host_indices"], None),
                    layer_mapper=lambda _: None,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name="mamba",
                    host_pool=mamba_pool,
                    device_pool=None,
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
                "anchor_host_indices": torch.tensor([0], dtype=torch.int64),
                "entries": {
                    "mamba": {
                        "host_indices": torch.tensor([5], dtype=torch.int64),
                        "page_ordinals": [0],
                    }
                },
            }
        )
        components = group.get_page_components(0)
        assert components is not None
        kv_pool.pages[0].zero_()
        mamba_pool.pages[5].zero_()
        group.set_from_page_components(0, components)
        self.assertTrue(
            torch.equal(
                kv_pool.pages[0],
                torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.bfloat16),
            )
        )
        self.assertTrue(
            torch.equal(
                mamba_pool.pages[5],
                torch.tensor([41, 42, 43, 44, 45, 46, 47, 48], dtype=torch.uint8),
            )
        )
