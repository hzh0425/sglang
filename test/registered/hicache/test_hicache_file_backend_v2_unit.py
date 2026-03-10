import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    StorageTransfer,
)


class FakeHostPool:
    def __init__(self, page_bytes: int, page_size: int = 1):
        self.page_size = page_size
        self.pages = {
            idx: torch.zeros(page_bytes, dtype=torch.uint8) for idx in range(32)
        }

    def get_data_page(self, index: int, flat: bool = True):
        page = self.pages[index].clone()
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.pages[index] = data_page.contiguous().view(torch.uint8).clone()

    def get_dummy_flat_data_page(self):
        first_page = next(iter(self.pages.values()))
        return torch.zeros_like(first_page)


class FakeTypedHostPool:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.page_size = 1
        self.shape = shape
        self.dtype = dtype
        self.pages = {idx: torch.zeros(shape, dtype=dtype) for idx in range(32)}

    def get_data_page(self, index: int, flat: bool = True):
        page = self.pages[index].clone()
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.pages[index] = data_page.reshape(self.shape).clone()

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.shape, dtype=self.dtype).flatten()


class TestHiCacheFileV2(unittest.TestCase):
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

    def test_batch_set_and_get_kv_roundtrip(self):
        kv_pool = FakeHostPool(page_bytes=12)
        kv_pool.pages[0] = torch.arange(12, dtype=torch.uint8)
        self.storage.register_host_pool(kv_pool)

        result = self.storage.batch_set_v2(
            [StorageTransfer("kv", ["page0"], torch.tensor([0], dtype=torch.int64))]
        )
        self.assertEqual(result, {"kv": [True]})

        kv_pool.pages[0].zero_()
        read_result = self.storage.batch_get_v2(
            [StorageTransfer("kv", ["page0"], torch.tensor([0], dtype=torch.int64))]
        )
        self.assertEqual(read_result, {"kv": [True]})
        self.assertTrue(torch.equal(kv_pool.pages[0], torch.arange(12, dtype=torch.uint8)))

    def test_batch_set_and_get_kv_with_mamba_sidecar(self):
        kv_pool = FakeHostPool(page_bytes=4)
        mamba_pool = FakeHostPool(page_bytes=8)
        kv_pool.pages[0] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        kv_pool.pages[1] = torch.tensor([5, 6, 7, 8], dtype=torch.uint8)
        mamba_pool.pages[5] = torch.tensor(
            [21, 22, 23, 24, 25, 26, 27, 28], dtype=torch.uint8
        )
        self.storage.register_host_pool(kv_pool)
        self.storage.register_pool("mamba", mamba_pool)

        write_result = self.storage.batch_set_v2(
            [
                StorageTransfer(
                    "kv", ["page0", "page1"], torch.tensor([0, 1], dtype=torch.int64)
                ),
                StorageTransfer(
                    "mamba", ["page1"], torch.tensor([5], dtype=torch.int64)
                ),
            ]
        )
        self.assertEqual(write_result, {"kv": [True, True], "mamba": [True]})
        self.assertTrue(os.path.exists(self.storage._get_component_path("page0")))
        self.assertTrue(os.path.exists(self.storage._get_component_path("page1")))
        self.assertTrue(os.path.exists(self.storage._get_component_path("page1", "mamba")))

        kv_pool.pages[0].zero_()
        kv_pool.pages[1].zero_()
        mamba_pool.pages[5].zero_()
        extra_info = HiCacheStorageExtraInfo(extra_info={})
        read_result = self.storage.batch_get_v2(
            [
                StorageTransfer(
                    "kv", ["page0", "page1"], torch.tensor([0, 1], dtype=torch.int64)
                ),
                StorageTransfer(
                    "mamba", ["page1"], torch.tensor([5], dtype=torch.int64)
                ),
            ],
            extra_info=extra_info,
        )
        self.assertEqual(read_result, {"kv": [True, True], "mamba": [True]})
        self.assertEqual(extra_info.extra_info["loaded_names"], {"kv", "mamba"})
        self.assertEqual(extra_info.extra_info["hit_page_count_by_name"]["kv"], 2)
        self.assertEqual(extra_info.extra_info["hit_page_count_by_name"]["mamba"], 1)
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
        self.storage.register_host_pool(kv_pool)
        self.storage.register_pool("mamba", mamba_pool)

        self.storage.batch_set_v2(
            [
                StorageTransfer(
                    "kv", ["page0", "page1"], torch.tensor([0, 1], dtype=torch.int64)
                ),
                StorageTransfer(
                    "mamba", ["page1"], torch.tensor([5], dtype=torch.int64)
                ),
            ]
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

    def test_batch_get_v2_restores_kv_dtype(self):
        kv_pool = FakeTypedHostPool(shape=(2, 1, 2), dtype=torch.bfloat16)
        kv_pool.pages[0] = torch.tensor(
            [[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.bfloat16
        )
        self.storage.register_host_pool(kv_pool)

        self.storage.batch_set_v2(
            [StorageTransfer("kv", ["typed-page"], torch.tensor([0], dtype=torch.int64))]
        )
        kv_pool.pages[0].zero_()
        self.storage.batch_get_v2(
            [StorageTransfer("kv", ["typed-page"], torch.tensor([0], dtype=torch.int64))]
        )
        self.assertTrue(
            torch.equal(
                kv_pool.pages[0],
                torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.bfloat16),
            )
        )


if __name__ == "__main__":
    unittest.main()
