import os
import unittest

import torch

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hybrid_radix_cache import HybridMambaRadixCache, HybridSWARadixCache
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device


class TestHybridRadixCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (
            (hasattr(torch, "cuda") and torch.cuda.is_available())
            or (hasattr(torch, "xpu") and torch.xpu.is_available())
        ):
            raise unittest.SkipTest("Hybrid radix cache tests require an accelerator.")

    class _DummyReq:
        def __init__(self):
            self.req_pool_idx = None
            self.mamba_pool_idx = None
            self.kv_committed_len = 0
            self.is_chunked = 0
            self._kv_committed_len = 0

        def is_dllm(self):
            return False

        def pop_committed_kv_cache(self):
            return self._kv_committed_len

    def _make_mamba_trees(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
        size = 128
        dtype = torch.bfloat16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        def build_one(cache_cls):
            req_to_token_pool = HybridReqToTokenPool(
                size=max_num_reqs,
                mamba_size=mamba_cache_size,
                mamba_spec_state_size=max_num_reqs,
                max_context_len=max_context_len,
                device=device,
                enable_memory_saver=False,
                cache_params=mamba2_cache_params,
                enable_mamba_extra_buffer=False,
                speculative_num_draft_tokens=3,
            )
            pool = HybridLinearKVPool(
                size=size,
                dtype=dtype,
                page_size=1,
                head_num=head_num,
                head_dim=head_dim,
                full_attention_layer_ids=full_attention_layer_ids,
                enable_kvcache_transpose=False,
                device=device,
                enable_memory_saver=False,
                mamba_pool=req_to_token_pool.mamba_pool,
            )
            allocator = TokenToKVPoolAllocator(
                size=size,
                dtype=dtype,
                device=device,
                kvcache=pool,
                need_sort=False,
            )
            tree = cache_cls(
                CacheInitParams(
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool_allocator=allocator,
                    page_size=1,
                    disable=False,
                )
            )
            return tree, allocator, req_to_token_pool

        return build_one(MambaRadixCache), build_one(HybridMambaRadixCache)

    def _make_mamba_components(self):
        (_, allocator, req_to_token_pool), _ = self._make_mamba_trees()
        return req_to_token_pool, allocator

    def _make_dummy_req(self, req_to_token_pool):
        req = self._DummyReq()
        req_to_token_pool.alloc([req])
        return req

    def test_hybrid_mamba_equivalence(self):
        (old_tree, old_alloc, old_req_pool), (new_tree, new_alloc, new_req_pool) = (
            self._make_mamba_trees()
        )

        old_req1 = self._make_dummy_req(old_req_pool)
        new_req1 = self._make_dummy_req(new_req_pool)
        old_kv1 = old_alloc.alloc(3)
        new_kv1 = new_alloc.alloc(3)
        old_res1 = old_tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=old_kv1,
                mamba_value=old_req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        new_res1 = new_tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=new_kv1,
                mamba_value=new_req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        self.assertEqual(old_res1.prefix_len, new_res1.prefix_len)
        self.assertEqual(old_res1.mamba_exist, new_res1.mamba_exist)

        old_req2 = self._make_dummy_req(old_req_pool)
        new_req2 = self._make_dummy_req(new_req_pool)
        old_kv2 = old_alloc.alloc(6)
        new_kv2 = new_alloc.alloc(6)
        old_res2 = old_tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5, 6]),
                value=old_kv2,
                mamba_value=old_req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        new_res2 = new_tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5, 6]),
                value=new_kv2,
                mamba_value=new_req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        self.assertEqual(old_res2.prefix_len, new_res2.prefix_len)
        self.assertEqual(old_res2.mamba_exist, new_res2.mamba_exist)

        old_match = old_tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4])))
        new_match = new_tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4])))
        self.assertEqual(old_match.device_indices.tolist(), new_match.device_indices.tolist())
        self.assertEqual(old_match.mamba_branching_seqlen, new_match.mamba_branching_seqlen)

        old_handle = old_tree.inc_lock_ref(old_match.last_device_node)
        new_handle = new_tree.inc_lock_ref(new_match.last_device_node)
        self.assertEqual(old_tree.full_protected_size(), new_tree.full_protected_size())
        self.assertEqual(old_tree.mamba_protected_size(), new_tree.mamba_protected_size())
        old_tree.dec_lock_ref(old_match.last_device_node)
        new_tree.dec_lock_ref(new_match.last_device_node)

        old_evict = old_tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        new_evict = new_tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertEqual(old_evict.mamba_num_evicted, new_evict.mamba_num_evicted)
        self.assertEqual(old_evict.num_tokens_evicted, new_evict.num_tokens_evicted)
        self.assertEqual(old_tree.total_size(), new_tree.total_size())

    def test_hybrid_mamba_extended_equivalence(self):
        (old_tree, old_alloc, old_req_pool), (new_tree, new_alloc, new_req_pool) = (
            self._make_mamba_trees()
        )

        old_requests = [self._make_dummy_req(old_req_pool) for _ in range(4)]
        new_requests = [self._make_dummy_req(new_req_pool) for _ in range(4)]
        payloads = [
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6, 7],
            [10, 11, 12],
            [1, 2, 3, 4, 5, 60, 70],
        ]
        for req_old, req_new, token_ids in zip(old_requests, new_requests, payloads):
            old_res = old_tree.insert(
                InsertParams(
                    key=RadixKey(token_ids),
                    value=old_alloc.alloc(len(token_ids)),
                    mamba_value=req_old.mamba_pool_idx.unsqueeze(0),
                )
            )
            new_res = new_tree.insert(
                InsertParams(
                    key=RadixKey(token_ids),
                    value=new_alloc.alloc(len(token_ids)),
                    mamba_value=req_new.mamba_pool_idx.unsqueeze(0),
                )
            )
            self.assertEqual(old_res.prefix_len, new_res.prefix_len)
            self.assertEqual(old_res.mamba_exist, new_res.mamba_exist)

        old_full_evict = old_tree.evict(EvictParams(num_tokens=1))
        new_full_evict = new_tree.evict(EvictParams(num_tokens=1))
        self.assertEqual(old_full_evict.num_tokens_evicted, new_full_evict.num_tokens_evicted)
        self.assertEqual(old_tree.total_size(), new_tree.total_size())

        for token_ids in ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 60, 70], [1, 2, 3, 4, 5, 6, 7]):
            old_match = old_tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
            new_match = new_tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
            self.assertEqual(old_match.device_indices.tolist(), new_match.device_indices.tolist())
            self.assertEqual(old_match.mamba_branching_seqlen, new_match.mamba_branching_seqlen)

        old_mamba_evict = old_tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        new_mamba_evict = new_tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertEqual(old_mamba_evict.mamba_num_evicted, new_mamba_evict.mamba_num_evicted)
        self.assertEqual(old_mamba_evict.num_tokens_evicted, new_mamba_evict.num_tokens_evicted)

        old_req = self._make_dummy_req(old_req_pool)
        new_req = self._make_dummy_req(new_req_pool)
        token_ids = [1, 2, 3, 4, 5, 6, 7]
        old_match = old_tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids), req=old_req, cow_mamba=True)
        )
        new_match = new_tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids), req=new_req, cow_mamba=True)
        )
        self.assertEqual(old_match.device_indices.tolist(), new_match.device_indices.tolist())
        self.assertIsNotNone(old_req.mamba_pool_idx)
        self.assertIsNotNone(new_req.mamba_pool_idx)

    def _make_swa_trees(self):
        req_size = 8
        max_context_len = 64
        kv_size = 64
        kv_size_swa = 32
        page_size = 1
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 24
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]

        def build_one(cache_cls):
            req_to_token_pool = ReqToTokenPool(
                size=req_size,
                max_context_len=max_context_len,
                device=device,
                enable_memory_saver=False,
            )
            kv_pool = SWAKVPool(
                size=kv_size,
                size_swa=kv_size_swa,
                page_size=page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                swa_attention_layer_ids=swa_attention_layer_ids,
                full_attention_layer_ids=full_attention_layer_ids,
                enable_kvcache_transpose=False,
                device=device,
            )
            allocator = SWATokenToKVPoolAllocator(
                size=kv_size,
                size_swa=kv_size_swa,
                page_size=page_size,
                dtype=dtype,
                device=device,
                kvcache=kv_pool,
                need_sort=False,
            )
            tree = cache_cls(
                CacheInitParams(
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool_allocator=allocator,
                    page_size=page_size,
                    disable=False,
                    sliding_window_size=sliding_window_size,
                )
            )
            return tree, allocator

        return build_one(SWARadixCache), build_one(HybridSWARadixCache)

    def _make_swa_components(self):
        (_, allocator), _ = self._make_swa_trees()
        req_size = 8
        max_context_len = 64
        page_size = 1
        kv_size = 64
        kv_size_swa = 32
        head_num = 8
        head_dim = 128
        num_layers = 24
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        req_to_token_pool = ReqToTokenPool(
            size=req_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        return req_to_token_pool, allocator

    def test_hybrid_swa_equivalence(self):
        (old_tree, old_alloc), (new_tree, new_alloc) = self._make_swa_trees()
        old_res1 = old_tree.insert(
            InsertParams(key=RadixKey([1, 2, 3]), value=old_alloc.alloc(3))
        )
        new_res1 = new_tree.insert(
            InsertParams(key=RadixKey([1, 2, 3]), value=new_alloc.alloc(3))
        )
        self.assertEqual(old_res1.prefix_len, new_res1.prefix_len)

        old_res2 = old_tree.insert(
            InsertParams(key=RadixKey([1, 2, 3, 4, 5, 6]), value=old_alloc.alloc(6))
        )
        new_res2 = new_tree.insert(
            InsertParams(key=RadixKey([1, 2, 3, 4, 5, 6]), value=new_alloc.alloc(6))
        )
        self.assertEqual(old_res2.prefix_len, new_res2.prefix_len)

        old_match = old_tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5, 6])))
        new_match = new_tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5, 6])))
        self.assertEqual(old_match.device_indices.tolist(), new_match.device_indices.tolist())

        old_handle = old_tree.inc_lock_ref(old_match.last_device_node)
        new_handle = new_tree.inc_lock_ref(new_match.last_device_node)
        self.assertEqual(
            old_handle,
            new_handle,
        )
        self.assertEqual(old_tree.full_protected_size(), new_tree.full_protected_size())
        self.assertEqual(old_tree.swa_protected_size(), new_tree.swa_protected_size())
        old_tree.dec_lock_ref(old_match.last_device_node, old_handle)
        new_tree.dec_lock_ref(new_match.last_device_node, new_handle)

        old_evict = old_tree.evict(EvictParams(num_tokens=0, swa_num_tokens=3))
        new_evict = new_tree.evict(EvictParams(num_tokens=0, swa_num_tokens=3))
        self.assertEqual(old_evict.swa_num_tokens_evicted, new_evict.swa_num_tokens_evicted)
        self.assertEqual(old_evict.num_tokens_evicted, new_evict.num_tokens_evicted)
        self.assertEqual(old_tree.total_size(), new_tree.total_size())

    def test_hybrid_swa_cache_finished_req_eagle_equivalence(self):
        def build(cache_cls):
            req_size = 8
            max_context_len = 64
            kv_size = 64
            kv_size_swa = 32
            page_size = 1
            sliding_window_size = 4
            head_num = 8
            head_dim = 128
            num_layers = 24
            global_interval = 4
            dtype = torch.bfloat16
            device = get_device()
            full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
            full_attention_layer_ids_set = set(full_attention_layer_ids)
            swa_attention_layer_ids = [
                i for i in range(num_layers) if i not in full_attention_layer_ids_set
            ]
            req_to_token_pool = ReqToTokenPool(
                size=req_size,
                max_context_len=max_context_len,
                device=device,
                enable_memory_saver=False,
            )
            kv_pool = SWAKVPool(
                size=kv_size,
                size_swa=kv_size_swa,
                page_size=page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                swa_attention_layer_ids=swa_attention_layer_ids,
                full_attention_layer_ids=full_attention_layer_ids,
                enable_kvcache_transpose=False,
                device=device,
            )
            allocator = SWATokenToKVPoolAllocator(
                size=kv_size,
                size_swa=kv_size_swa,
                page_size=page_size,
                dtype=dtype,
                device=device,
                kvcache=kv_pool,
                need_sort=False,
            )
            tree = cache_cls(
                CacheInitParams(
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool_allocator=allocator,
                    page_size=page_size,
                    disable=False,
                    is_eagle=True,
                    sliding_window_size=sliding_window_size,
                )
            )
            return tree, allocator, req_to_token_pool

        (old_tree, old_allocator, old_req_pool), (new_tree, new_allocator, new_req_pool) = (
            build(SWARadixCache),
            build(HybridSWARadixCache),
        )

        def make_req(req_pool_idx, allocator, req_pool, token_ids):
            req = self._DummyReq()
            req.req_pool_idx = req_pool_idx
            req.origin_input_ids = token_ids
            req.output_ids = []
            req._kv_committed_len = len(token_ids)
            kv_indices = allocator.alloc(req._kv_committed_len)
            req_pool.write((req.req_pool_idx, slice(0, req._kv_committed_len)), kv_indices)
            req.extra_key = None
            req.last_node = old_tree.root_node if req_pool is old_req_pool else new_tree.root_node
            req.swa_uuid_for_lock = None
            req.swa_evicted_seqlen = 0
            req.cache_protected_len = 1
            req.prefix_indices = torch.tensor([7, 8, 9, 10, 11], device=(old_tree.device if req_pool is old_req_pool else new_tree.device))
            return req

        old_req = make_req(0, old_allocator, old_req_pool, [1, 2, 3, 4, 5, 6])
        new_req = make_req(0, new_allocator, new_req_pool, [1, 2, 3, 4, 5, 6])
        old_capture = {}
        new_capture = {}
        old_insert = old_tree.insert
        new_insert = new_tree.insert

        def wrap_insert(captured, inner):
            def _wrapped(params):
                captured["prev_prefix_len"] = params.prev_prefix_len
                captured["is_bigram"] = params.key.is_bigram
                captured["key_len"] = len(params.key)
                return inner(params)

            return _wrapped

        old_tree.insert = wrap_insert(old_capture, old_insert)
        new_tree.insert = wrap_insert(new_capture, new_insert)
        old_tree.cache_finished_req(old_req, is_insert=True)
        new_tree.cache_finished_req(new_req, is_insert=True)
        self.assertEqual(old_capture, new_capture)

        old_req2 = make_req(1, old_allocator, old_req_pool, [11, 12, 13, 14, 15, 16])
        new_req2 = make_req(1, new_allocator, new_req_pool, [11, 12, 13, 14, 15, 16])
        old_freed = []
        new_freed = []
        old_free = old_allocator.free
        new_free = new_allocator.free

        def wrap_free(captured, inner):
            def _wrapped(indices):
                captured.append(int(indices.numel()))
                return inner(indices)

            return _wrapped

        old_allocator.free = wrap_free(old_freed, old_free)
        new_allocator.free = wrap_free(new_freed, new_free)
        old_tree.cache_finished_req(old_req2, is_insert=False)
        new_tree.cache_finished_req(new_req2, is_insert=False)
        self.assertEqual(old_freed, new_freed)

    def test_scheduler_uses_hybrid_mamba_tree(self):
        req_to_token_pool, allocator = self._make_mamba_components()

        class DummySpecAlgo:
            def is_eagle(self):
                return False

        class DummyServerArgs:
            disable_radix_cache = False
            radix_eviction_policy = "lru"
            chunked_prefill_size = 2048

            def enable_mamba_extra_buffer(self):
                return False

        class DummyTPWorker:
            is_hybrid_swa = False
            sliding_window_size = None

            def __init__(self):
                self.model_runner = type(
                    "MR",
                    (),
                    {"hybrid_gdn_config": object(), "mamba2_config": None},
                )()

            def get_memory_pool(self):
                return req_to_token_pool, allocator

        fake = Scheduler.__new__(Scheduler)
        fake.server_args = DummyServerArgs()
        fake.tp_worker = DummyTPWorker()
        fake.spec_algorithm = DummySpecAlgo()
        fake.attn_tp_cpu_group = None
        fake.tp_cpu_group = None
        fake.enable_metrics = False
        fake.enable_kv_cache_events = False
        fake.pp_rank = 0
        fake.pp_size = 1
        fake.page_size = 1
        fake.enable_hierarchical_cache = False

        Scheduler.init_cache_with_memory_pool(fake)
        self.assertIsInstance(fake.tree_cache, HybridMambaRadixCache)

    def test_scheduler_uses_hybrid_swa_tree(self):
        req_to_token_pool, allocator = self._make_swa_components()

        class DummySpecAlgo:
            def is_eagle(self):
                return False

        class DummyServerArgs:
            disable_radix_cache = False
            radix_eviction_policy = "lru"
            chunked_prefill_size = 2048

            def enable_mamba_extra_buffer(self):
                return False

        class DummyTPWorker:
            is_hybrid_swa = True
            sliding_window_size = 4

            def __init__(self):
                self.model_runner = type(
                    "MR",
                    (),
                    {"hybrid_gdn_config": None, "mamba2_config": None},
                )()

            def get_tokens_per_layer_info(self):
                return 1, 1

            def get_memory_pool(self):
                return req_to_token_pool, allocator

        fake = Scheduler.__new__(Scheduler)
        fake.server_args = DummyServerArgs()
        fake.tp_worker = DummyTPWorker()
        fake.spec_algorithm = DummySpecAlgo()
        fake.attn_tp_cpu_group = None
        fake.tp_cpu_group = None
        fake.enable_metrics = False
        fake.enable_kv_cache_events = False
        fake.pp_rank = 0
        fake.pp_size = 1
        fake.page_size = 1
        fake.enable_hierarchical_cache = False

        Scheduler.init_cache_with_memory_pool(fake)
        self.assertIsInstance(fake.tree_cache, HybridSWARadixCache)
