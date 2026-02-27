import importlib
import importlib.machinery
import random
import sys
import types
import unittest

import torch


def _install_runtime_stubs():
    if "triton" not in sys.modules:
        triton = types.ModuleType("triton")
        triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)

        def _jit(fn=None, **kwargs):
            return fn if fn is not None else (lambda f: f)

        triton.jit = _jit
        triton.language = types.ModuleType("triton.language")
        triton.language.__spec__ = importlib.machinery.ModuleSpec(
            "triton.language", loader=None
        )
        triton.language.constexpr = int
        triton.language.core = types.SimpleNamespace(TRITON_MAX_TENSOR_NUMEL=2**31)
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = triton.language

    if "zmq" not in sys.modules:
        sys.modules["zmq"] = types.ModuleType("zmq")

    if not hasattr(torch.get_device_module(), "Stream"):
        torch.get_device_module = lambda: types.SimpleNamespace(Stream=object)


class _FakeReqToTokenPool:
    def __init__(self):
        self.device = "cpu"


class TestHybridSWAEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_runtime_stubs()
        cls.import_error = None
        try:
            base = importlib.import_module("sglang.srt.mem_cache.base_prefix_cache")
            cls.EvictParams = base.EvictParams
            cls.InsertParams = base.InsertParams
            cls.MatchPrefixParams = base.MatchPrefixParams
            cls.CacheInitParams = importlib.import_module(
                "sglang.srt.mem_cache.cache_init_params"
            ).CacheInitParams
            cls.HybridRadixCache = importlib.import_module(
                "sglang.srt.mem_cache.hybrid_radix_cache"
            ).HybridRadixCache
            cls.RadixKey = importlib.import_module(
                "sglang.srt.mem_cache.radix_cache"
            ).RadixKey
            cls.SWAKVPool = importlib.import_module(
                "sglang.srt.mem_cache.swa_memory_pool"
            ).SWAKVPool
            cls.SWATokenToKVPoolAllocator = importlib.import_module(
                "sglang.srt.mem_cache.swa_memory_pool"
            ).SWATokenToKVPoolAllocator
            cls.SWARadixCache = importlib.import_module(
                "sglang.srt.mem_cache.swa_radix_cache"
            ).SWARadixCache
        except Exception as e:
            cls.import_error = e

    def setUp(self):
        if self.import_error is not None:
            self.skipTest(f"missing runtime dependencies: {self.import_error}")

    def _build_pair(self, window: int):
        def _allocator():
            return self.SWATokenToKVPoolAllocator(
                size=200000,
                size_swa=200000,
                page_size=1,
                dtype=torch.float32,
                device="cpu",
                kvcache=self.SWAKVPool(
                    size=200000,
                    size_swa=200000,
                    page_size=1,
                    dtype=torch.float32,
                    head_num=1,
                    head_dim=8,
                    swa_attention_layer_ids=[1],
                    full_attention_layer_ids=[0],
                    enable_kvcache_transpose=False,
                    device="cpu",
                ),
                need_sort=False,
            )

        legacy = self.SWARadixCache(
            self.CacheInitParams(
                disable=False,
                req_to_token_pool=_FakeReqToTokenPool(),
                token_to_kv_pool_allocator=_allocator(),
                page_size=1,
                sliding_window_size=window,
            )
        )
        hybrid = self.HybridRadixCache(
            self.CacheInitParams(
                disable=False,
                req_to_token_pool=_FakeReqToTokenPool(),
                token_to_kv_pool_allocator=_allocator(),
                page_size=1,
                sliding_window_size=window,
                hybrid_enabled=True,
                hybrid_components=["full", "swa"],
                hybrid_primary_component="full",
                hybrid_swa_window_size=window,
            )
        )
        return legacy, hybrid

    def _run_random_ops(self, seed: int, rounds: int, window: int):
        rng = random.Random(seed)
        legacy, hybrid = self._build_pair(window)
        universe = []
        for _ in range(800):
            length = rng.randint(1, 48)
            universe.append([rng.randint(1, 500) for _ in range(length)])

        for _ in range(rounds):
            op = rng.choice(["insert", "match", "evict", "lock"])
            if op == "insert":
                seq = rng.choice(universe)
                lres = legacy.insert(self.InsertParams(key=self.RadixKey(seq), value=legacy.token_to_kv_pool_allocator.alloc(len(seq))))
                hres = hybrid.insert(self.InsertParams(key=self.RadixKey(seq), value=hybrid.token_to_kv_pool_allocator.alloc(len(seq))))
                self.assertEqual(lres.prefix_len, hres.prefix_len)
            elif op == "match":
                seq = rng.choice(universe)
                lres = legacy.match_prefix(self.MatchPrefixParams(key=self.RadixKey(seq)))
                hres = hybrid.match_prefix(self.MatchPrefixParams(key=self.RadixKey(seq)))
                self.assertEqual(len(lres.device_indices), len(hres.device_indices))
            elif op == "evict":
                full_need = rng.randint(0, 20)
                swa_need = rng.randint(0, 20)
                lres = legacy.evict(self.EvictParams(num_tokens=full_need, swa_num_tokens=swa_need))
                hres = hybrid.evict(self.EvictParams(num_tokens=full_need, swa_num_tokens=swa_need))
                self.assertEqual(lres.num_tokens_evicted, hres.num_tokens_evicted)
                self.assertEqual(lres.swa_num_tokens_evicted, hres.swa_num_tokens_evicted)
            else:
                seq = rng.choice(universe)
                lnode = legacy.match_prefix(self.MatchPrefixParams(key=self.RadixKey(seq))).last_device_node
                hnode = hybrid.match_prefix(self.MatchPrefixParams(key=self.RadixKey(seq))).last_device_node
                lmark = legacy.inc_lock_ref(lnode)
                hmark = hybrid.inc_lock_ref(hnode)
                self.assertEqual(legacy.full_evictable_size(), hybrid.full_evictable_size())
                self.assertEqual(legacy.swa_evictable_size(), hybrid.swa_evictable_size())
                legacy.dec_lock_ref(lnode, lmark)
                hybrid.dec_lock_ref(hnode, hmark)

    def test_randomized_equivalence_large(self):
        self._run_random_ops(seed=20260226, rounds=1600, window=8)

    def test_multi_seed_stress(self):
        for seed in [9, 99, 999, 9999]:
            self._run_random_ops(seed=seed, rounds=900, window=8)

    def test_component_requests_mapping(self):
        legacy, hybrid = self._build_pair(window=4)
        seq = [1, 2, 3, 4, 5, 6, 7, 8]
        legacy.insert(self.InsertParams(key=self.RadixKey(seq), value=legacy.token_to_kv_pool_allocator.alloc(len(seq))))
        hybrid.insert(self.InsertParams(key=self.RadixKey(seq), value=hybrid.token_to_kv_pool_allocator.alloc(len(seq))))

        lres = legacy.evict(self.EvictParams(num_tokens=3, swa_num_tokens=2))
        hres = hybrid.evict(
            self.EvictParams(
                num_tokens=0,
                swa_num_tokens=0,
                component_requests={"full": 3, "swa": 2},
            )
        )
        self.assertEqual(lres.num_tokens_evicted, hres.num_tokens_evicted)
        self.assertEqual(lres.swa_num_tokens_evicted, hres.swa_num_tokens_evicted)


if __name__ == "__main__":
    unittest.main()
