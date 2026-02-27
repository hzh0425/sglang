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


class _DummyKVPool:
    def get_cpu_copy(self, indices):
        return indices

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return None


class _FakeMambaPool:
    def __init__(self):
        self.freed = []

    def free(self, value):
        if isinstance(value, torch.Tensor):
            self.freed.append(value.clone())
        else:
            self.freed.append(value)


class _FakeReqToTokenPool:
    def __init__(self):
        self.mamba_pool = _FakeMambaPool()


class TestHybridMambaEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_runtime_stubs()
        cls.import_error = None
        try:
            cls.TokenToKVPoolAllocator = importlib.import_module(
                "sglang.srt.mem_cache.allocator"
            ).TokenToKVPoolAllocator
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
            cls.MambaRadixCache = importlib.import_module(
                "sglang.srt.mem_cache.mamba_radix_cache"
            ).MambaRadixCache
        except Exception as e:
            cls.import_error = e

    def setUp(self):
        if self.import_error is not None:
            self.skipTest(f"missing runtime dependencies: {self.import_error}")

    def _build_pair(self):
        legacy_alloc = self.TokenToKVPoolAllocator(
            size=200000,
            dtype=torch.float32,
            device="cpu",
            kvcache=_DummyKVPool(),
            need_sort=False,
        )
        hybrid_alloc = self.TokenToKVPoolAllocator(
            size=200000,
            dtype=torch.float32,
            device="cpu",
            kvcache=_DummyKVPool(),
            need_sort=False,
        )
        legacy = self.MambaRadixCache(
            self.CacheInitParams(
                disable=False,
                req_to_token_pool=_FakeReqToTokenPool(),
                token_to_kv_pool_allocator=legacy_alloc,
                page_size=1,
            )
        )
        hybrid = self.HybridRadixCache(
            self.CacheInitParams(
                disable=False,
                req_to_token_pool=_FakeReqToTokenPool(),
                token_to_kv_pool_allocator=hybrid_alloc,
                page_size=1,
                hybrid_enabled=True,
                hybrid_components=["full", "mamba"],
                hybrid_primary_component="full",
            )
        )
        return legacy, hybrid, legacy_alloc, hybrid_alloc

    def _run_random_ops(self, seed: int, rounds: int):
        rng = random.Random(seed)
        legacy, hybrid, legacy_alloc, hybrid_alloc = self._build_pair()

        universe = []
        for _ in range(800):
            length = rng.randint(1, 40)
            universe.append([rng.randint(1, 400) for _ in range(length)])

        mid = 10_000
        for _ in range(rounds):
            op = rng.choice(["insert", "match", "evict", "lock"])
            if op == "insert":
                seq = rng.choice(universe)
                lres = legacy.insert(
                    self.InsertParams(
                        key=self.RadixKey(seq),
                        value=legacy_alloc.alloc(len(seq)),
                        mamba_value=torch.tensor([mid], dtype=torch.int64),
                    )
                )
                hres = hybrid.insert(
                    self.InsertParams(
                        key=self.RadixKey(seq),
                        value=hybrid_alloc.alloc(len(seq)),
                        mamba_value=torch.tensor([mid], dtype=torch.int64),
                    )
                )
                self.assertEqual(lres.prefix_len, hres.prefix_len)
                self.assertEqual(lres.mamba_exist, hres.mamba_exist)
                mid += 1
            elif op == "match":
                seq = rng.choice(universe)
                lres = legacy.match_prefix(self.MatchPrefixParams(key=self.RadixKey(seq)))
                hres = hybrid.match_prefix(self.MatchPrefixParams(key=self.RadixKey(seq)))
                self.assertEqual(len(lres.device_indices), len(hres.device_indices))
            elif op == "evict":
                full_need = rng.randint(0, 20)
                mamba_need = rng.randint(0, 6)
                lres = legacy.evict(self.EvictParams(num_tokens=full_need, mamba_num=mamba_need))
                hres = hybrid.evict(self.EvictParams(num_tokens=full_need, mamba_num=mamba_need))
                self.assertEqual(lres.num_tokens_evicted, hres.num_tokens_evicted)
                self.assertEqual(lres.mamba_num_evicted, hres.mamba_num_evicted)
            else:
                seq = rng.choice(universe)
                lnode = legacy.match_prefix(self.MatchPrefixParams(key=self.RadixKey(seq))).last_device_node
                hnode = hybrid.match_prefix(self.MatchPrefixParams(key=self.RadixKey(seq))).last_device_node
                legacy.inc_lock_ref(lnode)
                hhandle = hybrid.inc_lock_ref(hnode)
                self.assertEqual(legacy.full_evictable_size(), hybrid.full_evictable_size())
                self.assertEqual(legacy.mamba_evictable_size(), hybrid.mamba_evictable_size())
                legacy.dec_lock_ref(lnode)
                hybrid.dec_lock_ref(hnode, hhandle)

    def test_randomized_equivalence_large(self):
        self._run_random_ops(seed=20260226, rounds=1600)

    def test_multi_seed_stress(self):
        for seed in [7, 77, 777, 7777]:
            self._run_random_ops(seed=seed, rounds=900)

    def test_component_requests_mapping(self):
        legacy, hybrid, _, _ = self._build_pair()
        seq = [1, 2, 3, 4, 5, 6, 7]
        legacy.insert(
            self.InsertParams(
                key=self.RadixKey(seq),
                value=torch.arange(1, len(seq) + 1, dtype=torch.int64),
                mamba_value=torch.tensor([1], dtype=torch.int64),
            )
        )
        hybrid.insert(
            self.InsertParams(
                key=self.RadixKey(seq),
                value=torch.arange(1, len(seq) + 1, dtype=torch.int64),
                mamba_value=torch.tensor([1], dtype=torch.int64),
            )
        )
        lres = legacy.evict(self.EvictParams(num_tokens=3, mamba_num=1))
        hres = hybrid.evict(
            self.EvictParams(
                num_tokens=0,
                mamba_num=0,
                component_requests={"full": 3, "mamba": 1},
            )
        )
        self.assertEqual(lres.num_tokens_evicted, hres.num_tokens_evicted)
        self.assertEqual(lres.mamba_num_evicted, hres.mamba_num_evicted)


if __name__ == "__main__":
    unittest.main()
