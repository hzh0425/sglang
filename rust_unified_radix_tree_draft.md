# Rust UnifiedRadixTree Multi-Stage Draft

你是一个 SGLang / mem_cache / HiCache 方向的研发 agent。当前任务是把 Rust 版本 UnifiedRadixTree 从 POC 推进到可用的多阶段实现。请先学习 repo 当前实现，再按 milestone 逐步研发和测试。不要假设已有额外知识库文件；下面已经把必要背景总结在 prompt 中。

## 任务目标

当前分支有 Rust 版本 `UnifiedRadixTree` POC，核心代码在：

- `rust/sglang-mem-cache/`
- `python/sglang/srt/mem_cache/rust_unified_radix_cache.py`

目标分三步：

1. Rust backend 可构建、可注册，Full-only L1 跑通。
2. Full-only L2 HiCache 补齐正确性，验证 write_through/write_back 和 L2 命中率。
3. SWA/Mamba/sidecar L2 能力补齐，跑通对应 KL/HiCache 测试。

## 必须先读的 repo 文件

先读设计和 Rust/Python wrapper：

- `rust/sglang-mem-cache/DESIGN.md`
- `python/sglang/srt/mem_cache/rust_unified_radix_cache.py`
- `rust/sglang-mem-cache/src/radix_cache.rs`
- `rust/sglang-mem-cache/src/tree_node_lru.rs`
- `rust/sglang-mem-cache/src/radix_cache_wrapper.rs`
- `python/sglang/srt/mem_cache/registry.py`
- `python/pyproject.toml`

再读 Python reference：

- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/unified_cache_components/full_component.py`
- `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py`
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`

重点测试文件：

- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_full.py`
- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_swa.py`
- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py`
- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py`
- `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`
- `test/registered/unit/mem_cache/test_unified_radix_cache_bench.py`

## 关键背景知识

### Rust design 边界

Rust 版本不是要机械复制 Python `UnifiedRadixCache` 的 object design。保持这个所有权边界：

- Rust owns radix tree、node arena、children、per-component state、LRU、lock_ref、eviction decision。
- Python owns KV allocator、req_to_token_pool、host pool、HiCache controller、实际 D2H/H2D transfer。
- Rust 不能直接 free KV；只能返回 `DeferredAction`，由 Python orchestrator 处理 allocator/free/backup/loadback。
- Python request 上的 `req.last_node` 是 Rust opaque `NodeIdx`，不是 Python `UnifiedTreeNode`。不要让 scheduler 或 HiCache 逻辑直接访问 `last_node.children/value/parent`。

### HiCache 基本语义

HiCache 层级：

- L1 = device KV，容量近似 `--max-total-tokens`
- L2 = host KV，容量近似 `L1 * --hicache-ratio`
- L3 = storage，本任务暂不实现，保持 unsupported

节点状态：

- S0: device=no, host=no，absent
- S1: device=yes, host=no，device only
- S2: device=yes, host=yes，device + host backup
- S3: device=no, host=yes，host-only / device evicted

关键 transition：

- T1 insert -> S1
- T2 backup -> S2
- T3 device evict backed-up node -> S3
- T4 load_back -> S2
- T6 host_evict/delete
- T8 split redistribution

核心 invariant：

- Full 是 anchor。Aux component 不能脱离 Full。
- child 有数据则 parent 必须有数据。
- write-through backup 必须形成 root-to-leaf 连续 prefix。
- host leaf 判断必须看 `host_value`，不能只看 device `value`。
- H2D loadback 的 host source lock 必须保持到 load ack 后释放。
- host eviction 不能 evict 有 host-backed descendant 的 ancestor，否则 descendant loadback chain 会断。
- 对 SWA/Mamba，`full_lock >= aux_lock`，host/device LRU 不能双挂。

### 当前已知缺口

当前 Rust POC 大概率存在这些差距：

- `_mem_cache_core` 没接进 `python/pyproject.toml`，Python import 不通。
- `install_rust_radix_cache()` 存在但没有自动调用，`--radix-cache-backend rust_unified` 还不可选。
- KL full 测试现在靠 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` 跑 Python Unified，不是 Rust。
- Rust Full L2 只是 POC。
- Rust 会发 `FullWriteBackOnEvict`，但 Python `_process_evict_actions` 还未处理。
- `postprocess_load_back` 当前可能过早释放 host chain lock，需要把 host source lock 生命周期延到 `loading_check()` ack。
- `evict_host_full` 有 TODO：不能 evict 有 host-backed descendant 的 host ancestor。
- SWA/Mamba/sidecar L2 尚未接入完整 `PoolTransfer` / `HostPoolGroup` / `HybridCacheController` 语义。
- L3/file/mooncake 不在本任务范围内。

## 测试方法要求

先 unit/fuzz，再 server KL/accuracy，再 hit-rate workload。

容量公式：

- `L1 = --max-total-tokens`
- `L2 = L1 * --hicache-ratio`
- multi-turn peak = `clients * rounds * (request_length + output_length)`

日志稳定性必须 grep：

```bash
grep -c 'Sanity check FAILED' "$SERVER_LOG"
grep -c 'Scheduler hit an exception' "$SERVER_LOG"
grep -c 'Traceback' "$SERVER_LOG"
grep -c 'AssertionError' "$SERVER_LOG"
grep -c 'CUDA out of memory' "$SERVER_LOG"
```

HiCache evidence grep：

```bash
grep -c 'load_back' "$SERVER_LOG"
grep -c 'init_load_back' "$SERVER_LOG"
grep -c 'evict_to_host' "$SERVER_LOG"
```

multi-turn 测试使用：

```bash
python3 benchmark/hicache/bench_multiturn.py \
  --model-path "$MODEL_PATH" \
  --disable-random-sample \
  --request-length <REQ> \
  --output-length <OUT> \
  --num-clients <CLIENTS> \
  --num-rounds <ROUNDS> \
  --max-parallel <MAX_PARALLEL> \
  --request-rate <REQUEST_RATE> \
  --ready-queue-policy random \
  --disable-auto-run \
  --enable-round-barrier \
  --port "$PORT" \
  --log-file "$LOG_DIR/multiturn_metrics.jsonl"
```

结果汇报必须分开：

- correctness/stability
- per-round cache hit behavior
- TTFT
- loadback/evict evidence

## Milestone 1: Full-only L1 Rust backend 可构建、可注册、跑通 KL Full

### 研发任务

1. 在 `python/pyproject.toml` 增加 Rust extension：

   - target: `sglang.srt.mem_cache._mem_cache_core`
   - path: `../rust/sglang-mem-cache/Cargo.toml`
   - binding: `PyO3`

2. 确保 editable install 后可 import：

```python
from sglang.srt.mem_cache._mem_cache_core import RustPageRadixCacheWrapper
```

3. 在 mem_cache registry/import 路径懒注册 `install_rust_radix_cache()`。

   - 只有显式传 `--radix-cache-backend rust_unified` 才选择 Rust。
   - extension 缺失时给清晰错误，不影响默认 backend。

4. Rust factory 第一版只允许 Full-only L1：

   - `enable_hierarchical_cache=False`
   - 非 SWA
   - 非 Mamba
   - 非 LMCache
   - LRU only
   - no TTL
   - no KV events

5. 修改：

   - `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_full.py`

   要求：

   - server args 加 `--radix-cache-backend rust_unified`
   - 不依赖 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`
   - model 保持 `Qwen/Qwen3-32B`

### 验收

```bash
cargo test --manifest-path rust/sglang-mem-cache/Cargo.toml

python -m pip install -e python

python - <<'PY'
from sglang.srt.mem_cache._mem_cache_core import RustPageRadixCacheWrapper
from sglang.srt.mem_cache.registry import registered_radix_cache_backends
assert "rust_unified" in registered_radix_cache_backends()
PY

python test/registered/unit/mem_cache/test_unified_radix_cache_bench.py \
  --num-seqs 1000 --verify --components full --page-size 64

python -m pytest \
  test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_full.py \
  -v -s
```

必须看到：

```text
Tree cache initialized: source=registered('rust_unified') impl=RustUnifiedRadixCache
```

## Milestone 2: Full-only L2 HiCache 正确性与命中率

### 研发任务

1. 放开 Rust factory 的 Full-only HiCache gate。

2. 构造 Rust wrapper 时传：

```python
enable_hicache=True
hicache_write_back=(server_args.hicache_write_policy == "write_back")
```

3. write-through：

   - backup 必须保持 root-to-leaf 连续 prefix。
   - parent 未 backup 时，先 backup parent，再 backup child。
   - split/pending backup 后 publish node 不能破坏 host prefix。

4. write-back：

   - Python `_process_evict_actions` 支持 `FullWriteBackOnEvict`。
   - evict device-only leaf 时写 host backup。
   - write-back 不要求 parent backup。
   - loadback 仍只能沿完整 host chain 成功。

5. loadback lock：

   - `prepare_load_back` 锁 host chain 和 device ancestor。
   - H2D source host locks 必须保持到 `loading_check()` ack 后释放。
   - device handoff lock 对 request abort/失败路径也要释放。

6. host eviction：

   - host LRU 只能 evict host leaf，或维护 host-backed-child counter。
   - 禁止 evict 有 host-backed descendant 的 ancestor。

7. 增加 Rust Full-L2 invariant tests：

   - S1/S2/S3 state
   - T2 backup
   - T3 evict_to_host
   - T4 load_back
   - T6 host_evict
   - T8 split redistribution
   - write-through/write-back matrix
   - host pool full retry
   - loadback failed rollback

### 验收

```bash
python -m pytest \
  test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py \
  -k 'hicache and rust and full' -v -s

python test/registered/unit/mem_cache/test_unified_radix_cache_bench.py \
  --num-seqs 5000 --verify --components full --page-size 64
```

Server 测试：

- model: `Qwen/Qwen3-32B`
- 必须使用：

```bash
--radix-cache-backend rust_unified
--enable-hierarchical-cache
--enable-cache-report
--page-size 64
```

分别验证：

```bash
--hicache-write-policy write_through
--hicache-write-policy write_back
```

Hit-rate workload：

- workload A：multi-turn peak 约 `1.5 * L1`
  - `hicache-ratio=2`
  - 最后 2-3 轮 hit rate 不下降
  - 最终 hit rate `>=90%`

- workload B：multi-turn peak 约 `3 * L1`
  - 允许命中率受容量限制
  - 不能 crash
  - 不能出现 invariant failure / traceback / scheduler exception

## Milestone 3: SWA/Mamba/Sidecar L2 能力补齐

### 研发任务

1. 放开 SWA/Mamba HiCache gate。

2. 将 Python Unified 的 component transfer 语义接到 Rust orchestrator：

   - Full KV anchor
   - aux component host values
   - PoolTransfer build/commit
   - sidecar transfer

3. SWA：

   - host-only tombstone match
   - SWA host LRU
   - split redistribution
   - loadback 后 Full/SWA value 一致
   - 保持 `full_lock >= aux_lock`

4. Mamba：

   - chunk-aligned state backup/loadback
   - extra_buffer ping-pong ownership
   - loadback 后 CoW/fork 不 alias cached state

5. Sidecar：

   - 接入 `HostPoolGroup`
   - 接入 `HybridCacheController`
   - 接入 `SidecarPoolSpec`
   - Full KV 作为 anchor
   - INDEXER/DeepSeekV4 side pools 随 backup/loadback 一起迁移

6. Rust `sanity_check()` 覆盖：

   - A2/A3/A4/A5
   - INV-1 到 INV-5
   - 失败信息带 transition ID，方便固化 unit test

### 修改测试

修改并跑通：

- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_swa.py`
  - 加 `--radix-cache-backend rust_unified`

- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py`
  - Mamba L1/L2 类加 `--radix-cache-backend rust_unified`
  - L3 类暂不纳入本计划

- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py`
  - L2/sidecar 类加 `--radix-cache-backend rust_unified`
  - L3/Eagle 类暂不纳入本计划

### 验收

```bash
python test/registered/unit/mem_cache/test_unified_radix_cache_bench.py \
  --num-seqs 5000 --verify --components swa --page-size 64

python test/registered/unit/mem_cache/test_unified_radix_cache_bench.py \
  --num-seqs 5000 --verify --components mamba --page-size 64

python -m pytest \
  test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_swa.py \
  -v -s

python -m pytest \
  test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py \
  -v -s

python -m pytest \
  test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py \
  -k 'HiCache and not L3 and not Eagle' -v -s
```

Sidecar 额外 unit：

- KV anchor hit 但 sidecar miss 时必须截断 host hit。
- sidecar loadback 后 cache report host tokens 与实际可用 prefix 对齐。
- sidecar host eviction 不影响 Full host chain correctness。

L2 hit-rate smoke：

- SWA/Mamba/sidecar 各跑一个 `1.5*L1` multi-turn workload。
- 后几轮 hit rate 不下降，接近理论曲线。
- 各跑一个 `3*L1` stress workload。
- 不能 crash，不能出现 scheduler exception/invariant failure。

## 范围边界

- 本任务只覆盖：
  1. Full-only L1 Rust backend
  2. Full-only L2 HiCache
  3. SWA/Mamba/sidecar L2

- L3/file/mooncake 不在本三步范围内。
- 对 L3 相关配置保持显式 unsupported，不要半开口支持。
- 不要重构无关代码。
- 不要覆盖或 revert 用户已有改动。
- 每个 milestone 完成后必须先跑对应验收，再进入下一个 milestone。
