# Rust UnifiedRadixTree Multi-Stage Implementation Plan

## Goal Description

将当前 SGLang 仓库中的 Rust `UnifiedRadixTree` POC 推进为可按阶段启用的生产候选实现。目标是先让 Rust backend 通过 PyO3 extension 构建、注册并以 Full-only L1 形态跑通 KL Full；再补齐 Full-only L2 HiCache 的 write-through/write-back 正确性、lock 生命周期、host eviction 和命中率验证；最后补齐 SWA、Mamba、sidecar L2 语义并跑通对应 KL、unit、hit-rate smoke。

计划以现有 Python `UnifiedRadixCache` 为行为参考，但不复制其对象模型。所有实现必须保持所有权边界：Rust owns radix tree、node arena、children、per-component state、LRU、lock_ref 和 eviction decision；Python owns KV allocator、req_to_token_pool、host pool、HiCache controller 以及 D2H/H2D transfer。Rust 不直接 free KV，只通过 `DeferredAction` 请求 Python orchestrator 执行 allocator/free/backup/loadback。

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Rust mem-cache PyO3 extension is buildable and importable from the Python package.
  - Positive Tests (expected to PASS):
    - `python/pyproject.toml` contains a `[[tool.setuptools-rust.ext-modules]]` entry for target `sglang.srt.mem_cache._mem_cache_core`, path `../rust/sglang-mem-cache/Cargo.toml`, binding `PyO3`, matching the repo's existing `setuptools-rust` build backend.
    - After `python -m pip install -e python`, this import succeeds:
      ```python
      from sglang.srt.mem_cache._mem_cache_core import RustPageRadixCacheWrapper
      ```
    - `cargo test --manifest-path rust/sglang-mem-cache/Cargo.toml` passes.
  - Negative Tests (expected to FAIL):
    - Removing or misnaming the extension target makes the Python import fail during the import smoke test.
    - A missing native extension must not break default backend import paths; it should only produce a clear error when `--radix-cache-backend rust_unified` is explicitly selected.

- AC-2: `rust_unified` is lazily registered and selected only by explicit backend flag.
  - Positive Tests (expected to PASS):
    - After editable install:
      ```python
      from sglang.srt.mem_cache.registry import registered_radix_cache_backends
      assert "rust_unified" in registered_radix_cache_backends()
      ```
    - A server launched with `--radix-cache-backend rust_unified` logs:
      ```text
      Tree cache initialized: source=registered('rust_unified') impl=RustUnifiedRadixCache
      ```
    - Default backend selection remains unchanged when `--radix-cache-backend` is omitted.
  - Negative Tests (expected to FAIL):
    - `--radix-cache-backend rust_unified` with a missing extension fails fast with an actionable error naming `_mem_cache_core`.
    - `--radix-cache-backend rust_unified` rejects unsupported configurations for the current milestone instead of silently falling back.

- AC-3: Milestone 1 supports only Full-only L1 with conservative gates.
  - Positive Tests (expected to PASS):
    - `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_full.py` launches `Qwen/Qwen3-32B` with `--radix-cache-backend rust_unified` and without `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`.
    - `python test/registered/unit/mem_cache/test_unified_radix_cache_bench.py --num-seqs 1000 --verify --components full --page-size 64` passes while actually exercising the Rust implementation. If the current bench still instantiates only Python `UnifiedRadixCache`, add a Rust selector/config first.
    - `python -m pytest test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_full.py -v -s` passes.
  - Negative Tests (expected to FAIL):
    - In Milestone 1, Rust factory rejects HiCache, SWA, Mamba, LMCache, non-LRU eviction, TTL, KV events, and other out-of-scope modes with `RadixCacheInfraPyError`.
    - The KL Full test fails if it accidentally runs Python Unified via `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` instead of the registered Rust backend.

- AC-4: Full-only L2 HiCache write-through preserves a contiguous root-to-leaf host prefix.
  - Positive Tests (expected to PASS):
    - Rust Full-L2 invariant tests cover S1/S2/S3, T2 backup, T8 split redistribution, pending write-through publish, and root-to-leaf host prefix preservation.
    - Parent backup is attempted before child backup when write policy is write-through or write-through selective.
    - Host stamps are published only for successfully backed-up nodes; partial backup failures leave no host gap.
  - Negative Tests (expected to FAIL):
    - A child host backup without a host-backed non-root parent is rejected by Rust with a typed error.
    - Splitting a pending write-through node must not publish only the suffix and leave the prefix unbacked.

- AC-5: Full-only L2 HiCache write-back backs up device-only evictions correctly.
  - Positive Tests (expected to PASS):
    - Python `_process_evict_actions` handles `FullWriteBackOnEvict` by writing the evicted device value to host and committing host state on success.
    - Device-only leaf eviction under write-back transitions to host-only state when host capacity is available.
    - Existing backed-up device eviction still frees only device KV through `FullDeviceEvictOnBackedUp`.
  - Negative Tests (expected to FAIL):
    - `FullWriteBackOnEvict` must not be reported as an unsupported evict action.
    - Failed write-back due to host pool exhaustion must not drop device value state without either retrying host eviction or rolling back safely.

- AC-6: Full-only L2 loadback lock ownership is correct through ack and failure paths.
  - Positive Tests (expected to PASS):
    - `prepare_load_back` locks the host source chain and the device ancestor.
    - H2D host source locks remain held until `loading_check()` observes the ack and releases them.
    - Device handoff locks are released on successful ack, request abort, scheduler failure, and loadback rollback.
  - Negative Tests (expected to FAIL):
    - A loadback source host value cannot be host-evicted while its H2D copy is in flight.
    - A failed `cache_controller.load(...)` must not leak device or host locks.

- AC-7: Full-only L2 host eviction never breaks host-backed descendant loadback chains.
  - Positive Tests (expected to PASS):
    - Host eviction evicts only host leaves, or uses an equivalent host-backed-child counter to skip host ancestors with backed descendants.
    - Tests cover T6 host eviction and then T4 loadback from surviving descendants.
  - Negative Tests (expected to FAIL):
    - Evicting a host-backed ancestor while a descendant remains host-backed is rejected or skipped.
    - `PrepareLoadBackMissingHostValue` must not occur in normal host eviction/loadback workloads.

- AC-8: Full-only L2 passes unit, bench, server stability, and hit-rate acceptance.
  - Positive Tests (expected to PASS):
    - `python -m pytest test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py -k 'hicache and rust and full' -v -s` passes.
    - `python test/registered/unit/mem_cache/test_unified_radix_cache_bench.py --num-seqs 5000 --verify --components full --page-size 64` passes.
    - Server tests pass for both `--hicache-write-policy write_through` and `--hicache-write-policy write_back` with `--radix-cache-backend rust_unified --enable-hierarchical-cache --enable-cache-report --page-size 64`.
    - Workload A, with multi-turn peak about `1.5 * L1` and `hicache-ratio=2`, has no late-round hit-rate decline and final hit rate `>=90%`.
    - Workload B, with multi-turn peak about `3 * L1`, does not crash and has no invariant failure, traceback, scheduler exception, assertion error, or CUDA OOM.
  - Negative Tests (expected to FAIL):
    - Any nonzero count from the required stability grep commands fails acceptance.
    - Missing `load_back`, `init_load_back`, or `evict_to_host` evidence in L2 workloads fails evidence collection even if the server does not crash.

- AC-9: SWA L2 integrates Full anchor, SWA host values, locking, split redistribution, and hit truncation semantics.
  - Positive Tests (expected to PASS):
    - `python test/registered/unit/mem_cache/test_unified_radix_cache_bench.py --num-seqs 5000 --verify --components swa --page-size 64` passes.
    - `python -m pytest test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_swa.py -v -s` passes with `--radix-cache-backend rust_unified`.
    - SWA unit tests cover host-only tombstone match, SWA host LRU, split redistribution, loadback Full/SWA value consistency, and `full_lock >= aux_lock`.
  - Negative Tests (expected to FAIL):
    - A SWA host hit without a valid Full anchor is rejected or truncated.
    - Device or host LRU double-membership for the same SWA value is detected by `sanity_check()`.

- AC-10: Mamba L2 integrates chunk-aligned backup/loadback and state ownership.
  - Positive Tests (expected to PASS):
    - `python test/registered/unit/mem_cache/test_unified_radix_cache_bench.py --num-seqs 5000 --verify --components mamba --page-size 64` passes.
    - `python -m pytest test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py -v -s` passes for Mamba L1/L2 classes with `--radix-cache-backend rust_unified`.
    - Tests cover chunk-aligned state backup/loadback, extra_buffer ping-pong ownership, and post-loadback CoW/fork non-aliasing.
  - Negative Tests (expected to FAIL):
    - Non-chunk-aligned Mamba host state must not be committed as a valid host hit.
    - A loaded Mamba cached state must not alias a request-owned mutable state after CoW/fork.

- AC-11: Sidecar L2 integrates `HostPoolGroup`, `HybridCacheController`, and `SidecarPoolSpec` with Full KV as anchor.
  - Positive Tests (expected to PASS):
    - `python -m pytest test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py -k 'HiCache and not L3 and not Eagle' -v -s` passes with Rust backend on L2/sidecar classes.
    - Sidecar unit tests verify KV anchor hit with sidecar miss truncates host hit, sidecar loadback cache report host tokens match usable prefix, and sidecar host eviction does not break Full host chain correctness.
    - INDEXER/DeepSeekV4 side pools migrate with backup/loadback operations through `PoolTransfer` and `HybridCacheController`.
  - Negative Tests (expected to FAIL):
    - Sidecar hit longer than the valid Full KV host prefix is rejected or truncated.
    - Sidecar host eviction must not remove required Full host chain state.

- AC-12: Rust `sanity_check()` covers L2 cross-component invariants with actionable failures.
  - Positive Tests (expected to PASS):
    - `sanity_check()` covers A2/A3/A4/A5 and INV-1 through INV-5 as defined by the implementation.
    - Failure messages include transition IDs such as T2, T3, T4, T6, or T8 where applicable.
  - Negative Tests (expected to FAIL):
    - Artificially constructed host prefix gaps, `full_lock < aux_lock`, host/device LRU double-membership, or aux-without-Full-anchor states are detected.

- AC-13: Scope boundaries for L3/file/mooncake/Eagle remain explicit.
  - Positive Tests (expected to PASS):
    - L3/file/mooncake storage configs remain unsupported for the Rust backend in this plan.
    - L3 and Eagle KL classes are not modified as part of Milestone 3 acceptance.
  - Negative Tests (expected to FAIL):
    - Any partial L3 support path that can be selected without complete tests fails review.
    - Any fallback from Rust backend to Python backend after `--radix-cache-backend rust_unified` is specified fails acceptance.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation completes all three milestones: Rust extension packaging and lazy registry selection; Full-only L1 KL correctness; Full-only L2 HiCache write-through/write-back correctness with lock-safe loadback and host eviction; SWA/Mamba/sidecar L2 integration through existing Python pool transfer infrastructure; Rust invariant coverage; unit, bench, KL, stability grep, and hit-rate workload evidence for each milestone.

The upper bound may include additional Rust unit/fuzz/property tests for tree invariants and transition matrices if they directly support the listed acceptance criteria. It may include small helper abstractions for deferred actions, host lock bookkeeping, or sidecar transfer construction when they reduce real duplication and match existing module boundaries.

### Lower Bound (Minimum Acceptable Scope)

The minimum viable implementation for each milestone is sequential and non-skippable:

1. Milestone 1 must make `rust_unified` buildable, importable, registered, explicitly selectable, and KL Full runnable under Full-only L1.
2. Milestone 2 must make Full-only L2 HiCache correct for both write-through and write-back, including lock lifecycle and host eviction correctness, before any SWA/Mamba/sidecar HiCache gate is opened.
3. Milestone 3 must make SWA, Mamba, and sidecar L2 pass their targeted unit/KL/smoke tests while keeping L3/file/mooncake unsupported.

If any milestone acceptance fails, the implementation stops at that milestone and does not broaden the enabled factory gates.

### Allowed Choices

- Can use: existing `setuptools-rust`/PyO3 packaging pattern in `python/pyproject.toml`; existing `register_radix_cache_backend` registry; existing Python `UnifiedRadixCache` component and HiCache semantics as reference; existing `PoolTransfer`, `HostPoolGroup`, `HybridCacheController`, `SidecarPoolSpec`; Rust-side node arena/LRU/component ownership; Python-side allocator and transfer orchestration.
- Can choose: host eviction via host-leaf-only structural predicate or host-backed-child counter, as long as it is correct and tested; small Python helper classes for in-flight write/load lock bookkeeping; Rust unit tests, Python unit tests, and benchmark verifier extensions.
- Cannot use: Rust-side direct KV free/allocator ownership; scheduler code that introspects Rust `NodeIdx` as a Python `UnifiedTreeNode`; implicit fallback to Python cache after explicit Rust backend selection; partial L3/file/mooncake support; unrelated refactors; broad scheduler rewrites; changes that revert user work.

## Feasibility Hints and Suggestions

> This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Milestone 1 should be treated as packaging and routing first, not HiCache work. The current repo already uses `setuptools-rust` for `sglang.srt.grpc._core`, so add the mem-cache extension using the same mechanism. Because `python/sglang/srt/mem_cache/rust_unified_radix_cache.py` currently imports `_mem_cache_core` at module import time, make registration safe by deferring the native import or by storing a clear import error that is raised only when the Rust backend is selected. The current benchmark verifier primarily constructs Python `UnifiedRadixCache`; make the Rust benchmark path explicit before counting it as Rust acceptance.

Milestone 2 should center on host/device state transitions. Model Full nodes as S0/S1/S2/S3 and test T2, T3, T4, T6, and T8 directly in Rust where possible. Keep Python as the orchestrator for host pool allocation and D2H/H2D transfer. The most important implementation risk is lock lifetime: current Rust `postprocess_load_back` releases host chain locks immediately, but acceptance requires source host locks to survive until `loading_check()` ack.

Milestone 3 should port transfer semantics rather than duplicating Python object structure. Build component and sidecar `PoolTransfer` sets in Python where the pools live, then commit only after Rust and transfer controller agree on the usable prefix. Full remains the anchor for every aux component.

### Relevant References

- `rust/sglang-mem-cache/DESIGN.md` - Current Rust/Python ownership split and deferred action contract.
- `python/pyproject.toml` - Existing `setuptools-rust` pattern for PyO3 extensions.
- `python/sglang/srt/mem_cache/registry.py` - Backend registry and `create_tree_cache` log line.
- `python/sglang/srt/mem_cache/rust_unified_radix_cache.py` - Rust orchestrator, factory, HiCache gate, `_process_evict_actions`, `_write_backup`, `init_load_back`, `loading_check`.
- `rust/sglang-mem-cache/src/radix_cache.rs` - Rust core loadback, host value stamping, lock accounting.
- `rust/sglang-mem-cache/src/tree_node_lru.rs` - Full/device eviction, write-back deferred action, host LRU eviction.
- `python/sglang/srt/mem_cache/unified_radix_cache.py` - Reference write_backup/load_back/component/sidecar semantics.
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` - Existing multi-pool write/load transfer controller.
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` - Existing `HostPoolGroup` assembly for SWA/Mamba/sidecar.
- `test/registered/radix_cache/unified_radix_tree/` - KL tests that must be switched to explicit Rust backend only where in scope.
- `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` and `test_unified_radix_cache_bench.py` - Unit and verifier entry points to extend.

## Dependencies and Sequence

### Milestones

1. Milestone 1: Full-only L1 Rust backend is buildable, registered, and KL Full capable.
   - Phase A: Add PyO3 extension packaging and native import smoke.
   - Phase B: Make `install_rust_radix_cache()` safe to invoke from registry/import paths.
   - Phase C: Add conservative Full-only L1 factory gates and explicit errors.
   - Phase D: Switch KL Full test to `--radix-cache-backend rust_unified`, remove `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`, and make the Full bench verifier exercise the Rust implementation.
   - Phase E: Run cargo, editable install, import smoke, bench verifier, and KL Full acceptance.

2. Milestone 2: Full-only L2 HiCache is correct for write-through/write-back and has hit-rate evidence.
   - Phase A: Open only the Full-only HiCache factory gate and pass `enable_hicache=True` plus `hicache_write_back=(server_args.hicache_write_policy == "write_back")`.
   - Phase B: Fix write-through prefix backup and split/pending publish behavior.
   - Phase C: Implement `FullWriteBackOnEvict` orchestration and rollback/retry behavior.
   - Phase D: Extend loadback bookkeeping so host source locks and device handoff locks release on ack or failure.
   - Phase E: Fix host eviction to avoid evicting host ancestors with backed descendants.
   - Phase F: Add Full-L2 invariant tests, run unit/bench/server/write-policy/hit-rate acceptance.

3. Milestone 3: SWA/Mamba/sidecar L2 are integrated and validated.
   - Phase A: Open SWA/Mamba HiCache gates only after Full-only L2 acceptance passes.
   - Phase B: Connect component `PoolTransfer` build/commit semantics to the Rust orchestrator.
   - Phase C: Implement SWA host-only tombstone, host LRU, split redistribution, loadback consistency, and lock invariants.
   - Phase D: Implement Mamba chunk-aligned state backup/loadback, extra_buffer ownership, and CoW/fork non-aliasing.
   - Phase E: Integrate sidecar pools with `HostPoolGroup`, `HybridCacheController`, and `SidecarPoolSpec`.
   - Phase F: Extend `sanity_check()` and run SWA/Mamba/DSV4 L2 KL, verifier, and hit-rate smoke acceptance.

Milestones are strictly ordered. Do not enable a later milestone's factory gate before the previous milestone has passing acceptance evidence.

## Task Breakdown

Each task includes exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Inspect current Rust crate, Python wrapper, registry, pyproject, and Full KL test; record exact starting gaps. | AC-1, AC-2, AC-3 | analyze | - |
| task2 | Add `sglang.srt.mem_cache._mem_cache_core` PyO3 extension to `python/pyproject.toml` using existing `setuptools-rust` style. | AC-1 | coding | task1 |
| task3 | Make native import and `install_rust_radix_cache()` lazy/safe so default backend imports do not require the extension. | AC-2 | coding | task2 |
| task4 | Add Full-only L1 Rust factory gates and clear unsupported errors. | AC-3, AC-13 | coding | task3 |
| task5 | Update KL Full test to use `--radix-cache-backend rust_unified` without `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`, and add a Rust Full bench/verifier path if the current bench only instantiates Python `UnifiedRadixCache`. | AC-3 | coding | task4 |
| task6 | Run Milestone 1 cargo, editable install, import smoke, bench verifier, and KL Full acceptance; summarize failures before proceeding. | AC-1, AC-2, AC-3 | analyze | task5 |
| task7 | Design Full-L2 Rust state-transition tests for S1/S2/S3 and T2/T3/T4/T6/T8. | AC-4, AC-5, AC-6, AC-7 | analyze | task6 |
| task8 | Open Full-only HiCache gate and pass `enable_hicache`/`hicache_write_back` into the Rust wrapper. | AC-8, AC-13 | coding | task7 |
| task9 | Fix write-through backup ordering, pending backup split redistribution, and host-prefix publish behavior. | AC-4 | coding | task8 |
| task10 | Implement Python orchestration for `FullWriteBackOnEvict`, including host allocation retry and safe rollback behavior. | AC-5 | coding | task9 |
| task11 | Extend loadback in-flight bookkeeping so host locks and device handoff locks release only on ack or failure cleanup. | AC-6 | coding | task10 |
| task12 | Make host eviction host-leaf-safe or counter-safe and add descendant-chain tests. | AC-7 | coding | task11 |
| task13 | Add Full-L2 Rust/Python invariant tests and bench verifier coverage. | AC-4, AC-5, AC-6, AC-7, AC-8 | coding | task12 |
| task14 | Run Milestone 2 unit, bench, server write-through/write-back, stability grep, evidence grep, and hit-rate workloads. | AC-8 | analyze | task13 |
| task15 | Map Python Unified component/sidecar transfer semantics to the Rust orchestrator boundary. | AC-9, AC-10, AC-11 | analyze | task14 |
| task16 | Implement SWA L2 host state, LRU, split redistribution, loadback consistency, and lock invariants. | AC-9, AC-12 | coding | task15 |
| task17 | Implement Mamba L2 chunk-aligned backup/loadback, extra_buffer ownership, and CoW/fork isolation. | AC-10, AC-12 | coding | task16 |
| task18 | Implement sidecar L2 integration through `HostPoolGroup`, `HybridCacheController`, and `SidecarPoolSpec`. | AC-11, AC-12 | coding | task17 |
| task19 | Extend Rust `sanity_check()` and failure messages with transition IDs. | AC-12 | coding | task18 |
| task20 | Update in-scope SWA/Mamba/DSV4 KL tests to use `--radix-cache-backend rust_unified`; leave L3/Eagle out of scope. | AC-9, AC-10, AC-11, AC-13 | coding | task19 |
| task21 | Run Milestone 3 verifier, KL, stability grep, evidence grep, and L2 hit-rate smoke/stress workloads. | AC-9, AC-10, AC-11, AC-12, AC-13 | analyze | task20 |

## Claude-Codex Deliberation

### Agreements

- The Rust implementation should not mechanically copy Python `UnifiedRadixCache`; it should preserve the Rust/Python ownership split already documented in `DESIGN.md`.
- Packaging and registry must land before any runtime behavior is trusted.
- Factory gates should open only after the corresponding acceptance tests pass.
- Full is the anchor for all L2 aux and sidecar behavior.
- L3/file/mooncake and Eagle-specific KL coverage are outside this three-milestone plan.

### Resolved Disagreements

- Packaging mechanism: use the repo's existing `setuptools-rust` extension configuration rather than introducing a separate build system. This follows the current `python/pyproject.toml` pattern.
- Metric interpretation: numeric thresholds in the draft, especially final hit rate `>=90%` for Workload A, are treated as hard acceptance because they are under "必须/验收" language.
- Host eviction implementation shape: either host-leaf-only eviction or a host-backed-child counter is acceptable. Correctness and tests determine the choice.

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- None. The draft is treated as complete and self-contained. Quantitative thresholds are treated as hard acceptance requirements unless later explicitly relaxed by the user.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
- Do not expose Rust tree internals through Python request objects; `req.last_node` remains an opaque Rust `NodeIdx`.
- Do not bypass the deferred action contract by freeing KV from Rust.

### Test and Evidence Reporting

For every server or hit-rate run, report results under separate headings:

- correctness/stability
- per-round cache hit behavior
- TTFT
- loadback/evict evidence

Any `test_unified_radix_cache_bench.py` command in this plan counts as Rust backend evidence only if the benchmark path instantiates `RustUnifiedRadixCache` or the Rust wrapper for the requested component set. A Python `UnifiedRadixCache` run may be useful as a reference comparison, but it does not satisfy Rust acceptance.

Run these stability greps against each server log and require zero matches:

```bash
grep -c 'Sanity check FAILED' "$SERVER_LOG"
grep -c 'Scheduler hit an exception' "$SERVER_LOG"
grep -c 'Traceback' "$SERVER_LOG"
grep -c 'AssertionError' "$SERVER_LOG"
grep -c 'CUDA out of memory' "$SERVER_LOG"
```

Run these evidence greps for HiCache workloads and include the counts in the report:

```bash
grep -c 'load_back' "$SERVER_LOG"
grep -c 'init_load_back' "$SERVER_LOG"
grep -c 'evict_to_host' "$SERVER_LOG"
```

Use the multi-turn driver with workload parameters chosen so:

- `L1 = --max-total-tokens`
- `L2 = L1 * --hicache-ratio`
- `multi-turn peak = clients * rounds * (request_length + output_length)`

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
