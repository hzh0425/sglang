# UnifiedRadixCache HiCache L3 Implementation Notes
> Project: unified radix tree + HiCache storage
> Status: L3 data plane implemented for startup configuration; Unified runtime attach/detach is still pending in the current checkout.
> Last updated: 2026-05-16

## Goal

`UnifiedRadixCache` should support the same three-tier cache model as HiCache:

- L1: device memory
- L2: host memory
- L3: storage backend, for example `file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, or `eic`

The design goal is not only to make Full KV work, but to preserve the unified-tree component model:

- `Full`, `SWA`, and `Mamba` each define their own transfer and lock semantics.
- The tree owns radix structure, request bookkeeping, and state transitions.
- The controller owns storage backend lifecycle, transfer queues, backend I/O, and host-pool registration.
- Component-specific behavior stays inside `TreeComponent` implementations, not in the tree main path.

## Current Progress

| Area | Current status | Notes |
|------|----------------|-------|
| Startup L3 enablement | Implemented | `--hicache-storage-backend ...` works through `UnifiedRadixCache.init_hicache()` and `HybridCacheController`. |
| Full KV L3 backup/prefetch | Implemented | Full host pages are the storage backbone and use node page hashes as keys. |
| Mamba sidecar L3 transfer | Implemented | Uses trailing-page semantics, one host state slot per matched suffix. |
| SWA sidecar L3 transfer | Implemented | Uses trailing window pages only; host window ownership is component-managed. |
| L3 -> L2 -> L1 replay | Implemented | Storage prefetch materializes host nodes, then normal load-back can move data to device. |
| Host lock model | Implemented | `IncLockRefResult` / `DecLockRefParams` carry host-side SWA boundary metadata. |
| Runtime storage clear | Implemented | `clear_storage_backend()` calls controller-level clear when backend supports it. |
| Unified runtime attach/detach | Not implemented in current checkout | `UnifiedRadixCache.attach_storage_backend()` and `detach_storage_backend()` still return unsupported. HiRadix/HiMamba runtime APIs are separate and already exist. |
| Validation harness | In progress | `validte_l3_hicache.sh` records accuracy and multiturn validation flow, but it is currently untracked and should be cleaned up before merge. |

## Important Code Paths

### Initialization

The scheduler selects `UnifiedRadixCache` when `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`:

```text
Scheduler
  -> UnifiedRadixCache(params)
  -> init_hicache(server_args, params)
  -> attach_hybrid_pool_to_unified_cache(...)
  -> HybridCacheController(...)
```

Files:

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`

`attach_hybrid_pool_to_unified_cache()` builds the host pool group and registers sidecar pool specs for the selected component set:

- `FULL`
- `FULL + SWA`
- `FULL + MAMBA`

The current pool assembler still treats `FULL + SWA + MAMBA` as a future extension.

### Controller Boundary

`HybridCacheController` is the owner of storage I/O and host-pool transfer mechanics:

- Creates storage backends through `StorageBackendFactory`.
- Registers all host pools with `register_mem_host_pool_v2`.
- Owns backup and prefetch queues.
- Moves KV and sidecar pool indices between execution-local and host-pool-local coordinate spaces.
- Performs `batch_exists_v2`, `batch_get_v2`, and `batch_set_v2` when sidecar transfers are present.
- Applies rate limiting and prefetch timeout policy support.

`UnifiedRadixCache` should not contain backend-specific lifecycle code. It only calls:

- `cache_controller.write(...)` for L1 -> L2
- `cache_controller.load(...)` for L2 -> L1
- `cache_controller.write_storage(...)` for L2 -> L3
- `cache_controller.prefetch(...)` and `terminate_prefetch(...)` for L3 -> L2
- `cache_controller.append_host_mem_release(...)` for async host-page release

### Tree Boundary

`UnifiedRadixCache` owns state that is inseparable from the radix tree:

- `ongoing_write_through`
- `ongoing_load_back`
- `ongoing_backup`
- `ongoing_prefetch`
- `prefetch_loaded_tokens_by_reqid`
- host-only node insertion and split logic
- host lock acquire/release against tree nodes
- LRU updates after component commit
- cleanup for revoke, abort, timeout, and completed transfer events

This is why `write_backup_storage()`, `prefetch_from_storage()`, `check_prefetch_progress()`, and `drain_storage_control_queues()` remain in the tree. They are not backend lifecycle logic; they are tree-state transitions after controller I/O completes.

### Component Boundary

Component-specific L3 behavior is expressed through hooks:

```python
def build_hicache_transfers(
    self,
    node: UnifiedTreeNode,
    phase: CacheTransferPhase,
    **kw,
) -> Optional[list[PoolTransfer]]:
    ...

def commit_hicache_transfer(
    self,
    node: UnifiedTreeNode,
    phase: CacheTransferPhase,
    transfers: list[PoolTransfer] = (),
    **kw,
) -> None:
    ...
```

The tree does not special-case `Mamba` or `SWA` in the main transfer flow. It asks each component to build descriptors, submits those descriptors through the controller, then asks each component to commit or free the transferred resources.

## Data Structures

### ComponentData

Each node stores device and host state independently per component:

```python
@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    host_value: Optional[torch.Tensor] = None
    host_lock_ref: int = 0
```

`value` points to L1/device indices. `host_value` points to L2/host indices. A node may have either, both, or neither depending on eviction and transfer state, but dead host-only nodes must not remain in the radix tree.

### CacheTransferPhase

The unified component hooks use one enum for all HiCache movements:

```python
class CacheTransferPhase(str, Enum):
    BACKUP_HOST = "backup_host"       # L1 -> L2
    LOAD_BACK = "load_back"           # L2 -> L1
    BACKUP_STORAGE = "backup_storage" # L2 -> L3
    PREFETCH = "prefetch"             # L3 -> L2
```

### PoolTransfer

`PoolTransfer` describes an auxiliary pool transfer, such as Mamba state or SWA KV. Full KV is still the anchor transfer.

Important fields:

- `name`: target pool name, for example `PoolName.MAMBA` or `PoolName.SWA`.
- `host_indices`: host-side indices for L2/L3 transfer.
- `device_indices`: device-side indices for L1/L2 transfer.
- `keys`: storage keys for L3 sidecar objects.
- `hit_policy`: storage hit policy, usually `TRAILING_PAGES` for sidecar pools.
- `indices_from_pool`: optional source pool when a sidecar transfer derives indices from the KV anchor.

`HybridCacheController` normalizes these descriptors before executing an I/O operation. The tree-owned descriptors are intentionally left unchanged because they may still reference tree state for later commit or cleanup.

### Host Lock Result

Host locks reuse the existing request lock plumbing:

```python
@dataclasses.dataclass
class IncLockRefResult:
    delta: Optional[int] = None
    swa_uuid_for_lock: Optional[int] = None
    swa_uuid_for_host_lock: Optional[int] = None

@dataclasses.dataclass
class DecLockRefParams:
    swa_uuid_for_lock: Optional[int] = None
    swa_uuid_for_host_lock: Optional[int] = None
```

This avoids adding a parallel host-only lock result type. `to_dec_params()` carries both device and host SWA UUIDs so acquire and release remain paired.

### InsertResult Host Fields

Host insertion reuses `InsertResult`:

```python
@dataclasses.dataclass
class InsertResult:
    prefix_len: int
    mamba_exist: bool = False
    total_len: int = 0
    inserted_host_node: Any = None
```

Host semantics:

- `prefix_len`: host prefix already matched in the local tree.
- `total_len`: total Full host length passed into `_insert_helper_host()`.
- `inserted_host_node`: new host leaf that owns `[prefix_len, total_len)`, or `None` if nothing new was inserted.
- `mamba_exist`: whether the prefetched Mamba host state should be discarded because a usable host Mamba state already exists or no new host suffix was inserted.

## Host Lock Semantics

Host locks are intentionally not identical to device locks.

| Component | Device lock | Host lock |
|-----------|-------------|-----------|
| Full | Path lock to root | Single-node host lock, HiRadix-compatible |
| Mamba | Single-node lock | Single-node lock plus host LRU detach/reattach |
| SWA | Window lock with `metadata["uuid"]` | Window lock with independent `metadata["host_uuid"]` |

Rationale:

- Full host eviction only deletes host leaves, so ancestor protection is mostly structural.
- Mamba host state is a sidecar pool with independent LRU eviction; it must be pinned while storage I/O is in flight.
- SWA host data is window-scoped, so it needs its own host window boundary and cannot reuse the device SWA UUID.

## Transfer Semantics

### L1 -> L2: `write_backup()`

The tree builds one Full KV transfer and asks non-Full components for `BACKUP_HOST` sidecar transfers.

Commit order:

1. Controller allocates host pages and starts D->H copy.
2. Full component records `host_value`.
3. Sidecar components record their own `host_value`.
4. Tree records `ongoing_write_through` and locks the node if write-through completion needs protection.

### L2 -> L1: `load_back()`

The tree starts from the last host-hit node:

1. Full component builds the host KV range to load.
2. Mamba/SWA components build `LOAD_BACK` sidecar transfers.
3. Tree locks the ancestor path and evicts device memory if needed.
4. Controller performs H->D copy for KV and sidecar pools.
5. Components commit device indices.
6. Tree records `ongoing_load_back` until controller ack.

### L2 -> L3: `write_backup_storage()`

This method stays in `UnifiedRadixCache` because it must pin the exact host node and release it only after the storage ack is drained.

Flow:

1. Ensure node hash values exist.
2. Optionally build `prefix_keys` when `hicache_storage_pass_prefix_keys=True`.
3. Ask components for `BACKUP_STORAGE` transfers.
4. Add sidecar-derived transfers where needed.
5. Submit `cache_controller.write_storage(...)`.
6. Store `(node, host_dec_params)` in `ongoing_backup`.

Full KV uses `node.hash_value` as per-page storage keys. Sidecar components usually use trailing-page keys because sidecar states are only meaningful near the matched suffix.

### L3 -> L2: `prefetch_from_storage()` and `check_prefetch_progress()`

`prefetch_from_storage()` runs after local match when the unmatched suffix is large enough:

1. Page-align the suffix.
2. Check threshold and controller rate limit.
3. Host-lock `last_host_node`.
4. Allocate Full host pages, evicting host memory if necessary.
5. Ask components for `PREFETCH` transfers and allocate sidecar host slots.
6. Submit controller prefetch.
7. Record `(last_host_node, prefetch_key, host_indices, operation, host_lock_params, comp_xfers)`.

`check_prefetch_progress()` completes the transfer:

1. Apply `best_effort`, `wait_complete`, or `timeout` termination policy.
2. Synchronize completed tokens across TP ranks with `all_reduce(min)`.
3. Insert the fetched Full host suffix with `_insert_helper_host()`.
4. Ask sidecar components to commit or free prefetched host slots.
5. Free matched-prefix Full host slots and release unused tail pages through controller release queues.
6. Release the host lock.
7. Record request-level `prefetch_loaded_tokens_by_reqid`.

The key invariant is that Full KV materializes the host tree skeleton first; sidecar components only attach to the resulting host nodes.

## Component Details

### FullComponent

Full is the radix backbone.

- `BACKUP_HOST`: records host KV indices on the node.
- `LOAD_BACK`: loads host KV indices from the matched host path.
- `BACKUP_STORAGE`: the tree submits Full host pages directly as the anchor payload.
- `PREFETCH`: the tree allocates Full host pages and inserts the fetched suffix; Full commit is effectively tree-owned.
- Host lock is a single-node lock.

### MambaComponent

Mamba state is leaf-oriented.

- `BACKUP_HOST`: backs up the node's Mamba device state into a host Mamba slot.
- `LOAD_BACK`: loads the Mamba state for the matched node.
- `BACKUP_STORAGE`: writes one trailing-page Mamba object keyed by the last page hash.
- `PREFETCH`: preallocates one host Mamba slot. Commit adopts it only when `_insert_helper_host()` created a new host leaf and storage actually hit the Mamba object.

If no new host leaf was inserted, or the storage sidecar hit is missing, the prefetched Mamba host slot is freed.

### SWAComponent

SWA is window-oriented.

- `BACKUP_HOST`: backs up the host-visible trailing SWA KV pages.
- `LOAD_BACK`: loads the trailing SWA window needed by the request.
- `BACKUP_STORAGE`: writes only trailing window pages.
- `PREFETCH`: preallocates host slots for `min(prefetch_len, sliding_window_size)` trailing pages.

SWA commit uses `InsertResult.prefix_len`, `InsertResult.total_len`, and `InsertResult.inserted_host_node` to compute the overlap between the newly inserted Full suffix and the SWA trailing window. If the new host leaf crosses the SWA window boundary, the component can request a secondary split so SWA host data stays window-aligned.

## Storage Key Rules

Full keys:

- One hash per page.
- Hash chain starts from `last_hash` for prefetch and from the node parent for backup.
- `node.hash_value` is maintained for L3-enabled nodes.

Sidecar keys:

- Mamba uses the last hit page key.
- SWA uses the last N hit page keys where N is the trailing SWA window in pages.
- `HybridCacheController._sync_trailing_keys()` realigns sidecar keys after the storage KV hit is truncated.

Prefix keys:

- Disabled by default.
- Enabled with `hicache_storage_pass_prefix_keys` in storage extra config.
- Used by backends that need the entire prefix chain to disambiguate objects.

## Runtime Attach/Detach Status

HiRadixCache and HiMambaRadixCache have runtime attach/detach implementations.

In the current checkout, `UnifiedRadixCache` still has these stubs:

```python
def attach_storage_backend(...):
    return False, "UnifiedRadixCache does not support runtime HiCache storage attach yet..."

def detach_storage_backend(...):
    return False, "UnifiedRadixCache does not support runtime HiCache storage detach yet..."
```

The intended implementation boundary is:

- The scheduler still calls `tree_cache.attach_storage_backend(...)` and `detach_storage_backend(...)` after strict idle checks.
- Unified tree should keep only thin wrappers for API compatibility.
- The actual lifecycle should be centralized in `HybridCacheController`, including backend creation, host-pool registration, thread startup/shutdown, config parsing, policy updates, and state rollback on failure.
- Tree-side wrappers should only apply runtime config to tree fields and reset storage bookkeeping if the controller operation succeeds.

This keeps attach/detach out of the tree data path while preserving the existing admin API surface.

## Observability

Useful logs:

- `HiCache prefetch queued`: request queued for L3 prefetch.
- `HiCache prefetch success`: L3 prefetch completed and host tree was updated.
- `load_back committed`: L2 -> L1 transfer committed.
- `storage backup queued` / `storage backup acked`: L2 -> L3 write path.
- `split in-flight cached node`: host/device node split happened while transfer state existed.

Useful counters:

- `cached_tokens_storage` on request usage reports tokens loaded from L3.
- `prefetch_loaded_tokens_by_reqid` records per-request L3-loaded token count inside the tree.
- `StorageMetricsCollector` records storage prefetch and backup tokens when metrics are enabled.

## Correctness Validation

Validation should prove three different properties:

- Functional correctness: L3 replay does not change generated answers.
- State-machine correctness: every L3 hit goes through valid L3 -> L2 -> L1 transitions and releases locks/pages exactly once.
- Performance correctness: when the working set exceeds L2 capacity, L3 improves cache hit rate compared with L2-only.

### 1. Static and unit checks

Run these before starting a real server:

```bash
python3 -m py_compile \
  python/sglang/srt/mem_cache/unified_radix_cache.py \
  python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py \
  python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py \
  python/sglang/srt/mem_cache/unified_cache_components/tree_component.py \
  python/sglang/srt/mem_cache/unified_cache_components/full_component.py \
  python/sglang/srt/mem_cache/unified_cache_components/swa_component.py \
  python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py

FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m pytest -q test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py

FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m pytest -q test/registered/unit/managers/test_scheduler_flush_cache.py

git diff --check
```

What this catches:

- component hook regressions
- lock/unlock counter underflow in unit scenarios
- host/device pool accounting bugs exposed by the unit cache checker
- formatting problems such as trailing whitespace

### 2. Real accuracy replay test

Use a real model server, enable Unified Radix Tree, enable HiCache, and configure L3 at startup:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/data/hicache

python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-write-policy write_through \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete \
  --max-mamba-cache-size 100 \
  --max-total-tokens 1310720 \
  --port 30000 \
  --tp 2
```

Then run the same accuracy workload twice:

```bash
python benchmark/gsm8k/bench_sglang.py \
  --port 30000 --num-questions 200 --num-shots 6 --parallel 50

curl -s -X POST http://127.0.0.1:30000/flush_cache

python benchmark/gsm8k/bench_sglang.py \
  --port 30000 --num-questions 200 --num-shots 6 --parallel 50
```

Pass criteria:

- round 1 and round 2 accuracy differ by no more than 2 percentage points
- both rounds stay above the expected model baseline for the selected benchmark
- invalid rate does not increase after replay
- server logs show L3 replay in round 2, not only local L1/L2 hits

For fast debugging, `--num-questions 10` is acceptable, but it should not replace a larger final validation because the statistical signal is weaker.

### 3. L3 replay log assertions

After round 1, the server should write populated host entries to storage. Look for:

```text
storage backup queued
storage backup acked
```

After `/flush_cache` and round 2, local cache has been dropped while L3 storage remains available. Look for:

```text
storage query result ... hit_tokens=...
HiCache prefetch success ...
storage prefetch committed ...
init_load_back triggered ...
load_back committed ...
```

Correct interpretation:

- `storage query result ... hit_tokens=0` in the first round is expected after a fresh storage clear.
- `storage query result ... hit_tokens>0` in the second round proves L3 metadata/data was found.
- `storage prefetch committed` proves L3 pages were materialized into L2 host memory.
- `load_back committed` proves the prefetched L2 pages were loaded into L1 device memory.
- `cached_tokens_storage > 0` in request usage, when available, proves user-visible accounting sees L3-loaded tokens.

Failure interpretation:

- Accuracy changes but logs show L3 hits: suspect wrong host insertion, split, hash alignment, sidecar ownership, or TP sync.
- Logs show storage hits but no `load_back committed`: suspect L2 -> L1 threshold, device memory eviction, or sidecar allocation failure.
- Logs show prefetch queued but never success/revoke: suspect termination policy, prefetch timeout, or storage worker liveness.
- `host_lock_ref < 0`, `lock_ref < 0`, or pool leak warnings: suspect missing release in abort/revoke/error path.

### 4. State-machine invariants

The following invariants should hold during unit tests and real workloads:

- Full host tree backbone: any host-only node must have `Full.host_value` and a matching `hash_value`.
- Hash length: `len(node.hash_value) == len(node.key) // page_size` for storage-backed node spans.
- Prefix-key mode: when `hicache_storage_pass_prefix_keys=True`, prefix keys must describe the ancestor chain before the current node.
- Host lock pairing: every `inc_host_lock_ref()` in backup/prefetch must have exactly one `dec_host_lock_ref()` on success, abort, revoke, and timeout.
- Sidecar ownership: Mamba/SWA prefetched host slots are either adopted by a component or freed, never both and never neither.
- Split safety: `_split_node()` must split `host_value`, `hash_value`, and component host state consistently when an in-flight cached node is split.
- TP consistency: completed storage tokens are synchronized with `all_reduce(min)` before host insertion, so all ranks insert the same prefix length.
- Queue drain: `ongoing_prefetch` and `ongoing_backup` entries disappear after completion or revoke, and controller release queues do not retain stale pages indefinitely.

`split in-flight cached node` is not itself an error. It means the radix tree split a node that already had host/device transfer state. The correctness condition is that both new parent and child keep consistent Full hash/host slices and sidecar state after the split.

### 5. Multiturn hit-rate validation

Accuracy proves replay safety, but it does not prove L3 is useful. Use a multiturn workload whose working set is around 2x L2 capacity and run L3-on and L3-off serially.

L3-on server:

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python3 -m sglang.launch_server ... \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete
```

L3-off server:

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python3 -m sglang.launch_server ... \
  --enable-hierarchical-cache
```

Example benchmark:

```bash
python3 benchmark/hicache/bench_multiturn.py \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --disable-random-sample \
  --request-length 4096 \
  --output-length 1 \
  --num-clients 64 \
  --num-rounds 20 \
  --max-parallel 16 \
  --request-rate 8 \
  --ready-queue-policy random \
  --disable-auto-run \
  --enable-round-barrier \
  --seed 1 \
  --port 30000 \
  --log-file /tmp/bench_multiturn_l3.jsonl
```

Pass criteria:

- L3-on overall cache hit rate is materially higher than L2-only.
- Late-round hit rate is high and stable because repeated prefixes should have moved through L3.
- L3-on TTFT should not regress unexpectedly for hit-heavy late rounds.
- Server logs show storage prefetch/load-back activity during L3-on and no storage activity during L3-off.

Reference result from one Qwen3-Next validation run:

| Mode | Overall hit rate | Round 19 hit rate | Average TTFT | Round 19 TTFT |
|------|------------------|-------------------|--------------|---------------|
| L3 on | 0.904562 | 0.949780 | 1.208s | 2.523s |
| L2 only | 0.269435 | 0.089042 | 8.536s | 23.126s |

These exact numbers depend on model, memory budget, request shape, and hardware. The correctness signal is the large hit-rate separation when the workload exceeds L2 capacity.

### 6. Runtime API validation

Current checkout:

- `HiRadixCache` and `HiMambaRadixCache` should pass runtime attach/detach tests.
- `UnifiedRadixCache` should report runtime attach/detach as unsupported until the planned controller-managed wrapper is implemented.

After Unified runtime attach/detach is implemented, validate this sequence:

1. Start Unified with HiCache L2 enabled and no `--hicache-storage-backend`.
2. Confirm status reports `hicache_storage_backend: null`.
3. Attach `file` backend through `PUT /hicache/storage-backend` while the scheduler is idle.
4. Confirm status reports `hicache_storage_backend: file`.
5. Clear storage through `POST /hicache/storage-backend/clear`.
6. Run the two-round GSM8K replay test above and confirm L3 replay logs.
7. Detach through `DELETE /hicache/storage-backend`.
8. Confirm status reports `hicache_storage_backend: null`.
9. Re-run a small request and confirm no new storage prefetch/backup occurs.

Negative tests:

- Attach/detach while requests are running must fail with HTTP 400 and must not partially mutate scheduler state.
- Re-attaching the same backend without extra config should preserve existing prefetch knobs unless explicitly changed.
- Attach failure after write-policy mutation must restore the previous write policy.

## Remaining Work

1. Implement controller-managed runtime attach/detach for `UnifiedRadixCache`.
2. Move runtime lifecycle state into a small controller-owned state object so tree code remains a thin API wrapper.
3. Add targeted unit tests for Unified runtime attach/detach once implemented.
4. Promote `validte_l3_hicache.sh` into a tracked benchmark/validation script after cleaning naming, defaults, and output paths.
5. Extend pool assembler coverage if `FULL + SWA + MAMBA` must be supported by a single tree.
