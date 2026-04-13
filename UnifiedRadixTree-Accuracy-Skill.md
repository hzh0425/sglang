---
name: unified-radix-tree-testing
description: Guide for testing UnifiedRadixTree correctness and accuracy with HiCache L2. Covers tree assumptions A1-A8, sanity_check implementation, GSM8K precision tests, shared-prefix eviction stress tests, and multi-turn cache-hit benchmarks. Use when working on UnifiedRadixCache, HiCache, radix tree testing, tree state validation, or cache correctness verification.
---

# UnifiedRadixTree Correctness & Accuracy Testing

## Tree Assumptions (A1-A8)

These invariants govern the multi-component hierarchical radix tree (Full+Mamba+SWA, device+host layers).

### A1. Leaf Layer Consistency

Leaf nodes at each layer (device/host) must have ALL components present or ALL absent.
Internal nodes allow Mamba/SWA tombstone (independently missing), but Full never tombstones.

```
# Valid leaf:   Full=YES, Mamba=YES  (all present)
# Valid leaf:   Full=NO,  Mamba=NO   (all absent)
# Invalid leaf: Full=YES, Mamba=NO   (partial — violates A1)
```

### A2. Leaf Layer Combinations

A leaf can be device-only, host-only, device+host (post write_backup), or dead (delete it).

| Device | Host | State |
|--------|------|-------|
| all present | all absent | device-only leaf |
| all present | all present | device+host (WriteBackup done) |
| all absent | all present | host-only (evicted to host) |
| all absent | all absent | dead node -> must be deleted |

### A3. Full is Backbone

Full component is the tree skeleton. No device or host data can exist without Full also having data in that layer.

```python
# VIOLATION: any component has device data but Full.value is None
if any(dev.values()) and not dev[FULL]:
    raise "[A3] device data without Full"
# Same check for host layer
```

### A4. No Dead Nodes

Every non-root node must have device data OR host data (or both). Nodes with neither must be deleted by `_iteratively_delete_tombstone_leaf`.

### A5. Backup Continuity (Layer Contiguity)

Backed-up nodes form a continuous prefix from root (write-through mode). If a child has host data, its parent must also have host data. Same for device.

```python
if has_host and parent.host_value is None:
    raise "[A5] host gap"
if has_device and parent.device_value is None:
    raise "[A5] device gap"
```

### A6. Lock Hierarchy

- Full/SWA: path locks (node to root). Mamba: point lock (current node only).
- `full_lock_ref >= mamba_lock_ref` always.
- All locks non-negative.
- Any lock > 0 prevents eviction.

### A7. Host Leaf Set Conditions

Node enters `evictable_host_leaves` IFF:
- `node.evicted` (Full device value = None)
- `node.backuped` (any component has host_value)
- All `host_lock_ref == 0`
- No child is `evicted AND backuped`

### A8. LRU Pointer Isolation

Device LRU and host LRU use independent linked-list pointer slots.
`lru_prev/lru_next` array size = `NUM_COMPONENT_TYPES * 2`.
Device uses `[0, N)`, host uses `[N, 2N)`. A node can exist in both LRUs simultaneously.

---

## Writing sanity_check()

The `sanity_check()` method (unified_radix_cache.py) collects all violations before reporting. It runs on every scheduler idle cycle.

### Structure

```python
def sanity_check(self):
    errors: list[str] = []
    E = errors.append
    all_nodes = self._collect_all_nodes()

    # 1. Per-component checks: LRU membership, linked-list integrity, evictable size
    # 2. Per-node checks: A1-A6
    # 3. Set checks: A7 (evictable_host_leaves), evictable_device_leaves

    if errors:
        self.pretty_print()
        raise AssertionError(f"{len(errors)} violations:\n" + "\n".join(errors))
```

### Check Categories

**Per-component (for each ComponentType)**:
1. `evictable_size >= 0`, `protected_size >= 0`
2. Device LRU membership: tree nodes with `cd.value != None` must match `lru.cache.keys()` exactly
3. Device LRU linked-list walk: verify `prev` pointers, no cycles, list length == cache size
4. Recompute evictable_size by walking LRU, compare with tracked value
5. Host LRU integrity (non-FULL components); Full host_lru must be empty

**Per-node (skip root)**:
1. **A1**: if device-leaf, all components must have device value. Same for host-leaf.
2. **A3**: if any component has data, Full must also have data (per layer)
3. **A4**: node must have device or host data (not neither)
4. **A5**: if child has data, parent must also (per layer)
5. **A6**: locks non-negative, `full_lock >= aux_lock`

**Set checks**:
1. **A7 forward**: every node in `evictable_host_leaves` must satisfy the 4 conditions
2. **A7 reverse**: every qualifying node must be IN the set
3. **evictable_device_leaves**: similar forward+reverse membership check

### Key Helper

```python
def _check_lru_linked_list(self, lru, ct, label, errors):
    """Walk forward through LRU, verify prev pointers, membership, no cycles."""
    visited = set()
    x = lru.head.lru_next[pt]
    prev = lru.head
    while x is not None and x != lru.tail:
        if x.lru_prev[pt] != prev: errors.append(f"broken prev")
        if x.id not in lru.cache: errors.append(f"not in cache")
        if x.id in visited: errors.append(f"cycle"); break
        visited.add(x.id)
        prev = x; x = x.lru_next[pt]
    if len(visited) != len(lru.cache):
        errors.append(f"list size != cache size")
```

---

## Testing Strategy

### 1. GSM8K Precision Test (L2 Correctness)

Validates that L2 backup/restore doesn't corrupt model outputs.

```bash
# Round 1: baseline accuracy
python benchmark/gsm8k/bench_sglang.py \
    --port 30000 --num-questions 100 --num-shots 24 --parallel 10

# Flush L1 (device), keep L2 (host)
curl localhost:30000/flush_cache

# Round 2: verify accuracy after L2->L1 restore
python benchmark/gsm8k/bench_sglang.py \
    --port 30000 --num-questions 100 --num-shots 24 --parallel 10
```

**What to check**:
- Both rounds accuracy >= 0.85 (within normal variance)
- Server log shows `init_load_back triggered` in round 2
- No `Sanity check FAILED` in server log

### 2. Shared Prefix Bench (Eviction Stress)

Generates massive eviction pressure to test tree state under churn.

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 128 --gsp-prompts-per-group 16 \
    --gsp-system-prompt-len 5000 --gsp-question-len 5000 \
    --gsp-output-len 1 --num-prompts 2048 \
    --request-rate inf --base-url http://127.0.0.1:30000
```

**Memory pressure**: 128 groups x 5000 prefix = 640K tokens >> device pool. Forces heavy device eviction (write_through -> evict_to_host) and host eviction.

**What to check**:
- All requests succeed (no server crash)
- No sanity_check failures in server log
- Compare with no-L2 baseline for hit rate improvement

### 3. Multi-turn Bench (Eviction + Restore Interleaving)

Growing context across rounds triggers eviction/restore cycles.

```bash
python3 benchmark/hicache/bench_multiturn.py \
    --model-path <MODEL> --disable-random-sample \
    --request-length 2048 --output-length 1024 \
    --num-clients 20 --num-rounds 8 --max-parallel 4 \
    --request-rate 4 --ready-queue-policy random \
    --disable-auto-run --enable-round-barrier --port 30000
```

**Tuning for eviction pressure**:
- `num_clients * num_rounds * (request_length + output_length)` should exceed `device_pool + host_pool`
- Too little eviction = no bugs found. Too much = hit rate near zero.
- Target: total tokens ~2-4x of combined pool size.

**What to check**:
- All rounds complete without crash
- Cache hit rate > 0 (L2 is providing value)
- Compare with no-L2 baseline (expect 2-5x hit rate improvement)

### 4. flush_cache + Replay (L2 Restore at Scale)

```bash
curl localhost:30000/flush_cache
# Then re-run shared prefix or multi-turn with fewer requests
```

Tests large-scale L2->L1 restore after L1 wipe.

---

## Checking Results

### Server Log Patterns

```bash
# Count load_back operations
grep -c "load_back" sglang.out

# Check for sanity violations (0 = good)
grep -c "Sanity check FAILED" sglang.out

# Check for crashes
grep "Scheduler hit an exception" sglang.out

# Extract cache hit stats
grep -oP "#cached-token: \K\d+" sglang.out | awk '{s+=$1;n++} END{print s/n}'
```

### Common Failure Patterns

| Symptom | Likely Cause | Where to Look |
|---------|-------------|---------------|
| `[A1] device-leaf partial` | Component not atomically evicted/restored | `evict_component`, `commit_hicache_transfer` |
| `[A3] device data but Full=None` | Full evicted without aux components | `_evict_to_host`, `_cascade_evict` |
| `[A4] no device and no host` | Tombstone deletion missed | `_iteratively_delete_tombstone_leaf` |
| `[A5] host but parent no host` | Backup continuity broken | `write_backup` backup-invariant check |
| `[A7] qualifies but missing` | Leaf set not updated | `_update_evictable_leaf_sets` call sites |
| `[DevLeaf] qualifies but missing` | New node/split didn't update set | `_add_new_node`, `_split_node` |
| `[LRU] device mismatch` | LRU insert/remove out of sync | `_evict_to_host`, `commit_hicache_transfer` |
| OOM with small eviction heap | Nodes not registered as evictable | `_update_device_leaf_status` in `_add_new_node`/`_split_node` |

### No-L2 Baseline Comparison

Always run the same benchmark without `--enable-hierarchical-cache` to establish:
1. Baseline hit rate (should be lower)
2. Baseline throughput (should be similar)
3. That any issues are L2-specific, not general

---

## Known Bug Patterns (Reference)

Bugs found during Phase 1 testing that illustrate common pitfalls:

1. **Missing import**: `PoolName` used in `load_back` but only imported locally in `_init_hicache`
2. **Wrong pool in controller**: KV PoolTransfer passed as `extra_pools` to controller which tries `device_pool.alloc()` on non-allocatable `MHATokenToKVPool`
3. **Walk past root**: `finalize_match_result` walking `node.parent` chain without `None` guard
4. **Insert vs match inconsistency**: `_insert_helper` counting evicted nodes in `prefix_length` while `match_prefix` skips them
5. **Leaf set not updated**: `_add_new_node` / `_split_node` missing `_update_evictable_leaf_sets` calls -> ghost nodes fill device pool -> OOM
6. **Mamba evicts locked Full**: Mamba `drive_eviction` calling `_evict_device_leaf` on nodes with `Full.lock_ref > 0`
7. **Cascade returns wrong count**: `_evict_device_leaf` cascade path returning `sum(all_tracker)` instead of `tracker[FULL]` only
