# UnifiedTree External Cache Connector Design

Date: 2026-07-20

## Summary

Add a direct external-cache connector to `UnifiedRadixCache` without modeling the external system as HiCache L2 host memory. The connector moves Full KV and hybrid component state directly between device pools and an external cache such as Mooncake. Existing HiCache keeps its current device/host/storage path.

The two paths share the existing transfer vocabulary:

- `PoolTransfer`
- `PoolTransferResult`
- `PoolHitPolicy`
- `SidecarPoolSpec`
- component prepare/commit/abort hooks

`UnifiedRadixCache` owns tree mutations, component locks, operation lifetime, and TP/CP agreement. A connector performs only local-rank lookup and I/O. It does not receive process groups or tree nodes.

Version 1 supports Full KV, SWA, Mamba, and the existing index-derived sidecars. Direct connectors and HiCache are mutually exclusive. Pipeline parallelism, asynchronous direct load, independently indexed sidecars, and runtime connector attach/detach are out of scope.

## Goals

1. Define one backend-neutral connector interface similar to the direct Mooncake radix-cache implementation.
2. Support UnifiedTree hybrid state: Full KV, SWA, Mamba, and sidecar pools.
3. Preserve the current HiCache controller, storage backend, component, and transfer contracts where possible.
4. Put TP/CP hit agreement, completion ordering, and cache publication in the tree layer.
5. Keep external-cache concepts out of `UnifiedTreeNode`; the local tree represents device-resident state only on the direct path.
6. Preserve normal UnifiedTree insert, split, LRU, lock, and eviction invariants after an external load.

## Non-goals

- Running HiCache and a direct external connector at the same time.
- Treating Mooncake DRAM and SSD as separate SGLang tiers.
- Pipeline-parallel direct external caching.
- Layer-wise or asynchronous external-to-device load overlap.
- Sidecars without `indices_from_pool`; version 1 sidecars must use the existing index-derived model.
- Runtime connector attach/detach.
- A generic graph of arbitrary storage tiers.
- Exposing Mooncake replica placement to scheduling decisions.

## Terminology

- **Local/L1**: device pools owned by SGLang.
- **External cache**: a system such as Mooncake that owns its internal memory/SSD placement.
- **Anchor pool**: Full KV, represented by `PoolName.KV`. Its page-prefix length defines the logical cache prefix.
- **Component pool**: state owned by a UnifiedTree component, such as SWA or Mamba.
- **Sidecar pool**: additional state registered through `SidecarPoolSpec`, possibly deriving indices from an anchor pool.
- **Logical key**: backend-independent page hash emitted by the tree/component layer.
- **Shard identity**: TP/CP rank coordinates used by a connector to isolate physical object keys.

## Architecture

```text
Scheduler
    |
    v
UnifiedRadixCache
    - radix tree and node locks
    - normal insert/split/eviction
    - transfer orchestration
    - TP/CP agreement
    |
    +--> TreeComponent: Full / SWA / Mamba
    |        |
    |        +--> PoolTransfer descriptors
    |
    +--> SidecarPoolSpec expansion
    |
    +--> existing HybridCacheController
    |        L1 <-> host <-> storage
    |
    `--> new CacheConnector
             L1 <-> external cache
```

The direct path does not create host pools, set `ComponentData.host_value`, maintain host LRU lists, or insert host-only nodes. Mooncake DRAM and SSD remain implementation details of Mooncake.

## Design Decisions

### Keep the existing HiCache transport separate

`HybridCacheController` is not converted into a connector in version 1. It owns host allocation, D2H/H2D streams, host release queues, storage queues, and layer-ready events. A direct connector has none of those responsibilities.

Both transports consume the same component-generated `PoolTransfer` descriptors. This delivers compatibility without forcing fake host indices into the direct path.

### Full KV remains the anchor pool

The current HiCache storage path treats Full KV as the prefix anchor and extra pools as constraints on that prefix. Version 1 keeps this model:

- The tree builds the `PoolName.KV` transfer.
- Non-Full components add their transfers.
- Registered sidecars are expanded from those transfers.
- The usable prefix is the Full KV prefix accepted by all required pools.

This avoids a broad rewrite of `batch_exists_v2()` semantics.

### Lookup and load are synchronous; store is asynchronous

This matches the direct Mooncake branch and the scheduler's current two-phase match/load contract:

- `lookup()` returns before `match_prefix()` completes.
- `load()` returns before the loaded prefix is published in the tree.
- `store_async()` returns an operation handle.
- `poll()` reports completed stores.

Asynchronous or layer-wise direct load can be added later without changing component descriptors, but it is not part of this design.

## Shared Transfer Types

The existing structures remain the common wire format:

```python
@dataclass
class PoolTransfer:
    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[list[str]] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    required_tail_pages: int = 1
    nodes_to_load: Optional[list[Any]] = None
    indices_from_pool: Optional[PoolName] = None


@dataclass
class PoolTransferResult:
    kv_hit_pages: int
    extra_pool_hit_pages: dict[str, int]
```

Direct connectors use `device_indices` and `keys`; they never populate `host_indices`.

For direct lookup over `N` suffix pages, every transfer supplies `N` logical keys aligned with those boundaries. `ALL_PAGES` makes boundary `i` depend on keys `[0:i]`. `TRAILING_PAGES` makes it depend on keys `[max(0, i - required_tail_pages):i]`: SWA sets its window width and Mamba uses one exact checkpoint key per boundary with width one. Boundary zero requires no key. Existing HiCache callers retain their current behavior because the added field has a compatible default.

For load and store, a transfer additionally contains destination or source `device_indices`; the tree has already selected one exact boundary.

## Pool I/O Adapter

The Mooncake branch currently resolves MHA/MLA KV slot indices to raw multi-buffer pointers inside `MooncakeConnector`. Hybrid state requires the same operation for SWA, Mamba, and sidecar pools. Physical pool layout is therefore isolated behind a pool adapter.

```python
@dataclass(frozen=True)
class BufferSpan:
    address: int
    size: int


class PoolIOAdapter(Protocol):
    @property
    def format_id(self) -> str:
        """Stable dtype/layout/version fingerprint for physical key isolation."""

    def registerable_buffers(self) -> list[BufferSpan]:
        """Return complete allocations that the connector must register."""

    def resolve_objects(
        self,
        indices: torch.Tensor,
        object_count: int,
    ) -> list[list[BufferSpan]]:
        """Return one multi-buffer span list for each logical object key."""
```

"Object" is deliberately broader than KV page: it is a KV page for Full/SWA and a checkpoint entry for Mamba. `resolve_objects()` must return exactly `object_count` entries, one for each key in the transfer.

Each connector maintains `dict[PoolName, PoolIOAdapter]`. Initialization fails if any transfer target has no adapter.

For a sidecar transfer, `indices_from_pool` selects the **source indices only**. The connector still resolves those indices through the target sidecar's own adapter:

```text
transfer.name = DEEPSEEK_V4_C4
transfer.indices_from_pool = KV

indices  <- KV transfer.device_indices
buffers  <- adapters[DEEPSEEK_V4_C4].resolve_objects(indices, len(keys))
```

Using the KV adapter would access the wrong physical buffers.

Initial adapters are:

- MHA/MLA Full KV adapter, extracted from Mooncake's current GPU-object metadata helper.
- SWA pool adapter.
- Mamba state-pool adapter.
- Target-pool adapters supplied by the hybrid pool assembler for existing index-derived sidecars.

## Connector Interface

```python
@dataclass(frozen=True)
class CacheShard:
    tp_rank: int
    tp_size: int
    cp_rank: int
    cp_size: int


@dataclass(frozen=True)
class CacheKeyContext:
    extra_key_digest: str
    shard: CacheShard


@dataclass
class ConnectorLoadResult:
    successes: dict[PoolName, list[bool]]


@dataclass
class ConnectorLookupResult:
    pool_valid_boundaries: dict[PoolName, list[bool]]
    pool_result: PoolTransferResult


@dataclass
class ConnectorCompletion:
    handle: object
    success: bool


class CacheConnector(Protocol):
    def register_pool(self, name: PoolName, adapter: PoolIOAdapter) -> None:
        ...

    def lookup(
        self,
        transfers: list[PoolTransfer],
        key_context: CacheKeyContext,
    ) -> ConnectorLookupResult:
        """Return exact-boundary availability for the external bundle."""

    def load(
        self,
        transfers: list[PoolTransfer],
        key_context: CacheKeyContext,
    ) -> ConnectorLoadResult:
        """Load external objects directly into device-pool destinations."""

    def store_async(
        self,
        transfers: list[PoolTransfer],
        key_context: CacheKeyContext,
    ) -> object | None:
        """Queue a store and return its handle, or None when rejected."""

    def poll(self, wait: bool = False) -> list[ConnectorCompletion]:
        """Return completed accepted stores; wait=True waits for all local handles."""

    def close(self) -> None:
        """Drain work, unregister buffers, and release backend resources."""
```

The connector must not accept `Req`, `UnifiedTreeNode`, lock parameters, or distributed process groups. Store handles are connector operation identities, not request IDs.

For a lookup covering absolute page boundaries `[start_pages, candidate_pages]`, `pool_valid_boundaries[pool][i]` states whether that external pool can materialize its part of exact boundary `start_pages + i`. Every mask length must be `candidate_pages - start_pages + 1`, and element zero is true because the already-common base requires no external objects. `pool_result` preserves the existing per-pool summary for diagnostics and HiCache-compatible policy code; distributed correctness uses the per-pool exact-boundary masks.

The connector validates that every lookup transfer supplies the expected aligned key count and computes one presence bit per pool and boundary from its policy. It does not combine pools: the tree must be able to satisfy Full locally while loading Mamba, SWA, or a sidecar externally. Misaligned key counts or invalid tail widths are invariant errors, not partial hits.

Lookup returns no backend-private load handle. After TP/CP agreement, the tree builds fresh load descriptors for only the agreed logical range and `load()` operates on those descriptors. This prevents rank-local lookup state for a different boundary from leaking into the common load.

### Connector key scoping

Components and the tree emit logical object hashes in `PoolTransfer.keys`. The direct path computes Full page hashes from `RadixKey` with a shared helper even when HiCache storage is disabled; it does not depend on `UnifiedTreeNode.hash_value` being populated.

The tree computes `extra_key_digest` with the same stable serialization rule for LoRA/cache salts on every backend. The connector converts each logical key to a physical key equivalent to:

```text
<connector namespace>/<extra_key_digest>/tp<TP_RANK>-of-<TP_SIZE>/
cp<CP_RANK>-of-<CP_SIZE>/<pool name>/<adapter format_id>/<logical key>
```

The connector namespace contains at least:

- model identity and revision
- KV/state dtype and layout identity
- page size
- connector key-format version

`CacheKeyContext` supplies the request namespace and rank coordinates that cannot be recovered from `PoolTransfer`. TP and CP coordinates are required because those ranks can own different slices for the same token page. PP is excluded from version 1.

## Component Interface

Components use one neutral transfer facade with typed request objects. External lookup does not require a target node; it receives the existing local anchor and the logical key range explicitly.

```python
class CacheTransferPhase(str, Enum):
    BACKUP_HOST = "backup_host"
    LOAD_BACK = "load_back"
    BACKUP_STORAGE = "backup_storage"
    PREFETCH = "prefetch"
    LOOKUP_EXTERNAL = "lookup_external"
    LOAD_EXTERNAL = "load_external"
    STORE_EXTERNAL = "store_external"


@dataclass(frozen=True)
class ExternalLookupRequest:
    anchor_node: UnifiedTreeNode
    req: Req
    key: RadixKey
    full_page_keys: list[str]
    start_pages: int
    full_device_pages: int
    candidate_pages: int


@dataclass(frozen=True)
class ExternalLoadRequest:
    anchor_node: UnifiedTreeNode
    req: Req
    key: RadixKey
    agreed_full_page_keys: list[str]
    common_local_pages: int
    full_device_pages: int
    agreed_pages: int


@dataclass(frozen=True)
class ExternalStoreRequest:
    node: UnifiedTreeNode
    key: RadixKey
    full_page_keys: list[str]


CacheTransferRequest = Union[
    ExistingHiCacheTransferRequest,
    ExternalLookupRequest,
    ExternalLoadRequest,
    ExternalStoreRequest,
]


class TreeComponent:
    def local_boundary_validity(
        self,
        request: ExternalLookupRequest,
    ) -> dict[PoolName, list[bool]]:
        """Return per-pool exact local validity for each requested boundary."""

    def build_cache_transfers(
        self,
        request: CacheTransferRequest,
    ) -> Optional[list[PoolTransfer]]:
        ...

    def commit_cache_transfer(
        self,
        request: CacheTransferRequest,
        transfers: list[PoolTransfer],
        load_result: Optional[ConnectorLoadResult] = None,
        insert_result: Optional[InsertResult] = None,
    ) -> None:
        ...

    def abort_cache_transfer(
        self,
        request: CacheTransferRequest,
        transfers: list[PoolTransfer],
    ) -> None:
        ...
```

All `*_pages` fields in the request dataclasses are absolute prefix boundaries measured from the root. `ExternalLookupRequest.full_page_keys` covers `[start_pages, candidate_pages)`. `ExternalLoadRequest.agreed_full_page_keys` covers `[common_local_pages, agreed_pages)`. `local_boundary_validity()` returns one boolean per pool and inclusive boundary; Full is monotonic, while SWA/Mamba may mark sparse exact boundaries. The tree derives index-sidecar masks and retains pool identity so local and external state can be combined independently.

`ExistingHiCacheTransferRequest` is an adapter around the arguments already accepted by `build_hicache_transfers()` and `commit_hicache_transfer()`; it does not change HiCache semantics.

Compatibility works in this direction:

```text
Unified tree HiCache call site
    -> build_cache_transfers(ExistingHiCacheTransferRequest)
    -> default facade delegates to existing component build_hicache_transfers()

Unified tree direct call site
    -> build_cache_transfers(External*Request)
    -> component handles the new typed request
```

The same adapter rule applies to commit. Existing component implementations can migrate one at a time without changing the current HiCache path.

For `ExternalLoadRequest`, allocation ownership is explicit:

- The tree owns staged Full KV destinations.
- SWA and Mamba components own destinations they allocate in `build_cache_transfers()`.
- Existing sidecars derive indices from an owning source pool and allocate no independent slots.
- Every returned staged component transfer receives exactly one commit or abort call.
- Commit attaches only the agreed state to the final node and releases unused tails.
- Abort releases all staged component destinations and attaches nothing.

Direct restore adds an `EXTERNAL_STAGED` component mode to `InsertParams`. It is a structural-insertion mode, not a second tree insertion implementation:

- Normal radix traversal, split, overlap handling, Full KV insertion, and node creation still run.
- Component split hooks still run because existing component data must follow topology changes.
- SWA/Mamba request-derived recovery and finalization hooks do not allocate, free, or attach state for the restored span; new component data starts as a tombstone.
- After insertion identifies `InsertResult.last_device_node`, `commit_cache_transfer()` attaches the already staged state across the affected path.
- The tree pre-validates every staged transfer before structural insertion. Commit performs no allocation or I/O and is treated as an invariant-preserving, non-failing operation.
- Structural insertion plus all component commits execute synchronously on the scheduler thread. The match result and cache events are published only after all commits and evictability updates complete.

This gives the restore one publication boundary while retaining normal UnifiedTree split and insertion mechanics. An unexpected commit invariant failure is fatal; it is not converted into a partially published cache hit.

### External component behavior

| State | Lookup policy | Load behavior | Store behavior |
|---|---|---|---|
| Full KV | Anchor, all pages | Tree allocates Full device slots | Tree gathers root-to-node Full slots |
| SWA | Trailing pages | Component allocates/restores SWA slots and mapping | Component exposes the valid trailing window |
| Mamba | Trailing pages | Component allocates one applicable checkpoint slot | Component exposes the checkpoint for the stored suffix |
| Index-derived sidecar | Follows its source pool | Target adapter writes using source-pool indices | Target adapter reads using source-pool indices |

Mamba and SWA staged state is attached after structural insertion returns the final target node but before the restore is published. `InsertResult.last_device_node` must therefore be populated by UnifiedTree insertion.

### Lookup and load result semantics

Connector lookup evaluates every exact boundary in the shared root-relative search interval. This is required because Full KV availability is monotonic but a longer Mamba checkpoint or SWA trailing window does not imply that a shorter boundary has the corresponding state.

| Operation | Policy/pool | Result interpretation |
|---|---|---|
| Lookup | Full KV anchor | Boundary `b` is externally valid only if every requested Full object through `b` exists. |
| Lookup | `ALL_PAGES` | Boundary `b` is valid only if all component objects required from the search base through `b` exist. |
| Lookup | `TRAILING_PAGES` | Boundary `b` is checked against the exact trailing window or checkpoint keys declared for `b`; validity is not inferred from another boundary. |
| Load | Every pool and policy | Transactional in version 1: every requested object must load successfully on every TP/CP rank. Any failure aborts the entire external load and publishes no new prefix. |
| Store | Every pool and policy | Partial backend failure is allowed, but the operation reports failure. A later lookup recomputes the usable common prefix. |

`ConnectorLookupResult.pool_valid_boundaries` is combined with local per-pool masks for exact-boundary agreement. Its `PoolTransferResult` summary remains lookup-only and is not reduced as a scalar hit length. `ConnectorLoadResult.successes` is used only for load and must contain one boolean per requested object. Version 1 deliberately does not partially commit a shorter Full prefix after load failure because SWA/Mamba trailing state is tied to the originally agreed prefix boundary.

After TP/CP lookup agreement, components build fresh load transfers from `ExternalLoadRequest.agreed_full_page_keys`. Rank-local descriptors or backend state for a longer lookup result are never reused.

## Tree State

The direct path adds only operation state to `UnifiedRadixCache`:

```python
pending_external_matches: dict[str, PendingExternalMatch]
ongoing_external_stores: OrderedDict[int, OngoingExternalStore]
external_store_sequence: int
```

`PendingExternalMatch` contains a snapshot of the matched logical key, key context, the cross-rank common exact-local boundary, the rank-local Full-device frontier, the rank-local per-pool validity bits at the agreed boundary, and the TP/CP-agreed absolute boundary. Fresh load requests are derived from this immutable range; connector-private lookup state and rank-local descriptors are not retained. When the agreed boundary exceeds the common local boundary, every rank creates the pending entry, including ranks that already hold all exact state for the agreed boundary locally.

`OngoingExternalStore` is keyed by the tree sequence number and contains an optional local connector handle, local completion/success state, the final tree node, and the exact `DecLockRefParams` returned when that node was protected. A rejected local submission therefore still occupies its sequence position.

No external residency bit is added to a node. Future lookups remain authoritative, which also tolerates partial store failure across ranks.

## Read Path

### Match

1. Run normal UnifiedTree local matching.
2. Ask every component for rank-local per-pool exact-boundary validity masks over the page-aligned candidate range and derive index-sidecar masks. AND the required pool masks to obtain the rank-local complete-bundle mask.
3. Pack that mask and apply TP/CP `MIN` elementwise. The highest true boundary is `common_local_pages`; root boundary zero is always true.
4. If `common_local_pages` equals the candidate boundary, return the common local hit. Every rank has entered the same local-mask collective, so the decision cannot diverge.
5. Build Full and component `LOOKUP_EXTERNAL` descriptors for `[common_local_pages, candidate_pages]`, expand sidecars, and call local connector `lookup()` on every rank.
6. For every pool and boundary, compute `pool_reachable = local_pool_valid OR external_pool_valid`, then AND all required pool results into the rank-local reachable-boundary mask. This is the L1-Full plus external-Mamba/SWA/sidecar composition point.
7. Pack the reachability mask and apply TP/CP `MIN` elementwise. Select the highest true exact boundary as `agreed_pages`; this is an intersection of valid boundaries, not a minimum of scalar hit lengths.
8. If `agreed_pages > common_local_pages`, every rank saves the agreed range and its per-pool local validity at that boundary under `req.rid`. Fresh component load descriptors are built only in the load phase for pools whose local bit is false.
9. Return device indices through `common_local_pages` plus the common `external_hit_length = (agreed_pages - common_local_pages) * page_size`.

`MatchResult`, `Req`, and load parameters gain `external_hit_length`. Because both the base and target are cross-rank exact boundaries, `external_hit_length > 0` is the same on every rank and safely drives `Req.needs_external_load()`. A rank that already has exact state at `agreed_pages` still enters `init_external_load()` but performs no local connector I/O. The direct path does not overload `host_hit_length`, `last_host_node`, or host-only semantics.

### Load and publication

1. Consume `pending_external_matches[req.rid]`.
2. Start with `local_load_ok = 1`. If every required pool is already exact-local at the agreed boundary on this rank, stage no resources and skip to agreement. Otherwise allocate Full device slots only when Full's local bit is false and only beyond the existing Full-device frontier; an aux-only load therefore has no Full destination transfer.
3. Ask components to build fresh `LOAD_EXTERNAL` transfers only for state missing at the agreed exact boundary and stage their device slots, then expand sidecars.
4. If any local allocation/build step fails, set `local_load_ok = 0`, keep all successfully staged resources for abort, and skip local connector I/O. Do not return.
5. Otherwise, call connector `load()` only when this rank has missing objects. A rank with no local I/O keeps `local_load_ok = 1`.
6. Validate result cardinality, object success, and staged attachability. Any exception or malformed result sets `local_load_ok = 0`.
7. Every rank reduces `local_load_ok` with `MIN` across TP/CP, regardless of whether it allocated or called the connector.
8. If the agreed result is zero, call abort for every locally staged component transfer, free all newly allocated Full slots, restore the scheduler-visible prefix to `common_local_pages`, and publish no new cache state.
9. Call normal UnifiedTree insertion/no-op insertion in `EXTERNAL_STAGED` mode to ensure the agreed target boundary exists. Set `prev_prefix_len` to `min(full_device_pages, agreed_pages) * page_size`, preserve the corresponding existing Full indices, and provide newly loaded Full slots only for the missing suffix.
10. Populate `InsertResult.last_device_node`, commit staged component and sidecar state in deterministic component order, refresh evictability, and publish the agreed boundary.
11. Return the Full indices that became newly usable to the scheduler. These can include already-resident Full indices exposed by an aux-only SWA/Mamba load as well as newly loaded Full indices.

No path manually constructs or grafts a tree node.

## Write Path

1. Complete normal `cache_finished_req()` preparation and UnifiedTree insertion.
2. Obtain the final target node from `InsertResult.last_device_node`.
3. Acquire component locks for that node and save the returned release parameters.
4. Build the Full root-to-node `STORE_EXTERNAL` transfer.
5. Ask non-Full components for store transfers and expand sidecars.
6. Allocate the same tree-local sequence number on every rank and submit one connector store containing all pool descriptors.
7. Insert an `OngoingExternalStore` entry on every rank. An accepted rank stores its connector handle; a rejected rank stores `handle=None`, `local_complete=True`, and `local_success=False`.
8. Poll accepted handles from the scheduler thread and update their sequence entries.
9. Release locks only when that sequence lies in the TP/CP-agreed consecutive completed operation prefix.

The external store does not need backend-level atomicity across pools. A later lookup applies all required hit policies and truncates the usable prefix if any required state is missing.

Normal eviction never waits for all stores. Locked nodes are already ineligible for eviction. Rejected placeholders remain locked only until the common completion-prefix poll can drain their sequence. Only reset, flush, and shutdown synchronously drain the common store sequence, waiting for any accepted local handles on the way.

## TP/CP Ownership and Agreement

Distributed coordination belongs to `UnifiedRadixCache`, not the connector.

### Lookup agreement

The tree performs elementwise `MIN` on two deterministic boolean boundary masks through the existing attention CP and attention TP groups:

1. The all-component local-validity masks are intersected to select the highest boundary already exact-local on every rank.
2. Above that base, each rank ORs local and external validity separately for every required pool, then ANDs the pools. Those rank-reachable masks are intersected, and the highest common true boundary becomes the external target.

This handles non-monotonic SWA windows and Mamba checkpoints: validity at boundary 100 does not imply validity at boundary 90. Scalar `MIN(hit_pages)` is therefore forbidden on the direct hybrid path. Only the two combined rank masks cross the network, encoded as page-aligned `uint8` vectors with identical root-relative ordering and length; per-pool composition stays local. `PoolTransferResult` remains diagnostic; pool names are sorted if per-pool metrics are packed.

### Load agreement

Each rank validates one boolean per requested object and staged attachability. A rank with no missing local objects starts with `load_ok = 1`; a rank with allocation/build failure sets it to zero and skips connector I/O. Every rank still enters the `MIN` reduction across TP/CP. All ranks commit the complete agreed prefix when the result is one; all ranks abort every locally staged destination when it is zero. Version 1 has no partial direct-load publication.

### Store completion agreement

Stores receive a tree-local monotonically increasing sequence number in submission order. At each event poll:

1. Each rank counts the consecutive completed operations at the head of its queue.
2. The tree reduces that count with `MIN` across TP/CP.
3. Every rank drains exactly that many operations in the same order.
4. Completion success is reduced for metrics, but locks are released after completion even when storage failed.

A rank-local submission rejection is an immediately completed failed placeholder, not an immediate dequeue. This preserves identical sequence positions when another rank accepted the same operation and is still performing I/O.

Future lookup agreement prevents a partially stored prefix from becoming a cache hit.

The connector never calls `all_reduce()` and never receives a process group. Mooncake's current `_tp_min()` logic moves to the tree.

## Scheduler and Base Cache Compatibility

The scheduler-facing changes are generic:

- Add `external_hit_length` alongside existing host/component hit lengths.
- Add `Req.needs_external_load()` based on the now cross-rank-common `external_hit_length`.
- Add `UnifiedRadixCache.init_external_load()`; keep `init_load_back()` for HiCache.
- Add `check_cache_events()` as the neutral polling entry.
- Keep `check_hicache_events()` as a compatibility alias.
- Add `requires_event_polling` on `BasePrefixCache`; the scheduler no longer checks Mooncake/FlexKV-specific flags.

The scheduler dispatches external load before constructing the prefill batch. Existing HiCache load-back behavior is unchanged.

## Initialization and Configuration

Add a connector registry separate from the radix-cache registry:

```python
register_cache_connector(name, factory)
get_cache_connector_factory(name)
```

Use a neutral server option:

```text
--cache-connector mooncake
```

When configured:

- Tree-cache construction selects `UnifiedRadixCache`.
- Normal model inspection selects Full, SWA, and Mamba components.
- The existing hybrid assembler registers component/sidecar pool adapters.
- The connector registers all device allocations before serving.
- Startup rejects HiCache, PP greater than one, missing pool adapters, unsupported layouts, and disabled radix cache.

If the direct Mooncake branch's `--radix-cache-backend mooncake` option must remain compatible, it is translated to `--cache-connector mooncake` with a deprecation warning. It must not construct `MooncakeRadixCache` once the connector path is available.

## Error and Lifetime Rules

### Lookup failure

Treat connector exceptions as an all-false external-boundary mask and still enter TP/CP agreement. Misaligned component keys, malformed mask lengths, configuration, and layout are invariant failures and remain fatal.

### Allocation failure

Set local `load_ok=0`, preserve successfully staged resources for rollback, and still enter TP/CP load agreement. After the common result is zero, abort all staged component transfers, free Full destinations, remove the pending match, and continue from the cross-rank common local prefix as an ordinary cache miss.

### Load failure or partial load

Set local `load_ok=0`, reduce it across TP/CP on every rank, call every component abort hook, free all newly allocated Full destinations, and publish no new prefix. A rank must not return early because its peer may be waiting in the collective. Partial load publication is deferred beyond version 1.

### Store rejection

Record an immediately completed failed placeholder for the operation's sequence. Keep the node locked until the common completion-prefix poll drains that sequence, then release it. Serving continues without external persistence.

### Store failure after acceptance

Keep the node locked until completion, record a failure metric, then release it. Do not delete local tree state or retry in version 1.

### Request abort

Remove only the pending lookup/load marker. Accepted stores are independent of request IDs and remain protected until connector completion.

### Reset and flush

Drain the common store sequence, waiting for accepted local handles and consuming rejected placeholders in order. Then release its locks, clear pending matches, and reset the local tree. External objects remain intact.

### Shutdown

Drain stores, close the connector, unregister device allocations, then release device pools.

## Observability

Add backend- and pool-labeled counters for:

- lookup requests, hit pages, and misses
- load requested/committed pages
- partial loads and load failures
- accepted/rejected/completed/failed stores
- pending store count
- TP/CP lookup or load disagreement

Replica placement such as Mooncake memory versus SSD is connector-specific diagnostic information and is not part of Tree scheduling.

## Testing

### Connector unit tests

- Stable key namespace varies with model, `extra_key`, TP rank, and CP rank.
- Every device allocation is registered once.
- MHA/MLA, SWA, Mamba, and sidecar adapters return one object-major span list per logical key.
- Lookup honors `ALL_PAGES` and `TRAILING_PAGES`.
- Load/store use matching keys and buffer shapes.
- Store rejection and completion are reported correctly.

### UnifiedTree unit tests

- A full local hit on every rank bypasses connector I/O after the common local-validity collective.
- External lookup reports `external_hit_length` without host state.
- External load uses normal insertion and preserves split/LRU/size invariants.
- Full plus Mamba, Full plus SWA, and sidecar loads commit only when all required state exists.
- Aux-only load exposes already resident Full indices after restoring missing SWA/Mamba state.
- Per-pool masks allow local Full plus external Mamba/SWA even when Full is absent externally.
- `EXTERNAL_STAGED` insertion leaves auxiliary tombstones until deterministic component commit and exposes no intermediate match result.
- Partial load frees every uncommitted pool slot.
- Store locks the exact final node and unlocks once.
- Request abort never unlocks an accepted store.
- Eviction skips store-locked nodes without globally draining stores.
- Reset drains stores while preserving external objects.

### Distributed tests

- TP ranks with different and non-monotonic local/external boundary masks publish their highest common exact boundary.
- A rank already exact-local at the agreed boundary skips load I/O but still joins load agreement required by another rank.
- CP ranks use distinct physical keys and intersect exact-boundary masks.
- One-rank allocation failure makes every rank abort without a collective hang.
- Any TP/CP load failure aborts publication of the entire previously agreed prefix.
- Store completion queues drain in identical order across ranks.

### Regression tests

- Existing UnifiedTree Full/SWA/Mamba tests pass without a connector.
- Existing Unified HiCache host/storage tests pass unchanged.
- Existing scheduler abort, timeout, and priority-preemption tests pass through generic event cleanup.

### End-to-end tests

- Mooncake direct round trip for MHA TP2.
- Mooncake direct round trip for MLA TP2.
- Hybrid Mamba state round trip.
- Hybrid SWA state round trip.
- At least one registered sidecar round trip.
- CP round trip on a model/configuration with CP-aware cache slices.
- Flush local tree, reload from Mooncake, and verify cached-token count and output correctness.

## Rollout Sequence

1. Introduce neutral component hook names and compatibility wrappers; keep all behavior unchanged.
2. Add connector registry, shared direct connector types, pool adapters, and generic scheduler polling.
3. Integrate Full-only direct Mooncake through UnifiedTree and delete the separate tree lifecycle from `MooncakeRadixCache`.
4. Add SWA and Mamba external phases with allocation rollback tests.
5. Register and validate sidecar adapters.
6. Enable TP/CP agreement and distributed tests.
7. Enable `--cache-connector mooncake` for the complete supported component matrix.

Each step must keep the existing HiCache path green. The final option is not enabled for a model until every required component and sidecar pool has an adapter and test coverage.

## Acceptance Criteria

The design is complete when all of the following hold:

- Direct external caching uses `UnifiedRadixCache`, not a backend-specific RadixCache subclass.
- No direct connector path allocates or publishes host indices.
- Full, SWA, Mamba, and registered index-derived sidecars use the same connector operation.
- Loaded data enters the tree only through normal UnifiedTree insertion and component commit hooks.
- TP/CP exact-boundary validity masks and load success are reduced in the tree, with connector-local I/O only.
- In-flight store data cannot be evicted or freed before completion.
- Existing HiCache behavior and tests remain unchanged.
- Direct Mooncake MHA, MLA, hybrid-state, sidecar, and CP round trips pass.
