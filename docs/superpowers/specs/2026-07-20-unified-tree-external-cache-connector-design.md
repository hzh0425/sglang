# UnifiedTree 外部缓存 Connector 设计

日期：2026-07-20

## 概述

为 `UnifiedRadixCache` 增加直连外部缓存的 Connector，不把外部系统建模成 HiCache L2 Host 内存。Connector 在设备 Pool 与 Mooncake 等外部缓存之间直接传输 Full KV 和 Hybrid Component 状态。现有 HiCache 继续保留 Device/Host/Storage 路径。

两条路径共用现有传输抽象：

- `PoolTransfer`
- `PoolTransferResult`
- `PoolHitPolicy`
- `SidecarPoolSpec`
- Component 的 prepare/commit/abort Hook

`UnifiedRadixCache` 负责 Tree 变更、Component Lock、操作生命周期和 TP/CP 一致性。Connector 只执行当前 Rank 的查询与 I/O，不接收 Process Group 或 Tree Node。

版本 1 支持 Full KV、SWA、Mamba，以及现有由其他 Pool 索引派生的 Sidecar。直连 Connector 与 HiCache 互斥。PP、异步直连 Load、独立索引 Sidecar，以及运行时挂载或卸载 Connector 均不在本版本范围内。

## 目标

1. 定义一套后端无关的 Connector 接口，能力与 Mooncake 直连 Radix Cache 实现相近。
2. 支持 UnifiedTree Hybrid State：Full KV、SWA、Mamba 和 Sidecar Pool。
3. 尽量兼容现有 HiCache Controller、Storage Backend、Component 和传输契约。
4. 在 Tree 层管理 TP/CP 命中一致性、完成顺序和缓存发布。
5. 不向 `UnifiedTreeNode` 引入外部缓存概念；直连路径中的本地 Tree 只表示设备常驻状态。
6. 外部 Load 后仍保持 UnifiedTree 正常的 Insert、Split、LRU、Lock 和 Eviction 不变量。

## 非目标

- 同时运行 HiCache 和直连外部 Connector。
- 把 Mooncake DRAM 与 SSD 建模成两个独立的 SGLang Tier。
- 支持 PP 的直连外部缓存。
- Layer-wise 或异步的 External-to-Device Load 重叠。
- 不带 `indices_from_pool` 的 Sidecar；版本 1 仅支持现有索引派生模型。
- 运行时挂载或卸载 Connector。
- 任意 Storage Tier 组成的通用图。
- 将 Mooncake Replica 放置暴露给调度逻辑。

## 术语

- **Local/L1**：SGLang 持有的设备 Pool。
- **外部缓存**：Mooncake 一类自行管理内部内存与 SSD 放置的系统。
- **Anchor Pool**：Full KV，对应 `PoolName.KV`；其 Page Prefix 长度定义逻辑缓存前缀。
- **Component Pool**：UnifiedTree Component 持有的状态，例如 SWA 或 Mamba。
- **Sidecar Pool**：通过 `SidecarPoolSpec` 注册的附加状态，可以从 Anchor Pool 派生索引。
- **Logical Key**：Tree/Component 层生成的、与后端无关的 Page Hash。
- **Shard Identity**：Connector 用于隔离物理对象 Key 的 TP/CP Rank 坐标。

## 总体架构

```text
调度器
    |
    v
UnifiedRadixCache
    - Radix Tree 与 Node Lock
    - 正常 Insert/Split/Eviction
    - 传输编排
    - TP/CP 一致性
    |
    +--> TreeComponent: Full / SWA / Mamba
    |        |
    |        +--> PoolTransfer 描述符
    |
    +--> 展开 SidecarPoolSpec
    |
    +--> 现有 HybridCacheController
    |        L1 <-> Host <-> Storage
    |
    `--> 新 CacheConnector
             L1 <-> 外部缓存
```

直连路径不创建 Host Pool，不设置 `ComponentData.host_value`，不维护 Host LRU，也不插入仅 Host 常驻的 Node。Mooncake DRAM 与 SSD 仍是 Mooncake 内部实现细节。

## 核心设计决策

### 保持现有 HiCache Transport 独立

版本 1 不把 `HybridCacheController` 改造成 Connector。它继续管理 Host 分配、D2H/H2D Stream、Host Release Queue、Storage Queue 和 Layer Ready Event；直连 Connector 不承担这些职责。

两种 Transport 都消费 Component 生成的 `PoolTransfer` 描述符，因此无需向直连路径伪造 Host Index，也能获得接口兼容性。

### 本地 Tree 一致性是已有不变量

TP/CP Rank 上的 UnifiedTree 具有相同的逻辑拓扑、Component 有效状态和本地匹配边界。直连方案直接复用该不变量，不为 Local Match 增加 Boundary Mask 或额外 Collective。

新的分布式协调只覆盖可能产生差异的阶段：Remote Lookup 候选、共同候选的精确 Load 结果，以及异步 Store 完成顺序。

### Full KV 继续作为 Anchor Pool

当前 HiCache Storage 路径以 Full KV 为 Prefix Anchor，其他 Pool 作为该前缀的约束。版本 1 保持这一模型：

- Tree 构建 `PoolName.KV` Transfer。
- 非 Full Component 添加各自的 Transfer。
- 从这些 Transfer 展开已注册 Sidecar。
- 可用前缀是所有必需 Pool 都接受的 Full KV 前缀。

这样可以避免大范围重写 `batch_exists_v2()` 语义。

### Lookup 和 Load 同步，Store 异步

这与 Mooncake 直连分支及调度器现有的两阶段 Match/Load 契约一致：

- `lookup()` 在 `match_prefix()` 完成前同步返回。
- `load()` 在已加载前缀发布到 Tree 前同步返回。
- `store_async()` 返回操作 Handle。
- `poll()` 上报已完成 Store。

以后可以在不改变 Component 描述符的情况下增加异步或 Layer-wise 直连 Load，但不属于本次设计。

## 共享传输类型

现有结构继续作为通用传输格式：

```python
@dataclass
class PoolTransfer:
    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[list[str]] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    nodes_to_load: Optional[list[Any]] = None
    indices_from_pool: Optional[PoolName] = None


@dataclass
class PoolTransferResult:
    kv_hit_pages: int
    extra_pool_hit_pages: dict[str, int]
```

直连 Connector 使用 `device_indices` 和 `keys`，绝不填充 `host_indices`。

直连 Lookup 沿用现有 `keys` 与 `hit_policy` 语义。Connector 返回当前 Rank 能够完整恢复的最长候选后缀；Tree 达成 TP/CP 候选边界一致后，再为该单一精确边界重新构建 Load 描述符。

对于 Load 和 Store，Transfer 还包含目标或源 `device_indices`；此时 Tree 已经选定一个精确边界。

## Pool I/O 适配器

Mooncake 分支当前在 `MooncakeConnector` 内把 MHA/MLA KV Slot Index 解析成原始多 Buffer 指针。Hybrid State 要求 SWA、Mamba 和 Sidecar Pool 也具备同样能力，因此用 Pool Adapter 隔离物理 Pool Layout。

```python
@dataclass(frozen=True)
class BufferSpan:
    address: int
    size: int


class PoolIOAdapter(Protocol):
    @property
    def format_id(self) -> str:
        """用于隔离物理 Key 的稳定 dtype/layout/version 指纹。"""

    def registerable_buffers(self) -> list[BufferSpan]:
        """返回 Connector 必须注册的完整内存分配。"""

    def resolve_objects(
        self,
        indices: torch.Tensor,
        object_count: int,
    ) -> list[list[BufferSpan]]:
        """为每个逻辑对象 Key 返回一组多 Buffer Span。"""
```

这里的 “Object” 有意设计得比 KV Page 更通用：对 Full/SWA，它表示一个 KV Page；对 Mamba，它表示一个 Checkpoint Entry。`resolve_objects()` 必须严格返回 `object_count` 个条目，与 Transfer 中的 Key 一一对应。

每个 Connector 维护 `dict[PoolName, PoolIOAdapter]`。任一 Transfer 目标缺少 Adapter 时，初始化直接失败。

对 Sidecar Transfer，`indices_from_pool` **只选择索引来源**。Connector 仍必须使用目标 Sidecar 自己的 Adapter 解析这些索引：

```text
transfer.name = DEEPSEEK_V4_C4
transfer.indices_from_pool = KV

indices  <- KV transfer.device_indices
buffers  <- adapters[DEEPSEEK_V4_C4].resolve_objects(indices, len(keys))
```

如果使用 KV Adapter，会访问错误的物理 Buffer。

首批 Adapter 包括：

- 从 Mooncake 当前 GPU Object Metadata Helper 中抽取的 MHA/MLA Full KV Adapter。
- SWA Pool 适配器。
- Mamba 状态 Pool 适配器。
- Hybrid Pool Assembler 为现有索引派生 Sidecar 提供的目标 Pool Adapter。

## Connector 接口

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
    ) -> PoolTransferResult:
        """返回当前 Rank 能完整恢复的最长候选后缀。"""

    def load(
        self,
        transfers: list[PoolTransfer],
        key_context: CacheKeyContext,
    ) -> ConnectorLoadResult:
        """将外部对象直接加载到 Device Pool 目标位置。"""

    def store_async(
        self,
        transfers: list[PoolTransfer],
        key_context: CacheKeyContext,
    ) -> object | None:
        """提交 Store 并返回 Handle；被拒绝时返回 None。"""

    def poll(self, wait: bool = False) -> list[ConnectorCompletion]:
        """返回已完成且已受理的 Store；wait=True 等待所有本地 Handle。"""

    def close(self) -> None:
        """排空任务、注销 Buffer，并释放后端资源。"""
```

Connector 不得接收 `Req`、`UnifiedTreeNode`、Lock 参数或分布式 Process Group。Store Handle 表示 Connector 操作，而不是 Request ID。

`lookup()` 只负责提出候选长度，其中 `PoolTransferResult.kv_hit_pages` 已按当前 Rank 上所有必需 Pool 的 Policy 截断。Tree 将各 Rank 的候选转换为绝对边界，并通过 TP/CP `MIN` 得到共同候选。

Lookup 不返回后端私有的 Load Handle。Tree 针对共同候选重新构建全新的 Load 描述符；实际 `load()` 结果就是该精确边界的最终校验。任一 Rank 缺少 Mamba/SWA 等精确状态时，`load_ok` 一致性会让所有 Rank 一起 Abort，避免部分 Insert。

### Connector Key 作用域

Component 和 Tree 通过 `PoolTransfer.keys` 生成 Logical Object Hash。即使关闭 HiCache Storage，直连路径也通过共享 Helper 从 `RadixKey` 计算 Full Page Hash，不依赖 `UnifiedTreeNode.hash_value` 是否填充。

Tree 使用所有后端一致的稳定序列化规则，为 LoRA/Cache Salt 计算 `extra_key_digest`。Connector 将每个 Logical Key 转换为等价的 Physical Key：

```text
<connector namespace>/<extra_key_digest>/tp<TP_RANK>-of-<TP_SIZE>/
cp<CP_RANK>-of-<CP_SIZE>/<pool name>/<adapter format_id>/<logical key>
```

Connector Namespace 至少包含：

- Model Identity 与 Revision
- KV/State dtype 与 Layout Identity
- Page 大小
- Connector Key 格式版本

`CacheKeyContext` 提供无法从 `PoolTransfer` 恢复的 Request Namespace 和 Rank 坐标。TP 与 CP Rank 可能持有同一个 Token Page 的不同切片，因此这两组坐标都是必需的。版本 1 不支持 PP。

## Component 接口

Component 通过带类型的 Request Object 使用统一的中立 Transfer Facade。外部 Lookup 不要求目标 Node，而是显式接收现有本地 Anchor 和 Logical Key 范围。

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
    local_prefix_pages: int
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

Request Dataclass 中所有 `*_pages` 字段都是从 Root 开始计算的绝对前缀边界。`ExternalLookupRequest.full_page_keys` 覆盖 `[start_pages, candidate_pages)`，`ExternalLoadRequest.agreed_full_page_keys` 覆盖 `[local_prefix_pages, agreed_pages)`。Component 在 `ExternalLoadRequest` 阶段根据现有 Local Tree 只为缺失状态构建并 Stage Transfer，不生成整段 Boundary Mask。

`ExistingHiCacheTransferRequest` 是对 `build_hicache_transfers()` 和 `commit_hicache_transfer()` 现有参数的 Adapter，不改变 HiCache 语义。

兼容关系如下：

```text
UnifiedTree 的 HiCache 调用点
    -> build_cache_transfers(ExistingHiCacheTransferRequest)
    -> 默认 Facade 委托给现有 Component build_hicache_transfers()

UnifiedTree 的直连调用点
    -> build_cache_transfers(External*Request)
    -> Component 处理新的 Typed Request
```

Commit 同样遵循该 Adapter 规则。现有 Component 可以逐个迁移，而无需改变当前 HiCache 路径。

对 `ExternalLoadRequest`，内存分配所有权必须明确：

- Tree 持有 Staged Full KV 目标。
- SWA 和 Mamba Component 持有它们在 `build_cache_transfers()` 中分配的目标。
- 现有 Sidecar 从所属源 Pool 派生索引，不独立分配 Slot。
- 每个返回的 Staged Component Transfer 必须且只能收到一次 Commit 或 Abort。
- Commit 只把已达成一致的状态挂到最终 Node，并释放未使用的 Tail。
- Abort 释放所有 Staged Component 目标，不挂载任何状态。

直连恢复为 `InsertParams` 增加 `EXTERNAL_STAGED` Component Mode。它是结构插入模式，而不是第二套 Tree Insert 实现：

- 仍然执行正常的 Radix 遍历、Split、Overlap 处理、Full KV 插入和 Node 创建。
- 仍然执行 Component Split Hook，因为已有 Component Data 必须跟随拓扑变化。
- 对恢复区间，SWA/Mamba 基于 Request 的 Recovery 和 Finalization Hook 不分配、不释放、也不挂载状态；新 Component Data 初始为 Tombstone。
- Insert 得到 `InsertResult.last_device_node` 后，`commit_cache_transfer()` 沿受影响路径挂载已 Stage 的状态。
- Tree 在结构插入前预校验所有 Staged Transfer。Commit 不再执行分配或 I/O，并被视为保持不变量、不会失败的操作。
- 结构插入与所有 Component Commit 在调度器线程同步执行；只有全部 Commit 和 Evictability 更新完成后，才发布 Match Result 和 Cache Event。

这样既保留 UnifiedTree 正常的 Split 与 Insert 机制，又为恢复操作提供单一发布边界。非预期的 Commit 不变量错误是致命错误，不能转换成部分发布的 Cache Hit。

### 外部 Component 行为

| 状态 | Lookup Policy | Load 行为 | Store 行为 |
|---|---|---|---|
| Full KV | Anchor，所有 Page | Tree 分配 Full Device Slot | Tree 收集 Root-to-Node Full Slot |
| SWA | 尾部 Page | Component 分配并恢复 SWA Slot 与 Mapping | Component 暴露有效 Tail Window |
| Mamba | 尾部 Page | Component 分配一个适用的 Checkpoint Slot | Component 暴露已存后缀对应的 Checkpoint |
| 索引派生 Sidecar | 跟随源 Pool | 目标 Adapter 使用源 Pool Index 写入 | 目标 Adapter 使用源 Pool Index 读取 |

Mamba 与 SWA 的 Staged State 在结构插入返回最终目标 Node 后、恢复结果发布前挂载。因此 UnifiedTree Insert 必须填充 `InsertResult.last_device_node`。

### Lookup 与 Load 结果语义

Connector Lookup 先提出最长候选边界，Tree 在 TP/CP 上取共同候选。随后针对该单一边界执行 Load；Load 的逐对象结果负责精确校验 Mamba Checkpoint、SWA Tail Window 和 Sidecar State。

| 操作 | Policy/Pool | 结果解释 |
|---|---|---|
| Lookup | Full KV Anchor | 只有直到边界 `b` 的所有 Full Object 都存在时，`b` 才在外部有效。 |
| Lookup | `ALL_PAGES` | 只有从搜索 Base 到 `b` 所需的所有 Component Object 都存在时，`b` 才有效。 |
| Load | 所有 Pool 和 Policy | 只加载共同候选边界 `b` 所需的精确对象。版本 1 使用事务语义：每个 TP/CP Rank 上的所有请求对象都必须成功；任意失败都会中止整个外部 Load，且不发布新前缀。 |
| Store | 所有 Pool 和 Policy | 允许后端部分失败，但操作需要上报失败；后续 Lookup 重新计算可用公共前缀。 |

`PoolTransferResult` 用于提出 Rank Local 候选长度；`ConnectorLoadResult.successes` 用于共同候选的精确 Load，并且每个请求对象必须对应一个 Boolean。版本 1 在 Load 失败后不尝试继续搜索更短前缀，因为 SWA/Mamba Tail State 与选定边界绑定。

TP/CP Lookup 达成一致后，Component 根据 `ExternalLoadRequest.agreed_full_page_keys` 重新构建 Load Transfer，绝不复用更长 Lookup 结果对应的 Rank Local 描述符或后端状态。

## Tree 状态

直连路径只向 `UnifiedRadixCache` 增加操作状态：

```python
pending_external_matches: dict[str, PendingExternalMatch]
ongoing_external_stores: OrderedDict[int, OngoingExternalStore]
external_store_sequence: int
```

`PendingExternalMatch` 包含以下快照：匹配的 Logical Key、Key Context、一致的本地前缀边界、Full Device Frontier，以及共同 Remote 目标边界。新的 Load Request 从这个不可变范围派生，不保留 Connector 私有 Lookup 状态或 Rank Local 描述符。

`OngoingExternalStore` 以 Tree Sequence Number 为 Key，保存可选的本地 Connector Handle、本地完成/成功状态、最终 Tree Node，以及保护该 Node 时返回的精确 `DecLockRefParams`。因此，即使本地提交被拒绝，也必须占据对应的 Sequence 位置。

Node 不增加外部 Residency Bit。后续 Lookup 始终是外部状态的权威来源，因此也能容忍不同 Rank 的 Store 部分失败。

## 读路径

### Match 阶段

1. 执行正常的 UnifiedTree 本地 Match。
2. 直接使用现有 UnifiedTree 不变量：TP/CP Rank 的 Local Tree 具有相同逻辑状态，因此得到相同的 `local_prefix_pages`。这里不生成 Boundary Mask，也不增加 Local Match Collective。
3. 如果本地前缀已覆盖候选边界，直接返回本地命中。
4. 为 `[local_prefix_pages, candidate_pages]` 构建 Full 和 Component 的 `LOOKUP_EXTERNAL` 描述符，展开 Sidecar，并在每个 Rank 调用本地 Connector `lookup()`。
5. 将每个 Rank 返回的后缀长度转换为绝对候选边界，通过 TP/CP `MIN` 得到 `agreed_pages`。
6. 如果 `agreed_pages <= local_prefix_pages`，按 Remote Miss 返回。
7. 每个 Rank 按 `req.rid` 保存相同的本地 Base 和 Remote 目标边界。Load 阶段再为这个共同边界构建全新的精确描述符。
8. 返回本地 Device Index，并附带公共的 `external_hit_length = (agreed_pages - local_prefix_pages) * page_size`。

`MatchResult`、`Req` 和 Load 参数增加 `external_hit_length`。由于 Local Tree 状态一致，且 Remote 候选边界经过 TP/CP 一致性，所有 Rank 上的 `external_hit_length > 0` 结果一致，可以安全驱动 `Req.needs_external_load()`。直连路径不复用 `host_hit_length`、`last_host_node` 或任何 Host-Only 语义。

### Load 与发布阶段

1. 消费 `pending_external_matches[req.rid]`。
2. 初始化 `local_load_ok = 1`。仅为现有 Full Device Frontier 之后的部分分配 Full Device Slot；因此 Full 已经本地常驻时，Aux-Only Load 不包含 Full 目标 Transfer。
3. 每个 Component 针对 `agreed_pages` 重新判断本地状态，只为缺失部分构建新的 `LOAD_EXTERNAL` Transfer 并 Stage 相应 Device Slot，然后展开 Sidecar。
4. 任一本地分配或构建步骤失败时，设置 `local_load_ok = 0`，保留所有已成功 Stage 的资源供 Abort 使用，跳过本地 Connector I/O，但不能提前返回。
5. 否则，仅当该 Rank 存在缺失对象时调用 Connector `load()`。无需本地 I/O 的 Rank 保持 `local_load_ok = 1`。
6. 校验结果数量、对象成功状态和 Staged State 是否可以挂载。任意异常或非法结果都将 `local_load_ok` 设为 0。
7. 无论是否完成分配或调用 Connector，每个 Rank 都通过 TP/CP 对 `local_load_ok` 执行 `MIN`。
8. 如果一致结果为 0，对每个本地 Staged Component Transfer 调用 Abort，释放所有新分配的 Full Slot，把调度器可见前缀恢复到 `local_prefix_pages`，且不发布任何新缓存状态。
9. 以 `EXTERNAL_STAGED` Mode 调用正常 UnifiedTree Insert 或 No-Op Insert，保证一致目标边界存在。将 `prev_prefix_len` 设置为 `min(full_device_pages, agreed_pages) * page_size`，保留对应的现有 Full Index，只为缺失后缀提供新加载的 Full Slot。
10. 填充 `InsertResult.last_device_node`，按确定的 Component 顺序提交 Staged Component 与 Sidecar State，刷新 Evictability，然后发布一致边界。
11. 向调度器返回新变为可用的 Full Index。其中既可以包含通过 Aux-Only SWA/Mamba Load 重新暴露的已有 Full Index，也可以包含新加载的 Full Index。

任何路径都不能手动构造或嫁接 Tree Node。

## 写路径

1. 完成正常的 `cache_finished_req()` 准备与 UnifiedTree Insert。
2. 从 `InsertResult.last_device_node` 获取最终目标 Node。
3. 获取该 Node 的 Component Lock，并保存返回的 Release 参数。
4. 构建 Full Root-to-Node `STORE_EXTERNAL` Transfer。
5. 要求非 Full Component 构建 Store Transfer，并展开 Sidecar。
6. 在每个 Rank 分配相同的 Tree Local Sequence Number，提交一个包含全部 Pool 描述符的 Connector Store。
7. 在每个 Rank 插入一个 `OngoingExternalStore` Entry。已受理的 Rank 保存 Connector Handle；被拒绝的 Rank 保存 `handle=None`、`local_complete=True` 和 `local_success=False`。
8. 在调度器线程 Poll 已受理 Handle，并更新对应 Sequence Entry。
9. 只有该 Sequence 进入 TP/CP 一致的连续已完成操作前缀后，才释放 Lock。

外部 Store 不要求后端在不同 Pool 间提供原子性。后续 Lookup 会应用所有必需 Hit Policy；任一必需状态缺失时，都会缩短可用前缀。

普通 Eviction 不等待所有 Store。已加锁 Node 本身就不能被驱逐。被拒绝的 Placeholder 只保持锁定，直到公共完成前缀 Poll 能够排空其 Sequence。只有 Reset、Flush 和 Shutdown 才同步排空公共 Store Sequence，并在过程中等待本地已受理 Handle。

## TP/CP 所有权与一致性

分布式协调属于 `UnifiedRadixCache`，不属于 Connector。

### Lookup 一致性

Local Tree 的逻辑状态一致是现有 UnifiedTree 的不变量，因此 Local Match 不新增分布式协议。Tree 只协调 Remote Query：

1. 每个 Rank 执行本地 Connector Lookup，提出一个最长候选边界。
2. Tree 通过 Attention CP Group 和 Attention TP Group 对绝对候选边界执行 `MIN`，得到共同候选。
3. 每个 Rank 保存共同候选，并在 Load 阶段为该单一边界重新构建精确描述符。

这里允许 Lookup 只给出候选。SWA/Mamba 的非单调问题由后续精确 Load 兜底：某个 Rank 缺少共同候选边界对应的 Checkpoint 时，该 Rank 的 `load_ok` 为 0，所有 Rank 一起 Abort，不执行 Insert。

### Load 一致性

每个 Rank 校验每个请求对象的 Boolean 结果以及 Staged State 是否可以挂载。没有本地缺失对象的 Rank 以 `load_ok = 1` 开始；分配或构建失败的 Rank 将其设为 0，并跳过 Connector I/O。所有 Rank 仍必须进入 TP/CP `MIN`。结果为 1 时，所有 Rank 提交完整的一致前缀；结果为 0 时，所有 Rank Abort 本地所有 Staged 目标。版本 1 不支持直连 Load 的部分发布。

### Insert 发布一致性

`load_ok` Collective 同时作为 Insert 的发布屏障。只有它为 1 时，所有 Rank 才进入 Insert，并满足以下条件：

- 使用同一个 `agreed_pages` 和相同的 Logical Key 范围。
- 使用相同的 `prev_prefix_len` 和相同的 Component/Sidecar 提交集合。
- `InsertParams` 的逻辑字段一致；只有 Device Index 等物理地址允许按 Rank 不同。
- 所有 Staged State 已完成形状、数量和挂载位置校验，Insert 后的 Component Commit 不再执行可能失败的分配或 I/O。
- 结构 Insert、Component Commit 和 Evictability 更新在调度器线程中作为一个不可见的发布区间执行，中间状态不返回给 Scheduler。

由于 Insert 前的 Local Tree 逻辑状态一致、目标边界一致、发布计划一致，正常 Insert 会在各 Rank 产生相同的逻辑 Tree 状态。任何 Insert/Commit 不变量错误都按致命错误处理，不能让部分 Rank 继续服务。

### Store 完成一致性

Store 按提交顺序获得 Tree Local 单调递增 Sequence Number。每次 Event Poll 时：

1. 每个 Rank 统计 Queue Head 连续完成的操作数。
2. Tree 通过 TP/CP 对该数量执行 `MIN`。
3. 每个 Rank 按相同顺序精确排空该数量的操作。
4. 完成成功状态参与 Reduce 以供 Metric 使用；即使 Storage 失败，操作完成后也要释放 Lock。

Rank Local 提交被拒绝时，记录一个立即完成且失败的 Placeholder，而不是立即出队。这样在其他 Rank 已受理同一操作并仍在执行 I/O 时，各 Rank 仍能保持一致的 Sequence 位置。

后续 Lookup 一致性会阻止部分 Store 的前缀成为 Cache Hit。

Connector 永远不调用 `all_reduce()`，也不接收 Process Group。Mooncake 当前的 `_tp_min()` 逻辑迁移到 Tree。

## Scheduler 与 Base Cache 兼容

面向调度器的改动保持通用：

- 在现有 Host/Component Hit Length 旁增加 `external_hit_length`。
- 增加 `Req.needs_external_load()`，基于所有 Rank 一致的 `external_hit_length`。
- 增加 `UnifiedRadixCache.init_external_load()`；HiCache 继续保留 `init_load_back()`。
- 增加中立的 Poll 入口 `check_cache_events()`。
- 保留 `check_hicache_events()` 作为兼容别名。
- 在 `BasePrefixCache` 增加 `requires_event_polling`；调度器不再检查 Mooncake/FlexKV 专用 Flag。

调度器在构建 Prefill Batch 前派发 External Load。现有 HiCache Load-Back 行为不变。

## 初始化与配置

增加独立于 Radix Cache Registry 的 Connector Registry：

```python
register_cache_connector(name, factory)
get_cache_connector_factory(name)
```

使用中立的 Server Option：

```text
--cache-connector mooncake
```

配置后：

- Tree Cache 构建选择 `UnifiedRadixCache`。
- 正常 Model Inspection 选择 Full、SWA 和 Mamba Component。
- 现有 Hybrid Assembler 注册 Component/Sidecar Pool Adapter。
- Connector 在开始 Serving 前注册所有 Device Allocation。
- 启动时拒绝以下配置：同时启用 HiCache、PP 大于 1、缺少 Pool Adapter、Layout 不受支持、或 Radix Cache 被关闭。

如果必须兼容 Mooncake 直连分支的 `--radix-cache-backend mooncake`，则将其转换为 `--cache-connector mooncake` 并输出弃用警告。Connector 路径可用后，该选项不得再构造 `MooncakeRadixCache`。

## 错误与生命周期规则

### Lookup 失败

Lookup 异常按 0 个 Remote Hit Page 处理，但仍必须进入 TP/CP 候选边界一致性流程。Component Key 不对齐、逐 Pool 结果缺失、配置错误和 Layout 错误属于不变量失败，仍然是致命错误。

### 分配失败

设置本地 `load_ok=0`，保留已成功 Stage 的资源以供 Rollback，同时仍进入 TP/CP Load 一致性流程。公共结果为 0 后，Abort 所有 Staged Component Transfer，释放 Full 目标，移除 Pending Match，并从跨 Rank 公共本地前缀开始按普通 Cache Miss 继续执行。

### Load 失败或部分 Load

设置本地 `load_ok=0`，并在每个 Rank 通过 TP/CP Reduce。调用所有 Component Abort Hook，释放所有新分配的 Full 目标，不发布新前缀。Rank 不得提前返回，因为其他 Rank 可能正在 Collective 中等待。部分 Load 发布推迟到版本 1 之后。

### Store 被拒绝

为该操作 Sequence 记录一个立即完成且失败的 Placeholder。保持 Node 锁定，直到公共完成前缀 Poll 排空该 Sequence，然后释放。Serving 继续执行，但本次状态不会持久化到外部。

### Store 受理后失败

保持 Node 锁定直到操作完成，记录失败 Metric，然后释放。版本 1 不删除本地 Tree State，也不重试。

### Request 中止

只移除 Pending Lookup/Load 标记。已受理 Store 与 Request ID 无关，在 Connector 完成前继续受到保护。

### Reset 与 Flush

排空公共 Store Sequence，等待已受理的本地 Handle，并按顺序消费被拒绝的 Placeholder。随后释放 Lock、清空 Pending Match，并 Reset 本地 Tree。外部对象保持不变。

### 关闭

排空 Store，关闭 Connector，注销 Device Allocation，然后释放 Device Pool。

## 可观测性

增加带 Backend 与 Pool Label 的 Counter：

- Lookup Request、Hit Page 和 Miss
- Load 请求与提交 Page 数
- 部分 Load 与 Load 失败
- 已受理/被拒绝/已完成/失败的 Store
- Pending Store 数量
- TP/CP Lookup 或 Load 不一致

Mooncake Memory 与 SSD 等 Replica Placement 属于 Connector 专用诊断信息，不参与 Tree 调度。

## 测试

### Connector 单元测试

- 稳定 Key Namespace 随 Model、`extra_key`、TP Rank 和 CP Rank 正确变化。
- 每个 Device Allocation 只注册一次。
- MHA/MLA、SWA、Mamba 和 Sidecar Adapter 为每个 Logical Key 返回一组 Object-Major Span。
- Lookup 正确遵循 `ALL_PAGES` 与 `TRAILING_PAGES`。
- Load 针对共同候选边界重新构建精确描述符，不复用第一次 Lookup 的内部状态。
- Load/Store 使用匹配的 Key 与 Buffer Shape。
- 正确上报 Store 拒绝与完成。

### UnifiedTree 单元测试

- 完整本地命中直接跳过 Connector I/O，不增加 Local Match Collective。
- External Lookup 在不创建 Host State 的情况下上报 `external_hit_length`。
- External Load 使用正常 Insert，并保持 Split/LRU/Size 不变量。
- Full+Mamba、Full+SWA 和 Sidecar Load 只有在所有必需状态存在时才 Commit。
- Aux-Only Load 恢复缺失 SWA/Mamba 后，能够暴露已经常驻的 Full Index。
- 即使外部不存在 Full，Load 描述符也允许组合本地 Full 与外部 Mamba/SWA。
- `EXTERNAL_STAGED` Insert 在确定性 Component Commit 前保留辅助 Tombstone，不暴露中间 Match Result。
- 部分 Load 释放所有未提交 Pool Slot。
- Store 锁定精确的最终 Node，并且只解锁一次。
- Request Abort 不得解锁已受理 Store。
- Eviction 跳过 Store-Locked Node，不全局排空 Store。
- Reset 排空 Store，同时保留外部对象。

### 分布式测试

- TP Rank 返回不同 Remote 候选长度时，取最小绝对边界，并在所有 Rank Load 该精确边界。
- 在一致边界上已经精确本地可用的 Rank 跳过 Load I/O，但仍参与其他 Rank 所需的 Load 一致性。
- CP Rank 使用不同 Physical Key，并共同归并 Load 结果。
- 单个 Rank 分配失败时，所有 Rank 都 Abort，且 Collective 不发生 Hang。
- 任意 TP/CP Load 失败都会中止此前一致前缀的完整发布。
- Load 成功后，各 Rank 使用相同逻辑 `InsertParams` 和 Component 提交集合，得到一致的逻辑 Tree 状态。
- 不同 Rank 的 Store Completion Queue 按相同顺序排空。

### 回归测试

- 不启用 Connector 时，现有 UnifiedTree Full/SWA/Mamba 测试通过。
- 现有 Unified HiCache Host/Storage 测试无需修改即可通过。
- 现有调度器 Abort、Timeout 和 Priority Preemption 测试通过通用 Event Cleanup。

### 端到端测试

- Mooncake 直连 MHA TP2 Round Trip。
- Mooncake 直连 MLA TP2 Round Trip。
- Hybrid Mamba 状态往返测试。
- Hybrid SWA 状态往返测试。
- 至少一个已注册 Sidecar Round Trip。
- 在具有 CP-Aware Cache Slice 的模型或配置上完成 CP Round Trip。
- Flush 本地 Tree，从 Mooncake 重新加载，并校验 Cached Token 数量与输出正确性。

## 落地顺序

1. 引入中立的 Component Hook 名称与兼容 Wrapper，保持所有行为不变。
2. 增加 Connector Registry、共享直连 Connector 类型、Pool Adapter 和通用 Scheduler Polling。
3. 通过 UnifiedTree 集成仅 Full 的 Mooncake 直连路径，并删除 `MooncakeRadixCache` 中独立的 Tree 生命周期。
4. 增加 SWA 与 Mamba External Phase，以及 Allocation Rollback 测试。
5. 注册并校验 Sidecar Adapter。
6. 启用 TP/CP 一致性与分布式测试。
7. 对完整受支持 Component Matrix 启用 `--cache-connector mooncake`。

每一步都必须保持现有 HiCache 路径测试通过。只有模型所需的每个 Component 与 Sidecar Pool 都具备 Adapter 和测试覆盖后，才能为该模型启用最终选项。

## 验收标准

满足以下全部条件后，设计才算完成：

- 直连外部缓存使用 `UnifiedRadixCache`，而不是后端专用 RadixCache 子类。
- 直连 Connector 路径不分配也不发布 Host Index。
- Full、SWA、Mamba 和已注册的索引派生 Sidecar 使用同一套 Connector 操作。
- 已加载数据只通过正常 UnifiedTree Insert 与 Component Commit Hook 进入 Tree。
- TP/CP Remote 候选边界与 Load 成功状态在 Tree 中 Reduce，Connector 只执行本地 I/O。
- In-Flight Store 数据在完成前不能被 Evict 或释放。
- 现有 HiCache 行为与测试保持不变。
- Mooncake 直连 MHA、MLA、Hybrid State、Sidecar 和 CP Round Trip 全部通过。
