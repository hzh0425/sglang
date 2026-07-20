# UnifiedTree 外部缓存 Connector 设计（V1）

日期：2026-07-20

## 一句话结论

V1 把一次远端恢复定义为一个不可拆分的 `TransferBundle`：

```text
Full KV + 所有必需 Component + 所有 Sidecar
```

Local Match 仍完全相信 UnifiedTree 的现有一致性，只返回所有 Component 都认可的 `last_device_node`。Remote Query 从这个 Node 之后构建 Bundle，并返回整个 Bundle 可恢复的最长前缀。Remote Load 要么把该前缀需要的全部状态一起加载并发布，要么全部回滚。

V1 明确不支持以下组合：

- 本地 Full + 远端 Mamba/SWA。
- 远端 Full + 本地已有 Component。
- 某些 Pool 成功就发布部分前缀。
- 为 Local Match 生成逐 Pool Mask 或执行额外 Collective。

这会多传输少量已经存在的 Full KV，但显著简化一致性、回滚和 Insert 语义。

## 与现有 L3 Prefetch 的关系

现有 L3 路径已经提供了 V1 所需的大部分语义：

1. `batch_exists_v2()` 先查 Full KV 的连续前缀。
2. 对每个额外 Pool 应用 `ALL_PAGES` 或 `TRAILING_PAGES`。
3. 用所有必需 Pool 的边界截断 `kv_hit_pages`，得到一个 Bundle 级命中长度。
4. Fetch 时先完成 Full KV，再读取 Component/Sidecar。
5. SWA/Mamba 任一必需状态不足时，丢弃整个 Prefetch。
6. 成功后先正常 Insert Full，再通过 Component Commit Hook 挂载 Hybrid State。

新 Connector 复用这些语义，只把传输路径从：

```text
L1 Device <-> L2 Host <-> L3 Storage
```

改为：

```text
L1 Device <-> External Connector
```

Connector 不创建 Host Pool，不设置 `host_value`，也不维护 Host LRU。

## 核心数据结构

继续复用现有 `PoolTransfer`、`PoolTransferResult`、`PoolHitPolicy` 和 `SidecarPoolSpec`。

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

为避免调用方漏掉某个 Pool，增加一个很薄的 Bundle 包装：

```python
@dataclass
class TransferBundle:
    full: PoolTransfer
    extras: list[PoolTransfer]

    def transfers(self) -> list[PoolTransfer]:
        return [self.full, *self.extras]
```

约束如下：

- `full.name` 必须是 `PoolName.KV`。
- `extras` 包含所有必需 Component 和 Sidecar，不包含可选的“尽力加载”状态。
- Query 阶段只填 `keys`、`hit_policy` 和 `indices_from_pool`。
- Load/Store 阶段填 `device_indices`，不填 `host_indices`。
- Sidecar 的 `indices_from_pool` 只表示索引来源；解析 Buffer 时仍使用 Sidecar 自己的 Pool Adapter。
- Query 阶段的 `TRAILING_PAGES` Transfer 只用 `len(keys)` 表示 Tail Window 大小；具体候选 Key 从 `full.keys` 派生。确定 `agreed_pages` 后，Load 阶段再写入该边界对应的精确 Tail Key。

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
class ConnectorCompletion:
    handle: object
    success: bool


class CacheConnector(Protocol):
    def query(
        self,
        bundle: TransferBundle,
        key_context: CacheKeyContext,
    ) -> PoolTransferResult:
        """返回当前 Rank 上整个 Bundle 可恢复的最长后缀。"""

    def load(
        self,
        bundle: TransferBundle,
        key_context: CacheKeyContext,
    ) -> PoolTransferResult:
        """把 Bundle 中的对象直接加载到 Device Pool。"""

    def store_async(
        self,
        bundle: TransferBundle,
        key_context: CacheKeyContext,
    ) -> object | None:
        """异步提交整个 Bundle，返回本地操作 Handle。"""

    def poll(self, wait: bool = False) -> list[ConnectorCompletion]:
        ...

    def close(self) -> None:
        ...
```

Connector 不接收 `Req`、`UnifiedTreeNode`、Process Group 或 Tree Lock。它只处理当前 Rank 的 Key、Buffer 和 I/O。

### Query 结果

`query()` 的语义直接对齐 `batch_exists_v2()`：

- 先计算 Full KV 连续命中页数。
- `ALL_PAGES` 要求目标前缀内每一页都存在。
- `TRAILING_PAGES` 要求目标边界之前最后 N 页存在，N 由 Transfer 的 Key 数量决定。
- `kv_hit_pages` 必须已经被所有 `extras` 截断，是 Bundle 级结果，不是只看 Full 的结果。
- `extra_pool_hit_pages` 保留作诊断和 Component Commit 使用。

### Load 结果

`load()` 可以在后端内部按顺序执行 I/O，但 Tree 只接受完整结果：

```python
load_ok = (
    result.kv_hit_pages == len(bundle.full.keys)
    and all(
        result.extra_pool_hit_pages.get(xfer.name, 0) == len(xfer.keys)
        for xfer in bundle.extras
    )
)
```

任何对象失败都令 `load_ok = False`。已经写入 Device Buffer 的内容仍是未发布的 Staged State，随后统一释放。

## Component 接口

V1 不引入复杂的逐 Pool Mask 或验证对象。Component 只负责三件事：构建 Transfer、成功后挂载、失败后释放。

```python
class CacheTransferPhase(str, Enum):
    # 现有 HiCache Phase 保持不变
    BACKUP_HOST = "backup_host"
    LOAD_BACK = "load_back"
    BACKUP_STORAGE = "backup_storage"
    PREFETCH = "prefetch"

    # 新增直连 Phase
    QUERY_REMOTE = "query_remote"
    LOAD_REMOTE = "load_remote"
    STORE_REMOTE = "store_remote"


class TreeComponent:
    def build_cache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        token_ids: Sequence[int] = (),
        page_hashes: Sequence[str] = (),
    ) -> list[PoolTransfer]:
        ...

    def commit_cache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer],
        *,
        insert_result: Optional[InsertResult] = None,
        transfer_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ...

    def abort_cache_transfer(
        self,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer],
    ) -> None:
        ...
```

返回空列表表示该 Component 对当前范围不要求远端对象；构建失败或分配失败直接抛出明确异常，由 Tree 进入统一 Abort 流程，不再复用 `None`、`[]` 表示多个含义。
为降低改动风险，现有 `build_hicache_transfers()` 和 `commit_hicache_transfer()` 先保留，默认 Facade 将现有 HiCache Phase 委托给旧 Hook。Mamba/SWA 逐步把 Key/Policy 计算抽成内部 Helper，供 HiCache 和 Connector 两条路径复用。

### 各 Component 的 V1 行为

| Component | Query | Load | Commit |
|---|---|---|---|
| Full KV | Tree 构建全部后缀 Page Hash，`ALL_PAGES` | Tree 为整个远端后缀分配 Device Slot | 正常 UnifiedTree Insert |
| Mamba | 以候选边界的最后一个 Hash 查询，`TRAILING_PAGES` | 分配并加载一个 Mamba Checkpoint | 挂到最终 `last_device_node` |
| SWA | 查询候选边界之前的完整 Window，`TRAILING_PAGES` | 分配并加载完整 SWA Window | 沿最终路径挂载并重建 Mapping |
| Sidecar | 跟随源 Pool 的 Key 和 Policy | 从源 Transfer 派生 Device Index | 无独立 Tree Node 状态 |

`QUERY_REMOTE` 不分配 Device Slot；`LOAD_REMOTE` 才分配。每个成功 Stage 的 Component Transfer 必须且只能收到一次 Commit 或 Abort。

## 读链路

```text
match local
    |
    v
所有 Component 一致的 last_device_node
    |
    v
构建 Query Bundle
    |
    v
Connector.query（Rank Local Bundle Hit）
    |
    v
TP/CP MIN 得到 agreed_pages
    |
    v
为 agreed_pages 重建完整 Load Bundle
    |
    v
Connector.load（Full + Component + Sidecar）
    |
    v
TP/CP MIN(load_ok)
   / \
  0   1
Abort  正常 Insert Full -> Commit Components -> 发布
```

### 1. Local Match

1. 执行现有 UnifiedTree `match_prefix()`。
2. 使用所有 Component 已经认可的 `last_device_node` 和对应前缀长度作为远端 Anchor。
3. Local Tree 在 TP/CP Rank 间逻辑一致是现有不变量，因此这里不生成 Mask，也不增加 Collective。
4. 本地已经完全命中时，不调用 Connector。

### 2. Remote Query

1. 将 `last_device_node` 之后的候选后缀按 Page 对齐。
2. Tree 为整个后缀构建 Full KV Query Transfer。
3. 每个激活的 Component 构建 `QUERY_REMOTE` Transfer。
4. 用现有 Sidecar 展开逻辑补齐 Sidecar Transfer。
5. 调用当前 Rank 的 `connector.query(bundle, key_context)`。
6. 将返回的后缀页数转换为 Root 起算的绝对边界。
7. 在 Attention TP Group 和 CP Group 上执行 `MIN`，得到 `agreed_pages`。
8. `agreed_pages` 不超过本地边界时按普通 Miss 处理；否则保存一个很小的 `PendingExternalMatch`。

```python
@dataclass(frozen=True)
class PendingExternalMatch:
    key: RadixKey
    anchor_node: UnifiedTreeNode
    local_prefix_pages: int
    agreed_pages: int
    key_context: CacheKeyContext
```

Pending 状态不保存 Connector 私有 Handle，也不复用第一次 Query 的 Transfer。因为 TP/CP `MIN` 可能改变最终边界，Load 必须按 `agreed_pages` 重新生成 Mamba/SWA 的 Tail Key。

### 3. Remote Load

1. 消费 `PendingExternalMatch`。
2. 为 `[local_prefix_pages, agreed_pages)` 的整个后缀分配 Full Device Slot。
3. 所有必需 Component 为同一目标边界构建 `LOAD_REMOTE` Transfer 并分配 Device Slot。
4. 展开 Sidecar，形成新的完整 Load Bundle。
5. 任一构建或分配失败时，当前 Rank 记 `local_load_ok = 0`，不调用 Connector，但仍参与后续 Collective。
6. 否则调用 `connector.load()`，并按 Bundle 中的全部对象校验结果。
7. 对 `local_load_ok` 在 TP/CP 上执行 `MIN`。
8. 公共结果为 0：所有 Rank 释放新分配的 Full Slot，调用所有 Component Abort，不修改 Tree。
9. 公共结果为 1：使用正常 UnifiedTree Insert 插入 Full KV，再按固定顺序调用 Component Commit，最后才把命中结果返回调度器。

V1 即使发现 Anchor 之后已有 Full-only Node，也仍然请求整个 Full 后缀。正常 Insert 的 Overlap 逻辑会保留 Tree 中已有 Full Index，并释放重复加载的临时 Full Slot。这样 `prev_prefix_len` 始终就是 `local_prefix_pages * page_size`，不需要额外维护 Full Device Frontier。

Insert 需要增加一个很小的 `EXTERNAL_STAGED` 模式，避免普通 Insert 流程重新创建或释放已经 Stage 的 Component State：

- Tree 结构、Split、Overlap、LRU 和 Full KV 插入仍走原路径。
- 新节点上的 Component 先保持 Tombstone。
- Insert 返回最终 Node 后，Component Commit 只挂载已经校验过的 State，不再分配或执行 I/O。
- Commit 完成后重新取得最终 Root-to-Node Device Index，再向 Scheduler 发布；不能直接拼接初始分配的 Full Slot，因为 Overlap 可能已经回收其中一部分。

## TP/CP 一致性

Connector 只做 Rank Local I/O，所有分布式协议都在 UnifiedTree：

### Query

```text
agreed_pages = MIN(rank_local_absolute_bundle_hit)
```

对于 `TRAILING_PAGES`，最长命中对边界不一定单调。某 Rank 能命中更长边界，不代表一定拥有另一个 Rank 选出的较短边界对应的 Mamba/SWA Tail。V1 不增加第三套 Mask 或搜索协议；对共同边界执行精确 Load，若任一 Rank 缺对象，则整次 Load Abort，也不自动重试更短边界。

### Load

```text
global_load_ok = MIN(rank_local_load_ok)
```

所有 Rank 必须进入这个 Collective。分配失败、I/O 失败或结果不完整的 Rank 不能提前返回。只有 `global_load_ok == 1` 时，所有 Rank 才使用相同逻辑 Key、相同目标边界和相同 Component 集合执行 Insert/Commit。

### Store

Store 沿用现有异步 Lock 和顺序完成模型：

1. Tree 锁住最终 `last_device_node`。
2. 构建完整 Store Bundle 并调用 `store_async()`。
3. 各 Rank 为操作分配相同顺序号。
4. Poll 时对连续完成数量取 TP/CP `MIN`，按相同顺序释放 Lock。

后端不需要提供跨 Pool 原子 Store。若只写入一部分，后续 Bundle Query 会因缺少必需 Pool 而缩短或拒绝命中。

## UnifiedTree 需要改什么

### 必需改动

1. 增加 `TransferBundle`、`CacheConnector` 和 Connector Registry。
2. 从现有 HiCache 中抽出共享的 Page Hash、`ALL_PAGES/TRAILING_PAGES` 和 Sidecar 展开逻辑。
3. 给 Component 增加 `QUERY_REMOTE / LOAD_REMOTE / STORE_REMOTE` 构建、Commit 和 Abort 能力。
4. `match_prefix()` 在本地结果之后增加同步 Remote Query，并记录 `external_hit_length` 和 `PendingExternalMatch`。
5. Scheduler 在 Prefill 前调用 `init_external_load()`。
6. 增加 `EXTERNAL_STAGED` Insert 模式，并保证 Component Commit 前不发布中间状态。
7. 增加 Remote Query 边界和 Load 成功状态的 TP/CP Reduce。
8. 将 `check_hicache_events()` 提升为中立的 `check_cache_events()`；保留旧名称作兼容入口。
9. Connector 初始化时为 Full、SWA、Mamba 和 Sidecar 注册 Device Pool Adapter。

### 不需要改动

- 不给 `UnifiedTreeNode` 增加 Remote Residency 状态。
- 不创建 L2 Host Node、Host Index 或 Host LRU。
- 不改变现有 HiCache Controller/Storage 调度流程。
- 不为 Local Match 增加 Component Mask。
- 不支持 Local/Remote 逐 Pool 混合恢复。
- 不支持部分 Load 发布。

## 与 HiCache 的兼容方式

V1 复用“数据契约”，不强行复用“执行器”：

| 能力 | 复用方式 |
|---|---|
| `PoolTransfer` / `PoolTransferResult` | 原样复用 |
| `ALL_PAGES` / `TRAILING_PAGES` | 原样复用 |
| Sidecar 描述 | 原样复用，增加 Device Index 解析 |
| Component Key/Window 计算 | 抽成内部 Helper，两条路径共用 |
| Component Commit 结构 | 沿用现有 Prefetch 的 Insert 后 Commit 模式 |
| L3 `batch_exists_v2()` | 抽出 Bundle 命中算法供 Connector 实现参考或共用 |
| HiCache Controller | 不改，仍负责 Device/Host/Storage |
| Connector | 新执行器，直接负责 Device/External |

这样不会为了直连 Mooncake 而把 L2 概念泄漏到 Connector，也不会让现有 HiCache 路径承担回归风险。

## 最小实现顺序

1. 定义 `TransferBundle` 和同步 `query/load` Connector 接口。
2. 先接入仅 Full KV，打通 Local Match、Remote Query、Remote Load、正常 Insert。
3. 增加 Mamba：一个 Tail Checkpoint，完整 Bundle 成功或回滚。
4. 增加 SWA：一个完整 Tail Window，完整 Bundle 成功或回滚。
5. 复用 `_build_sidecar_transfers()`，补 Device Pool Adapter。
6. 增加 TP/CP Query MIN 与 Load Boolean MIN。
7. 最后接入异步 Store 和统一 Event Poll。

## 必测场景

- 本地完整命中时不访问 Connector。
- Query 的 Full 命中更长，但 Mamba/SWA 较短时，返回 Component 约束后的 Bundle 边界。
- Full、Mamba、SWA 或 Sidecar 任一 Load 失败时，所有 Staged Slot 都释放，Tree 不变化。
- Anchor 之后已有 Full-only Node 时仍执行完整 Bundle Load；Insert 保留已有 Full Index 并回收重复 Slot。
- Load 成功后，Full Insert 与 Component Commit 只对 Scheduler 发布一次。
- TP/CP Rank Query 长度不同时取最小绝对边界。
- 任一 Rank 分配或 Load 失败时，所有 Rank 一起 Abort 且 Collective 不 Hang。
- Mamba/SWA 在共同边界非单调失效时整次 Abort，不回退到更短边界。
- 不启用 Connector 时，现有 UnifiedTree 和 HiCache 测试行为不变。

## V1 验收标准

- 直连路径使用 `UnifiedRadixCache`，不创建后端专用 Tree。
- Local Match 只依赖现有 `last_device_node`，没有 Boundary Mask。
- Remote Query 返回一个被所有必需 Pool 约束后的 Bundle 命中长度。
- Full、Component 和 Sidecar 在 Load 时要么全部成功并 Commit，要么全部回滚。
- TP/CP 的 Query 和 Load 一致性由 Tree 管理，Connector 不持有 Process Group。
- 直连路径不分配 Host State，现有 HiCache 路径保持不变。
