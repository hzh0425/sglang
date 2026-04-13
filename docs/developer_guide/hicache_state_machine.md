# HiCache Node State Machine

UnifiedRadixTree 支持多 Component（Full / Mamba / SWA），每个 Component 在每个节点上
独立维护 device / host 两层数据。本文档定义节点状态机、追踪结构、完整状态转移和核心不变量。

---

## 1. 追踪结构

| 结构 | 类型 | 作用域 | 用途 |
|------|------|--------|------|
| `evictable_device_leaves` | `set` | 全局 | Full 驱动 device 叶驱逐候选 |
| `evictable_host_leaves` | `set` | 全局 | Full 驱动 host 叶驱逐候选 |
| `lru_lists[ct]` | `UnifiedLRUList` | 非 Full component | device LRU 驱逐排序 |
| `host_lru_lists[ct]` | `UnifiedLRUList` | 非 Full component | host LRU 驱逐排序 |

**设计原则：**

- **Full** 通过 leaf set（`evictable_device_leaves` / `evictable_host_leaves`）驱动驱逐，
  以 `last_access_time` 排序，不依赖 LRU。
- **Mamba / SWA** 通过 `lru_lists[ct]` 驱动 device 驱逐，通过 `host_lru_lists[ct]` 驱动
  host 驱逐（`drive_host_eviction`）。

---

## 2. 状态定义

每个 component 在每个节点上有 4 种状态：

| 状态 | `value` (device) | `host_value` | 含义 |
|------|:-:|:-:|------|
| **S0** | None | None | 无数据（component 不适用于此节点） |
| **S1** | ✓ | None | 仅 device |
| **S2** | ✓ | ✓ | device + host（已备份） |
| **S3** | None | ✓ | 仅 host（已驱逐到 host） |

典型复合状态示例（Full, Mamba）：

- 新叶节点: `(S1, S1)` — 两者均有 device 数据
- 已备份叶: `(S2, S2)` — 两者均有 device + host
- 已驱逐叶: `(S3, S3)` — 两者均仅 host
- 内部节点:  `(S1, S0)` 或 `(S2, S0)` — Mamba 只附着在叶节点

---

## 3. D-leaf / H-leaf 定义

```
D-leaf = ∀ct ∈ tree_components: node.component_data[ct].value is not None # 所有 components 存在
       ∧ ∀child: child.component_data[Full].value is None   (无 device 子节点) # 无 device child
       ∧ lock_ref == 0 # 无锁

H-leaf = ∀ct ∈ tree_components: node.component_data[ct].host_value is not None  # 所有 host components 存在
       ∧ ∀child: child.component_data[Full].host_value is None  (无 host 子节点) # 无 host child
       ∧ host_lock_ref == 0
```

**关键规则**：不满足 all-component-complete 条件的 childless 节点是
**D-stranded(Internal)** / **H-stranded**，不进入 leaf set，不会被叶驱逐选中。
这从根本上保证 A1（层原子性）不被违反。

---

## 4. 完整状态转移表

以 `(Full, Aux)` 二元组表示，Aux 代表 Mamba 或 SWA。

| # | 转移 | 状态变化 | `lru_lists[Aux]` | `host_lru[Aux]` | `device_leaves` | `host_leaves` | `evict_size_[ct]` | 级联效果 |
|:-:|------|---------|:-:|:-:|:-:|:-:|:-:|---------|
| T1 | **insert** (新叶) | (S0,S0)→(S1,S1) | insert_mru | — | add(node) | — | +=(F,A) | discard(parent) |
| T2 | **backup** | 叶(S1,S1)→(S2,S2); 内部(S1,S0)→(S2,S0) | — | — | — | — | — | 仅写 host_value，不动追踪结构 |
| T3 | **evict_to_host** | (S2,S2)→(S3,S3) | remove | insert_mru | discard(node) | add(node) | -=(F,A) | parent: 缺Aux→D-stranded，不进device_leaves |
| T4 | **restore** | (S3,S3)→(S2,S2) | insert_mru | remove | add(node) | discard(node) | +=(F,A) | — |
| T5 | **cascade_evict** | (S1,S1)→deleted | remove | — | discard(node) | — | -=(F,A) | tombstone_leaf 向上清理 incomplete 祖先 |
| T6 | **host_evict** | (S3,S3)→deleted | — | remove | — | discard(node) | — | tombstone_leaf 向上清理 incomplete 祖先 |
| T7 | **unevict_on_insert** | (S3,S?)→(S2,S?) 仅Full | — | — | 不进(缺Aux) | — | +=(F) | 随后 insert 在其下建新叶 |
| T8 | **split** | new:(S0,S0)→(S1,S0); child:截短 | Aux清空new | — | 不变(new缺Aux) | — | 重分配 | host_value 按 split_len 切分 |
| T9 | **SWA overlap** | SWA: S0→S1 | insert_mru(SWA) | — | — | — | +=(SWA) | — |

### 转移说明

**T1 insert**: 新叶节点所有 component 一起创建。Full value 由树的 `_add_new_node` 写入，
Mamba value 由 `commit_insert_component_data` 写入。Aux 插入 `lru_lists` 做 device LRU 排序。
`evictable_device_leaves` add(node) 因为所有 component 都在（D-leaf complete）;
discard(parent) 因为 parent 不再是叶。

**T2 backup**: 只通过 `commit_hicache_transfer(BACKUP)` 写入 `host_value`，
**不动任何追踪结构**。host LRU 由后续 T3 evict_to_host 统一管理。

**T3 evict_to_host**: 原子释放所有 component 的 device 数据。Aux 从 `lru_lists` 移除，
插入 `host_lru_lists`（此处统一管理 host LRU）。`device_leaves` discard(node)。
`host_leaves` 检查 H-leaf complete 后 add(node)。
**级联**: parent 子节点减少后，`_update_device_leaf_status(parent)` 检查 all-component-complete;
若 parent 缺 Aux（internal 节点天然无 Mamba），则不进 `device_leaves`，变成 D-stranded。

**T4 restore (load_back)**: 原子恢复所有 component 的 device 数据。Aux 插回 `lru_lists`，
从 `host_lru_lists` 移除（保护 backup 不被 host evict 回收）。
`device_leaves` 检查 D-leaf complete 后 add(node)。

**T5 cascade_evict**: 无 backup 的 D-leaf 直接从树中删除（write_through 策略）。
删除后触发 `_iteratively_delete_tombstone_leaf` 向上检查祖先：
若祖先变成 childless 且 device/host 均不 complete，逐层清理直到遇到 complete 节点。

**T6 host_evict**: H-leaf 从树中删除，释放所有 host 数据。
级联同 T5，`_iteratively_delete_tombstone_leaf` 向上清理 incomplete 祖先。

**T7 unevict_on_insert**: insert 路径遇到 evicted 节点，用新鲜 KV 恢复 Full device。
**只恢复 Full，不恢复 Aux**。节点随后变成 D-internal（insert 在其下创建新叶），
不进 `device_leaves`。Aux 的 host_value 仍保留在 `host_lru_lists` 中。

**T8 split**: 新 parent 从 child 分裂出 Full value 前缀。Mamba 在
`redistribute_on_node_split` 中将 new_parent 的 Mamba 清空（S0），
因为 Mamba 只附着在叶节点。SWA 按长度切分。Full `host_value` 也按 split_len 切分。
新 parent 缺 Aux，不进 `device_leaves`。

---

## 5. Lock 操作（不改 S 状态）

| 操作 | `evict_size_[ct]` | `protected_size_[ct]` | `device_leaves` |
|------|:-:|:-:|:-:|
| `acquire_component_lock` | -= len | += len | discard(node) |
| `release_component_lock` | += len | -= len | _update_device_leaf_status(node) |

Lock 保护节点不被驱逐：lock_ref > 0 的节点不在 `evictable_device_leaves` 中。

---

## 6. 核心不变量

### 树结构不变量

| 编号 | 不变量 | 说明 |
|:----:|--------|------|
| A1 | Leaf 层级一致性 | D-leaf 所有 component 必须有 device value；H-leaf 所有 component 必须有 host_value；Internal 节点允许 Aux 独立缺失（tombstone） |
| A2 | Tombstone 规则 | Full 不允许 tombstone（Full 是 tree 骨架）；Aux 仅在 internal 节点允许 tombstone；Leaf 上任一 component 缺失则该层整体清空 |
| A3 | Tombstone 级联 | `_iteratively_delete_tombstone_leaf`: childless 节点只要 device 或 host 任一层 complete 就保留，两层都不 complete 才删除并继续向上级联 |
| A4 | Backup 连续性 | 已备份节点从 root 起形成连续前缀（write-through 模式下不允许间隙） |
| A5 | Lock 层级 | Full/SWA 是路径锁（Full 锁节点到 root，SWA 锁 window），Mamba 是点锁（仅锁当前节点）；`full_lock >= mamba_lock`；任何 lock > 0 的节点不可驱逐 |

### 追踪结构不变量

| 编号 | 不变量 | 说明 |
|:----:|--------|------|
| INV-1 | Aux S1/S2 ↔ `lru_lists[ct]` | Aux 有 device value 当且仅当在 `lru_lists[ct]` 中 |
| INV-2 | Aux S3 ↔ `host_lru[ct]`; S2 不在 | Aux host-only 当且仅当在 `host_lru_lists[ct]`；S2 **不在** host LRU |
| INV-3 | D-leaf ↔ `device_leaves` (A1) | all-component-complete + childless + unlocked ↔ 在 `evictable_device_leaves` |
| INV-4 | H-leaf ↔ `host_leaves` (A1) | all-component-complete + childless + unlocked ↔ 在 `evictable_host_leaves` |
| INV-5 | LRU 互斥 | 同一 Aux component 同一 node 不同时在 `lru_lists` 和 `host_lru_lists` |

---

## 7. 已知问题

### 7.1 ~~Full 参与 LRU~~ (已修复)

**位置**: `full_component.py` `drive_eviction`

**问题**: Full 的驱逐策略原本在 HiCache/非 HiCache 模式下有两条分支：
- HiCache: 用 `evictable_device_leaves` + `last_access_time` heap
- 非 HiCache: 用 `lru_lists[Full].get_leaf_lru_no_lock()`

Full 不应依赖 LRU。`evictable_device_leaves` 在两种模式下均被维护，
`_evict_device_leaf` 在非 HiCache 下自动 fallback 到 cascade 删除。
两条路径的选叶逻辑等价（LRU 序 ≈ `last_access_time` 序）。

**修复**: 合并为统一的 heap 路径，删除 LRU 分支。

### 7.2 Mamba BACKUP commit 违反 INV-2

**位置**: `mamba_component.py` `commit_hicache_transfer(BACKUP)`

**问题**: S1→S2 时执行 `host_lru_lists[ct].insert_mru(node)`，将 S2 节点插入 host LRU。
这违反 INV-2（S2 不应在 host LRU），可能导致 `drive_host_eviction` 回收正在 device 使用的
backup。

**修复**: 删除该行。host LRU 插入统一由 T3 `_evict_to_host` 负责。

### 7.3 `_update_device_leaf_status` 未检查 all-component-complete (INV-3)

**位置**: `unified_radix_cache.py` `_update_device_leaf_status`

**问题**: 当前只检查 Full value 和子节点，不检查其他 component 是否存在。
子节点被 evict_to_host 后，父节点可能以缺少 Mamba 的状态进入 `evictable_device_leaves`，
违反 INV-3 / A1。

**修复**: 增加 all-component-complete 检查。

### 7.4 `_update_host_leaf_status` 用 `any` 代替 `all` (INV-4)

**位置**: `unified_radix_cache.py` `_update_host_leaf_status`

**问题**: `node.backuped` 定义为 `any(cd.host_value is not None)`。
只要任一 component 有 host_value 就算 backuped，但 H-leaf 要求所有 component 的
host_value 都存在。这可能导致不完整节点进入 `evictable_host_leaves`，违反 INV-4。

**修复**: H-leaf 判断应改为检查所有 component 的 host_value 是否都存在。

### 7.5 `sanity_check` 验证逻辑与新定义不一致

**位置**: `unified_radix_cache.py` `sanity_check`

**问题**:
- device_leaves 验证只检查 Full，未检查 all-component-complete（与 7.3 同源）
- A7 (host_leaves) 验证使用 `node.backuped`（any），未用 all-complete（与 7.4 同源）

**修复**: 与 7.3/7.4 同步修复。

### 7.6 `_evict_to_host` host_lru `in_list` 防御性检查

**位置**: `unified_radix_cache.py` `_evict_to_host`

**问题**: T3 时 aux component 插入 host_lru 前做 `if not host_lru.in_list(node)` 检查。
若 7.2 修复后，S2→S3 转移时节点不可能已在 host_lru，该检查可简化为 assert。

**修复**: 修复 7.2 后，将 `if not in_list` 改为 `assert not in_list` + 直接 `insert_mru`。
